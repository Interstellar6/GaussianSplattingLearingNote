/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// 将 RGB to SH值
// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// 计算二维高斯分布协方差矩阵 2*2
// 参数: 均值 x&y方向焦距 x&Y垂直方向的视野范围 三维高斯分布的协方差矩阵 世界坐标系到相机坐标系到转换矩阵
// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean,
                               float focal_x, float focal_y,
                               float tan_fovx, float tan_fovy,
                               const float* cov3D,
                               const float* viewmatrix)
{
    // 在auxiliary.h中  实现矩阵乘法的函数
	// 将世界坐标系下的均值和转换矩阵相乘， 得到了相机坐标系中的坐标t
    // The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally, considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

    // 限制水平方向和垂直方向在视野范围之内
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
    const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

    // 雅各比矩阵 3*3
	// 作用 直接做了个近似变换 处理映射关系
    glm::mat3 J = glm::mat3(
		focal_x / t.z,           0.0f,       -(focal_x * t.x) / (t.z * t.z),
		   0.0f,             focal_y / t.z,   -(focal_y * t.y) / (t.z * t.z),
           0,                    0,                       0
    );

    // 世界坐标矩阵 从视图矩阵提取九个参数变成3*3的转换矩阵
	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

    // 3*3
	glm::mat3 T = W * J;

    // 由于三维协方差矩阵是一个对称矩阵， 因此只存六个参数即可 现在把这六个参数给还原出来一个完整的三位协方差矩阵
	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

    // \Sigma' = J_{3*3}^T * W_{3*3}^T * \Sigma_{3*3} * W_{3*3} * J_{3*3}
	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;


    // Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
    // 二维对角处加0.3
    // 若高斯点的协方差矩阵的对角线元素比较小，
    // 那么这个高斯点在屏幕上的投影会非常小，甚至小于一个像素 (类似bleeding效果)
    // 所以这里要给他加上一个常数， 使它至少要大于一个像素大小
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;

    // 对称的, 只需要三个就能表达2*2的二维平面协方差矩阵， 其他五维是非齐次坐标的(丢弃即可)
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// 计算三维协方差矩阵的函数 3*3
// 参数: scale缩放因子3*1 mod rot旋转因子4*1 cov3D：要求的三维协方差矩阵3*3
// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// 缩放矩阵
    // Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

    // 旋转因子
	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

    // 旋转矩阵
	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

    // 相机外参矩阵
	glm::mat3 M = S * R;

    // 三维协方差矩阵
	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

    // 将三维协方差矩阵的9个参数压缩为6个参数， 节省存储
	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// 高斯点光栅化的预处理阶段
// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,  // 线程下标的上阈值
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
    // 读取对应线程的下标
	auto idx = cg::this_grid().thread_rank();
	// 线程下标的  上阈值， 超过这个阈值就退出
    if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;  // 线程idx正在处理高斯点的半径
	tiles_touched[idx] = 0;  // 是否接触图块的变量

    // 点的视角
	// Perform near culling, quit if outside.
	float3 p_view;
    // 判断这个点是否在这个相机视角的视锥之内 不是就跳出
    // 思想： 将点转换为相机坐标系中点的坐标
    //       然后将z坐标(点到相机的垂直距离)直接和一个阈值(最近可视平面到相机的距离)比较
    //       如果z坐标小于这个最近平面到相机的距离， 这个点就不应该看到
    // 方法在auxiliary.h
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

    // 计算最终的投影点 将齐次坐标转换为空间坐标系中的点
    // 齐次坐标p_hom = p_原始坐标 * 投影矩阵 将原始坐标投影到屏幕上
    // Bring points to screen space
	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
    // [ x, y, z, w ]^T  -->  [ x/w, y/w, z/w ]^T
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

    // 判断cov3D之前是否已经被计算了, 被算了就直接用， 不然就计算一遍
	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
        // 构建三维三维协方差矩阵
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
        // 每个cov3D有6个参数， 这里就映射到正确的cov3D索引
        cov3D = cov3Ds + idx * 6;
	}

    // 计算得到投影到视图平面后的二维协方差矩阵
	// Compute 2D screen-space covariance matrix
    // cov2D = [ [ cov.x cov.y ]
    //           [ cov.y cov.z ] ]
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

    // 计算反协方差矩阵
	// Invert covariance (EWA algorithm)
    // 行列式的值
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
    // 计算得出的二维协方差的逆矩阵
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
    // 二维椭球的长轴短轴的值
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    // 使用的二维高斯椭球的半径
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
    // 得到点在二维图像上的真实坐标
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
    // 获取一个矩形
	getRect(point_image, my_radius, rect_min, rect_max, grid);
    // 如果这个椭球的面积为0, 那么这个高斯椭球不存在 不用处理
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

    // 检查颜色是否已经被预计算了， 如果没有预计算就算出来， 将SH三个方向的值赋值给RGB
	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

    // 点的深度就是高斯点中心到相机的垂直距离
    // 严格意义上来说这里的点的深度应该是高斯椭球的面到相机的垂直距离， 这里简化了
	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
    // 高斯点的半径
	radii[idx] = my_radius;
    // 投影后二维高斯分布的坐标
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
    // 将逆二维高斯分布和不透明度的数据集成到一个变量内
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
    // 图块是否被触及高斯椭球所在的矩形框 被触及的话就是大于0的， 没有被触及就是等于0的(这个高斯椭球不在栅格之内)
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// 最终渲染光栅化平面
// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
    // 标记正在处理的线程
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
    // 计算水平方向栅格的数量 =  W/BLOCK_X                  {总宽度除以单个栅格的宽度}
    //                       + (BLOCK-1)/BLOCK         {最后一个BLOCK可能是不完全的，需要向上取整}
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    // 标记这个栅格的对角两点 （左下和右上） 最大顶点需要考虑越界
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    // 根据栅格的最小一点以及线程序号 计算这个栅格在线程内的坐标
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	// 标记这个像素的id (从上到下， 从左到右)
    uint32_t pix_id = W * pix.y + pix.x;
    // 浮点数形式变量
	float2 pixf = { (float)pix.x, (float)pix.y };

	// 检查这个像素是否在光栅化平面之内
    // Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// 标记已经完成光栅化操作的像素
    // Done threads can help with fetching, but don't rasterize
	bool done = !inside;

    // range用来统计BLOCK的id
	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	// 处理这些像素还需要的轮次
    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	// 算出还需要处理的BLOCK个数
    int toDo = range.y - range.x;

    /** 处理的像素变量 坐标 以及对应坐标的反协方差矩阵与不透明度 */
	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];



/**
   体渲染公式
  $$
    C=\sum_{i=1}^{\mathbb{N}}T_i \alpha_i c_i

    T_i = \prod_{i-1}^{j-1} {1-\alpha_i}

    \alpha_i = ( 1 - \exp({-\sigma_i * \delta_i }) )

    C = \sum_{i \in \mathbb{N} } c_i \alpha_i \prod_{j=1}^{i-1} ( 1 - \alpha_j )
  $$
 */
    // 做alpha blending时候的T
    // Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
    // 颜色数组RGB
	float C[CHANNELS] = { 0 };

    // 开始在整个栅格内(内涵多个block)进行循环
	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// 如果所有block完成光栅化就跳出循环
        // End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

        // 开始在单个block内进行循环
		// Collectively fetch per-Gaussian data from global to shared
        // 定位进程号
		int progress = i * BLOCK_SIZE + block.thread_rank();
		// 如果进程没有处理完
        if (range.x + progress < range.y)
		{
            // 处理这些进程
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// 处理每个batch
        // 如果还在待光栅化平面以内 并且j还没处理完
        // Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
            // xy: 正在处理的像素坐标     pixf: 二维高斯分布椭圆的中心点
			float2 xy = collected_xy[j];
            // 正在处理的像素与高斯椭圆中心点之间的距离
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
            // 这个高斯椭圆的协方差逆矩阵与不透明度 con_o = [inverse(cov2D), opacity]
            float4 con_o = collected_conic_opacity[j];
            // 根据这个像素点与椭球之间的距离， 计算这个像素点的不透明度 power为对e的指数
            float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// 保证透明度不能大于0.99
            float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
            float test_T = T * (1 - alpha);
			// 这个点处理完了
            if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// 体渲染公式 综合一个像素格上不同深度的二维椭球的颜色， 渲染最终像素的颜色
            // Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

            // 进行这个像素点的下一次渲染
			// Keep track of last range entry to update this pixel.
			last_contributor = contributor;
		}
	}

	// 如果这个像素还在光栅化平面以内， 需要加上背景颜色
    // All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}