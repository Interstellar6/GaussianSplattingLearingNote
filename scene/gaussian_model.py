#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


class GaussianModel:

    def setup_functions(self):

        # 根据旋转缩放矩阵设置协方差
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        # 设置激活函数
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        # 协方差的激活函数
        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0  # 球谐函数阶数， 初始化为0
        self.max_sh_degree = sh_degree  # 最高阶数
        self._xyz = torch.empty(0)  # 椭球位置
        self._features_dc = torch.empty(0)  # 直流分量
        self._features_rest = torch.empty(0)  # 高阶分量
        self._scaling = torch.empty(0)  # 缩放因子
        self._rotation = torch.empty(0)  # 旋转因子
        self._opacity = torch.empty(0)  # 透明度
        self.max_radii2D = torch.empty(0)  # 投影到二维平面的椭球直径
        self.xyz_gradient_accum = torch.empty(0)  # 梯度累计值
        self.denom = torch.empty(0)  # 统计的分母数量
        self.optimizer = None
        self.percent_dense = 0  # 百分比密度
        self.spatial_lr_scale = 0  # 学习率因子
        self.setup_functions()  # 创建激活函数

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)  # 返回激活后的变量， 若要对变量提取处理， 需要使用反激活函数提取出来

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    # 这里的透明度已经是被sigmoid激活过后的，
    # 因此后续操作中如果要得到原本的透明度， 需要对get到的值进行inverse_sigmoid处理
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1  # 迭代球谐函数的阶数+1

    # 从Point Cloud Data中创建数据, 学习率的变化因子
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()  # 保存电云数据， 维度就是点云的点数
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())  # 把RGB转换为球谐系数
        # features 维度就是高斯分布的总数， 3为球谐函数的系数的数量，3就是3个通道 每个球谐函数系数的数量
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()  # 存球谐函数的系数
        features[:, :3, 0] = fused_color  # 第0维为颜色
        features[:, 3:, 1:] = 0.0  # 更高维暂定为0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])  # 打印初始化点的数量

        # 算点云的位置， 点云的点之间最近距离为0.0000001, 来避免点之间不会重合
        # distCUDA2函数在simple_KNN/spatial.cu里面有定义，
        # 用KNN算法计算这个高斯点最近的三个高斯点，并计算出和这三个高斯点的平均距离， 这个可以用来去构建一个初始的高斯球
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # 缩放因子
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        # 旋转张量， 维度为N*4的张量, 第0维设置为1, 其余为0
        # rots就是旋转的四元数q=(w,x,y,z), w是实数空间的值，最终它的角度值系数应该是2*cos(w)，
        # 若将w 设为1, 其他三个维度设为0, 那么最终这个单位四元数的整体值就是0
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        # 由于之前get_opacities()得到的透明度是已经被sigmoid激活过的， 因此这里需要inverse_sigmoid处理
        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 各种参数的初始化
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda") # 最大2D直径，用于后续投影的

    # 训练初始化 初始化了初始化点云中点的稠密度，坐标的累计梯度，球谐函数的分母数量，学习率，优化器等
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 定义不同指标的学习率
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        # 运行微调时xyz坐标的学习率
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    # 更新各个点的xyz坐标参数的学习率
    def update_learning_rate(self, iteration):
        """ Learning rate scheduling per step """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    # ply文件参数内容：
    #   vertex 高斯点的数量
    #   x y z:点坐标
    #   nx ny nz 点的法向量，但是考虑到高斯分布的性质这里都是0就没算
    #   f_dc: 3个 球谐函数的直流分量
    #   f_rest: 45个 球谐函数的高阶分量
    #   3+45刚好对应了三阶球谐函数(3+1)**2 * 3个颜色RGB 总系数的表达
    #   不透明度 1个
    #   scale: 3个 缩放因子
    #   rot： 4个 旋转因子
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    # 保存点云模型
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        # 先把各种参数转移到CPU上， 然后保存到文件里
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # 更新不透明度
    # 这里把不透明度限制在0.01以内(不会直接设置成0)
    # 在一定次数内要重置不透明度就设置在0.01以内(原论文)
    # 将(sigmoid激活后的)透明度提取出来后进行inverse sigmoid处理，
    # 然后将更新的不透明度添加到优化器里，最后转化成张量, 赋到原对象内
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # 加载点云文件
    def load_ply(self, path):
        plydata = PlyData.read(path)

        # 加载坐标值与不透明度
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # 将features的参数值加载到features的矩阵内
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # 加载参数名称？
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))

        # 判断所有系数是否都已经加载到文件内
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # 设置缩放矩阵
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 设置旋转矩阵
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 设置其他各种参数
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    # 将张量替换到优化器中，用于后续迭代
    # 为什么要从张量转换过来暂存到优化器内？
    # 张量在训练时候是可能发生变化的
    # 优化器内存储的动量与二次动量(优化后的状态, 代表了梯度进行的方向)是不能变化的
    # 这样才能保证参数在优化过程中平滑地迭代
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # 保存张量的数据(现有状态)
                stored_state = self.optimizer.state.get(group['params'][0], None)
                # 保存张量对应的动量(梯度方向)的值
                # zeros_like和ones_like都是创建和原来张量维度一致的全一矩阵或者全0矩阵
                stored_state["exp_avg"] = torch.zeros_like(tensor)  # 动量 代表了梯度进行的方向
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)  # 二次动量

                # 删除优化器中原来的值()
                del self.optimizer.state[group['params'][0]]
                # 在优化器中传入张量现有的状态
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                # 保存可以优化的张量的现有状态
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 重置优化器，通过设置一个mask掩码，只选择保留优化器中需要保留的那个状态，其余的全部重置为0
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # 以掩码形式选择性保留需要优化的张量(删掉多余的)
    def prune_points(self, mask):
        # 通过按位取反得到需要保留的参数的掩码
        valid_points_mask = ~mask
        # 通过掩码去选中可优化的张量
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # 从可优化张量数据中， 赋值给对象
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    # 去创建新的张量，并且存到优化器类 todo
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # 创建和初始化一个新高斯点 用于自适应密度控制
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation):
        # 要添加的属性
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        # 将使用上面的有效掩码，只将所需数据添加到可优化张量类， 用来创建一个初始化这个新的高斯点
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 初始化这个新高斯点的其他属性
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # 分裂高斯点 用于自适应密度控制
    # 参数： 梯度 梯度阈值 场景范围 特定常数为2
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # 读取高斯点的总数
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        # 根据高斯分布总数创建一个新的张量， 这个张量用来存储每个高斯点的梯度
        padded_grad = torch.zeros((n_init_points), device="cuda")
        # 根据梯度先扩展一个维度用来存储掩码， 根据梯度是否大于阈值来标记掩码， 筛选出需要分裂的那些高斯点
        # 筛选思想: 高斯点太大了就分裂！
        # 筛选条件: 根据高斯分布的缩放因子中最大的一个维度的值 > 场景的范围 * 对应的比例因子  (将这些点用掩码标记出来)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = (torch.logical_and(selected_pts_mask,
                                torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent))

        # 从这里开始创建新的高斯点
        # 新的两个点高斯分布的标准差 就等于 原本需要分裂的高斯点的标准差  因为是两个点 重复一次
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        # 新高斯点的均值规定为全0
        means = torch.zeros((stds.size(0), 3), device="cuda")
        # 新的采样点的位置由刚才的两个参数组成的正态分布中取随机数
        samples = torch.normal(mean=means, std=stds)
        # 根据原高斯点的旋转矩阵来创建新高斯点的旋转矩阵， 并赋值到两个点上
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        # samples.unsqueeze((-1))是新高斯点的形状
        # torch.bmm是一种矩阵乘法 torch.squeeze是删除矩阵中大小为1的维度
        # torch.bmm(rots, samples.unsqueeze(-1))，然后把新增的那个维度(1维的)删掉，得到的便是新高斯点的协方差矩阵， 所代表的便是新高斯点的形状
        # 新高斯点坐标(绝对位置) = 新高斯点旋转矩阵 * 采样点形状 + 原高斯点的位置(相对位置)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        # 新的缩放因子 = 原缩放因子 / (0.8*2) 要小于原缩放因子
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        # 新高斯点剩下的属性都直接采用原高斯点的属性
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        # 把这些新变量用来给新创建的高斯点做初始化
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        # 创建一个过滤器，选中老高斯点以及中间生成的高斯点
        # 把原本的高斯，以及中间创建的一些新高斯分布(中间变量)给删掉(新高斯分布此时已经用新的索引表达了)
        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    # 高斯点克隆
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        # 使用掩码去筛选需要处理的高斯点
        # 筛选条件: 高斯点的梯度需要大于阈值梯度 并且 高斯点形状小于场景的原本形状
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)

        # 直接把原高斯点的所有参数全部克隆给新高斯点即可
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # 创建这个克隆出来的新高斯点
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation)

    # 自适应控制高斯点的删除
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        # 累加的所有梯度除以一个分母， 得到现在的平均梯度
        grads = self.xyz_gradient_accum / self.denom
        # 归零操作
        grads[grads.isnan()] = 0.0

        # 在删除高斯点的时候进行克隆和分裂操作
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # 使用掩码选中所有不透明度小于最小不透明度阈值的高斯点
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            # 标记所有高斯分布尺寸 大于 最大屏幕尺寸的高斯点
            big_points_vs = self.max_radii2D > max_screen_size
            # 标记所有尺寸大于0.1倍场景范围的高斯点
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            # 设置一个掩码用来统一剔除这些不合规范的高斯点
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # 剔除这些不合规范的椭球
        self.prune_points(prune_mask)

        # 清除缓存
        torch.cuda.empty_cache()

    # 记录更新点的过程中累加的梯度
    # 添加自适应控制密度的状态 每处理一个点， 增加一次这个点的梯度就给点的分母加一
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # viewspace_point_tensor是一个二维高斯分布(椭圆)， 记录x和y方向上的梯度
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1,
                                                             keepdim=True)
        # 点的分母加一
        self.denom[update_filter] += 1
