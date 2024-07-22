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

#include "spatial.h"
#include "simple_knn.h"

//算对应点云最近点的函数
torch::Tensor
distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);

  // 采用KNN算法计算这个高斯点最近的三个高斯点，并计算出和这三个高斯点的平均距离， 这个可以用来去构建一个初始的高斯球
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());

  return means;
}