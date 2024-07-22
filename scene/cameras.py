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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

# 相机模型， 表达了3DGS从3D空间投影到2D空间(像平面)的转换过程
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid  # 相机的唯一标识符
        self.colmap_id = colmap_id  # 做colmap时候相机位姿的id
        self.R = R  # 旋转矩阵
        self.T = T  # 平移矩阵
        self.FoVx = FoVx  # 相机在水平方向视野范围
        self.FoVy = FoVy  # 相机在垂直方向的视野范围
        self.image_name = image_name  # 图像名字

        # 切换GPU卡
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # 把图像的范围限制到0~1之间(先归一化然后放到设备上)
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        # 设置图像新的尺寸(宽高)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        # 指定图像是否需要alpha掩码
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        # 相机的最近点和最远点
        self.zfar = 100.0
        self.znear = 0.01

        # 相机平移和缩放的值
        self.trans = trans
        self.scale = scale

        # 由世界坐标系到相机坐标系的转换，得到相机坐标系
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # 先投影到归一化的坐标系里面， 确保相机内参矩阵是统一的(标准相机内参矩阵)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

