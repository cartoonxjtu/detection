# -*- coding: utf-8 -*-
import torch.utils.data as data
import numpy as np
from scipy.misc import imread
import torch
from path import Path
import random
import cv2
from IPython import embed
from torch.autograd import Variable
import os
import time
import re
import math
import tqdm
import datetime
import argparse
from random import choice
import torchvision.transforms as transforms
import numpy.linalg as lg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model.utils.config import cfg
from IPython import embed
import datetime, time

def set_id_grid_2(h, w):
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w)
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w)
    ones = torch.ones(1, h, w)  # 全 1
    pixel_coords = torch.stack((j_range.float(), i_range.float(), ones), dim=1)
    return pixel_coords

def homegraphy_for_3d_test(crop_length, rois_for_3d, gt_Inistric_info):
    batch_size = 1
    rois_num = 1
    rois_batch, _ = rois_for_3d.view(-1, 5).shape

    crop_bbox_initial = rois_for_3d.view(-1, 5)[:, 1:]
    crop_bbox = crop_bbox_initial
    grid = set_id_grid_2(crop_length, crop_length).squeeze(0).cuda().repeat(rois_batch, 1, 1, 1)
    crop_bbox_h = crop_bbox[:, 3:] - crop_bbox[:, 1:2]
    crop_bbox_w = crop_bbox[:, 2:3] - crop_bbox[:, 0:1]

    grid[:, 0] = (grid[:, 0] / (crop_length - 1)) * crop_bbox_w.unsqueeze(2) + crop_bbox[:, 0:1].unsqueeze(2)
    grid[:, 1] = (grid[:, 1] / (crop_length - 1)) * crop_bbox_h.unsqueeze(2) + crop_bbox[:, 1:2].unsqueeze(2)
    grid_view = grid.view(rois_batch, 3, -1)

    Inistric= gt_Inistric_info.squeeze(1).view(1, 3, 4)
    u = grid_view[:, 0].float().cuda()
    v = grid_view[:, 1].float().cuda()
    cx = Inistric[:, 0, 2].unsqueeze(1).float().cuda()
    cy = Inistric[:, 1, 2].unsqueeze(1).float().cuda()
    fx = Inistric[:, 0, 0].unsqueeze(1).float().cuda()
    fy = Inistric[:, 1, 1].unsqueeze(1).float().cuda()
    dx = Inistric[:, 0, 3].unsqueeze(1).float().cuda()
    dy = Inistric[:, 1, 3].unsqueeze(1).float().cuda()
    dz = Inistric[:, 2, 3].unsqueeze(1).float().cuda()

    au = (u - cx) / fx
    bu = (v - cy) / fy
    cu1 = (u * dz - dx) / fx
    cu2_z = (u - cx) / fx
    du1 = (v * dz - dy) / fy
    du2_z = (v - cy) / fx

    point_x = []
    point_y = []
    point_z = []
    for point_i in [-1, 1]:
        for point_j in [0, -2]:
            for point_k in [-1, 1]:
                point_x.append(point_i)  # L  X
                point_y.append(point_j)  # H  Y
                point_z.append(point_k)  # W  Z
    eight_vertex = np.vstack((np.array(point_x), np.array(point_y), np.array(point_z)))
    eight_vertex_tensor = torch.from_numpy(eight_vertex).float().cuda()

    param_for_txtytz = torch.cat([au.unsqueeze(1), bu.unsqueeze(1), cu1.unsqueeze(1), cu2_z.unsqueeze(1),
                                    du1.unsqueeze(1), du2_z.unsqueeze(1)], dim=1)
    param_for_txtytz_batch = param_for_txtytz.unsqueeze(0)
    eight_vertex_tensor_batch = eight_vertex_tensor.repeat(batch_size, rois_num, 1, 1)

    return param_for_txtytz_batch, eight_vertex_tensor_batch


