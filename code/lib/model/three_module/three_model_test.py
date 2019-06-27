# -*- coding: UTF-8 -*-
import torch
import argparse
import numpy as np
import torchvision.transforms as transforms
from torchvision.models import vgg
from torchvision.models import alexnet
import os
from tqdm import tqdm
import datetime
import random
import path
import torch.nn.functional as F
from IPython import embed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from model.utils.config import cfg
from model.loss.loss_functions import L1_Sum
import math

def rotation_matrix(ry):
    b, bins = ry.size()
    ry = ry.view(b * bins)
    zero = torch.abs(ry).detach() * 0
    one = torch.abs(ry).detach() * 0 + 1
    rotation_x = torch.stack([one,  zero,  zero,
                              zero,  one, -zero,
                              zero, zero,  one], dim=1).view(b * bins, 3, 3)

    rotation_y = torch.stack([torch.cos(ry),   zero,  torch.sin(ry),
                              zero,    one,   zero,
                              -torch.sin(ry),   zero,  torch.cos(ry)], dim=1).view(b * bins, 3, 3)

    rotation_z = torch.stack([one, -zero, zero,
                              zero,  one, zero,
                              zero,  zero,  one], dim=1).view(b * bins, 3, 3)

    rot_mat = rotation_x.bmm(rotation_y).bmm(rotation_z)
    rot_mat = rot_mat.view(b, bins, 3, 3)
    return rot_mat


def three_model_test(local_xyz_0, softmax_mask_0, HWL_bias_0, DR_four_0, rois_for_3d, gt_Inistric_info,
                param_for_txtytz_batch, eight_vertex_tensor_batch):

    rois_for_3d = rois_for_3d.view(-1, rois_for_3d.shape[1])
    param_for_txtytz_batch = param_for_txtytz_batch.view(-1, param_for_txtytz_batch.shape[2], param_for_txtytz_batch.shape[3])
    eight_vertex_tensor_batch = eight_vertex_tensor_batch.view(-1, eight_vertex_tensor_batch.shape[2], eight_vertex_tensor_batch.shape[3])

    gt_dimension_anchor = torch.from_numpy(np.array([1.52159147, 1.64443089, 3.85813679])).float().cuda()

    focal_x = gt_Inistric_info[0][0].float()
    center_x = gt_Inistric_info[0][2].float()
    gt_crop_ray = torch.atan2(focal_x, (rois_for_3d[0][1] + rois_for_3d[0][3]) / 2 - center_x)
    gt_crop_ray_90 = gt_crop_ray - np.pi / 2

    Rot_ray_90_fuzhi = rotation_matrix(-gt_crop_ray_90.unsqueeze(0).unsqueeze(1)).squeeze(1).cuda()

    batch = 1
    for circle_index in range(5):
        xyz_computed = Rot_ray_90_fuzhi.float().bmm(local_xyz_0.view(batch, 3, -1)).permute(1, 0, 2).contiguous().view(3, -1)
        x_computed = xyz_computed[0].view(batch, -1)
        y_computed = xyz_computed[1].view(batch, -1)
        z_computed = xyz_computed[2].view(batch, -1)
        pu = softmax_mask_0.view(batch, -1).type_as(z_computed)
        au = param_for_txtytz_batch[:, 0, :]
        bu = param_for_txtytz_batch[:, 1, :]
        cu1 = param_for_txtytz_batch[:, 2, :]
        cu2_z = param_for_txtytz_batch[:, 3, :]
        du1 = param_for_txtytz_batch[:, 4, :]
        du2_z = param_for_txtytz_batch[:, 5, :]

        cu = cu2_z * z_computed + (-1) * x_computed + cu1
        du = du2_z * z_computed + (-1) * y_computed + du1
        tz_down = (((au * pu).sum(dim=1)) * ((au * pu).sum(dim=1)) + ((bu * pu).sum(dim=1)) * ((bu * pu).sum(dim=1))) / (
            pu.sum(dim=1)) - (au * au * pu + bu * bu * pu).sum(dim=1)
        tz_top = (au * cu * pu + bu * du * pu).sum(dim=1) - ((au * pu).sum(dim=1) * (cu * pu).sum(dim=1)) / (pu.sum(dim=1)) - (
                    (bu * pu).sum(dim=1) * (du * pu).sum(dim=1)) / (pu.sum(dim=1))

        tz = tz_top / (tz_down + cfg.THREEDIM.DET_THRESHOLD)
        tx = (((au * pu).sum(dim=1)) / (pu.sum(dim=1) + cfg.THREEDIM.DET_THRESHOLD)) * tz + (((cu * pu).sum(dim=1)) / (pu.sum(dim=1) + cfg.THREEDIM.DET_THRESHOLD))
        ty = (((bu * pu).sum(dim=1)) / (pu.sum(dim=1) + cfg.THREEDIM.DET_THRESHOLD)) * tz + (((du * pu).sum(dim=1)) / (pu.sum(dim=1) + cfg.THREEDIM.DET_THRESHOLD))
        pred_Ray = torch.atan2(tz, tx)
        Rot_ray_90_fuzhi = rotation_matrix(-(pred_Ray.unsqueeze(1) - np.pi / 2)).squeeze(1).cuda()
        pred_location_temp = torch.cat([tx.unsqueeze(1), ty.unsqueeze(1), tz.unsqueeze(1)], dim=1).detach()

        tqdm.write('prtx:%.4f   prty:%.4f   prtz:%.4f' % (
            pred_location_temp[0][0], pred_location_temp[0][1], pred_location_temp[0][2]))
        print('------------------------------------------------------------------------------------------------')


    pred_location = torch.cat([tx.unsqueeze(1), ty.unsqueeze(1), tz.unsqueeze(1)], dim=1).detach()
    pred_dimension = gt_dimension_anchor + HWL_bias_0

    pred_Rot_ray_90_fuzhi = rotation_matrix((0.5 * np.pi - pred_Ray[0:1]).unsqueeze(1)).squeeze(1).cuda()
    eight_vertex_0_two = torch.cat([DR_four_0[0:1][:, :2], DR_four_0[0:1][:, :2] * 0, DR_four_0[0:1][:, 2:]]).unsqueeze(0)
    pred_eight_vertex_0_global = pred_Rot_ray_90_fuzhi.bmm(eight_vertex_0_two)
    temp_global = torch.cat([pred_eight_vertex_0_global[0][0].unsqueeze(1), pred_eight_vertex_0_global[0][2].unsqueeze(1)],dim=1).transpose(1, 0)
    temp_obj = torch.cat([(eight_vertex_tensor_batch[0][0, :] * ((pred_dimension[0][2] / 2))).unsqueeze(1),
                          (eight_vertex_tensor_batch[0][2, :] * ((pred_dimension[0][1] / 2))).unsqueeze(1)], dim=1)
    compute_ry_ = 0
    for temp_ry_i in range(2):
        compute_ry_ += ((torch.atan2(temp_obj[temp_ry_i][1], temp_obj[temp_ry_i][0])) -
                        torch.atan2(temp_global[temp_ry_i][1], temp_global[temp_ry_i][0])) % (2 * np.pi)
    pred_ry = compute_ry_ / 2

    tqdm.write('prHH:%.4f   prWW:%.4f   prLL:%.4f  prtx:%.4f   prty:%.4f   prtz:%.4f   prRy:%.4f' % (
        pred_dimension[0][0], pred_dimension[0][1], pred_dimension[0][2],
        pred_location[0][0], pred_location[0][1], pred_location[0][2], pred_ry))
    print('------------------------------------------------------------------------------------------------')

    return pred_location, pred_dimension, pred_ry