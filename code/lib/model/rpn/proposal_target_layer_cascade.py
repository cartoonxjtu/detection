from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr
from ..utils.config import cfg
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch
import pdb
from generate_anchors import generate_anchors
import random

class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)

    def forward(self, all_rois, gt_boxes, num_boxes, gt_3d_info):

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        gt_boxes_append[:,:,1:5] = gt_boxes[:,:,:4]

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        labels, rois, bbox_targets, bbox_inside_weights, rois_for_3d, gt_bbox_for_3d, gt_3d_info_rois = self._sample_rois_pytorch(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes, gt_3d_info)
        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights, rois_for_3d, gt_bbox_for_3d, gt_3d_info_rois

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights


    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets = bbox_transform_batch(ex_rois, gt_rois)

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets


    def compute_iou(self, box1, box2):
      box_num = box1.shape[0]
      box2 = np.repeat(np.expand_dims(np.array(box2, np.newaxis), axis=0), box_num, axis=0)
      xmin1 = box1[:, 0]
      ymin1 = box1[:, 1]
      xmax1 = box1[:, 2]
      ymax1 = box1[:, 3]
      xmin2 = box2[:, 0]
      ymin2 = box2[:, 1]
      xmax2 = box2[:, 2]
      ymax2 = box2[:, 3]

      xx1 = np.max([xmin1, xmin2], axis=0)
      yy1 = np.max([ymin1, ymin2], axis=0)
      xx2 = np.min([xmax1, xmax2], axis=0)
      yy2 = np.min([ymax1, ymax2], axis=0)

      area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
      area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

      inter_area = (np.max([np.zeros((xx1.shape[0],)), xx2 - xx1], axis=0)) * (
        np.max([np.zeros((xx1.shape[0],)), yy2 - yy1], axis=0))
      iou = inter_area / (area1 + area2 - inter_area + 1e-6)

      return iou

    def jitter_gt2dbbox(self, bbox_float_1):
      bbox_initial = bbox_float_1.clone()
      # crop argument

      scale_x = random.uniform(-0.3, 0.3)
      scale_y = random.uniform(-0.3, 0.3)
      bbox_2d_w = bbox_float_1[2] - bbox_float_1[0]
      bbox_2d_h = bbox_float_1[3] - bbox_float_1[1]
      bbox_float_1[0] = bbox_float_1[0] + scale_x * bbox_2d_w
      bbox_float_1[2] = bbox_float_1[2] + scale_x * bbox_2d_w
      bbox_float_1[1] = bbox_float_1[1] + scale_y * bbox_2d_h
      bbox_float_1[3] = bbox_float_1[3] + scale_y * bbox_2d_h

      anchors = generate_anchors(int(bbox_float_1[0]), int(bbox_float_1[1]), int(bbox_float_1[2] - bbox_float_1[0]) + 1,
                                 int(bbox_float_1[3] - bbox_float_1[1]) + 1, [3 / 4, 4 / 5, 1, 5 / 4, 4 / 3],
                                 [8 / 9, 9 / 10, 1, 10 / 9, 9 / 8])


      anchors[:, 0] = np.clip(anchors[:, 0], 0, np.ceil(cfg.TRAIN.SCALES[0] * (1242 / 375)))
      anchors[:, 2] = np.clip(anchors[:, 2], 0, np.ceil(cfg.TRAIN.SCALES[0] * (1242 / 375)))
      anchors[:, 1] = np.clip(anchors[:, 1], 0, cfg.TRAIN.SCALES[0])
      anchors[:, 3] = np.clip(anchors[:, 3], 0, cfg.TRAIN.SCALES[0])

      # # half argument
      if random.randint(0, 1) == 0:
          iou = self.compute_iou(anchors, bbox_float_1)
          to_selecte_index = np.where(iou > 0.6)
          if len(to_selecte_index[0]) != 0:
              selected_index = random.sample(to_selecte_index[0].tolist(), 1)
              crop_bbox = anchors[selected_index[0]]
          else:
              crop_bbox = bbox_initial.cpu().numpy()
      else:
        crop_bbox = bbox_initial.cpu().numpy()
      return crop_bbox

    def compute_iou(self, box1, box2):
      box_num = box1.shape[0]
      box2 = np.repeat(np.expand_dims(np.array(box2, np.newaxis), axis=0), box_num, axis=0)
      xmin1 = box1[:, 0]
      ymin1 = box1[:, 1]
      xmax1 = box1[:, 2]
      ymax1 = box1[:, 3]
      xmin2 = box2[:, 0]
      ymin2 = box2[:, 1]
      xmax2 = box2[:, 2]
      ymax2 = box2[:, 3]

      xx1 = np.max([xmin1, xmin2], axis=0)
      yy1 = np.max([ymin1, ymin2], axis=0)
      xx2 = np.min([xmax1, xmax2], axis=0)
      yy2 = np.min([ymax1, ymax2], axis=0)

      area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
      area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

      inter_area = (np.max([np.zeros((xx1.shape[0],)), xx2 - xx1], axis=0)) * (
        np.max([np.zeros((xx1.shape[0],)), yy2 - yy1], axis=0))
      iou = inter_area / (area1 + area2 - inter_area + 1e-6)

      return iou

    def jitter_gt2dbbox(self, bbox_float_1):
      bbox_initial = bbox_float_1.clone()
      # crop argument

      scale_x = random.uniform(-0.3, 0.3)  # x轴方向的平移比例
      scale_y = random.uniform(-0.3, 0.3)  # x轴方向的平移比例
      bbox_2d_w = bbox_float_1[2] - bbox_float_1[0]
      bbox_2d_h = bbox_float_1[3] - bbox_float_1[1]
      bbox_float_1[0] = bbox_float_1[0] + scale_x * bbox_2d_w
      bbox_float_1[2] = bbox_float_1[2] + scale_x * bbox_2d_w
      bbox_float_1[1] = bbox_float_1[1] + scale_y * bbox_2d_h
      bbox_float_1[3] = bbox_float_1[3] + scale_y * bbox_2d_h

      anchors = generate_anchors(int(bbox_float_1[0]), int(bbox_float_1[1]), int(bbox_float_1[2] - bbox_float_1[0]) + 1,
                                 int(bbox_float_1[3] - bbox_float_1[1]) + 1, [3 / 4, 4 / 5, 1, 5 / 4, 4 / 3],
                                 [8 / 9, 9 / 10, 1, 10 / 9, 9 / 8])

      anchors[:, 0] = np.clip(anchors[:, 0], 0, np.ceil(cfg.TRAIN.SCALES[0] * (1242 / 375)))
      anchors[:, 2] = np.clip(anchors[:, 2], 0, np.ceil(cfg.TRAIN.SCALES[0] * (1242 / 375)))
      anchors[:, 1] = np.clip(anchors[:, 1], 0, cfg.TRAIN.SCALES[0])
      anchors[:, 3] = np.clip(anchors[:, 3], 0, cfg.TRAIN.SCALES[0])

      # # half argument
      if random.randint(0, 1) == 0:
          iou = self.compute_iou(anchors, bbox_float_1)
          to_selecte_index = np.where(iou > 0.6)
          if len(to_selecte_index[0]) != 0:
              selected_index = random.sample(to_selecte_index[0].tolist(), 1)
              crop_bbox = anchors[selected_index[0]]
          else:
              crop_bbox = bbox_initial.cpu().numpy()
      else:
        crop_bbox = bbox_initial.cpu().numpy()
      return crop_bbox

    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes, gt_3d_info):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)

        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

        max_overlaps, gt_assignment = torch.max(overlaps, 2)

        batch_size = overlaps.size(0)
        num_proposal = overlaps.size(1)
        num_boxes_per_img = overlaps.size(2)

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        labels = gt_boxes[:, :, 4].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)

        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch  = all_rois.new(batch_size, rois_per_image, 5).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()

        rois_batch_for_3d = all_rois.new(batch_size, 4, 5).zero_()
        gt_rois_batch_for_3d = all_rois.new(batch_size, 4, 5).zero_()
        gt_3d_info_batch_for_3d = all_rois.new(batch_size, 4, gt_3d_info.shape[2]).zero_()        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):

            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)
            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            bg_num_rois = bg_inds.numel()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # fg_rois_per_this_image_for_3d = min(fg_rois_per_this_image, 8)
                # fg_inds_for_3d = fg_inds[rand_num[:fg_rois_per_this_image_for_3d]]

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
                # bg_rois_per_this_image_for_3d = 8 - fg_rois_per_this_image_for_3d

                # Seems torch.rand has a bug, it will generate very large number and make an error.
                # We use numpy rand instead.
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                bg_inds = bg_inds[rand_num]
                # bg_inds_for_3d = bg_inds[rand_num[:bg_rois_per_this_image_for_3d]]

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0

            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i,:,0] = i
            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

            if len(fg_inds) >= cfg.THREEDIM.ThreeDimRois:
                rand_num = torch.from_numpy(np.random.permutation(len(fg_inds))).type_as(gt_boxes).long()
                fg_3d_inds = fg_inds[rand_num[:cfg.THREEDIM.ThreeDimRois]]
            elif len(fg_inds) == cfg.THREEDIM.ThreeDimRois:
                fg_3d_inds = fg_inds
            elif len(fg_inds) == 3:
                rand_num = torch.from_numpy(np.random.permutation(len(fg_inds))).type_as(gt_boxes).long()
                fg_3d_inds_temp = fg_inds[rand_num[:(cfg.THREEDIM.ThreeDimRois - len(fg_inds))]]
                fg_3d_inds = torch.cat([fg_inds, fg_3d_inds_temp], 0)
            elif len(fg_inds) == 2:
                fg_3d_inds = torch.cat([fg_inds, fg_inds], 0)
            elif len(fg_inds) == 1:
                fg_3d_inds = torch.cat([fg_inds, fg_inds, fg_inds, fg_inds], 0)

            rois_batch_for_3d[i] = all_rois[i][fg_3d_inds]
            rois_batch_for_3d[i, :, 0] = i
            gt_rois_batch_for_3d[i] = gt_boxes[i][gt_assignment[i][fg_3d_inds]]
            gt_3d_info_batch_for_3d[i] = gt_3d_info[i][gt_assignment[i][fg_3d_inds]]


        bbox_target_data = self._compute_targets_pytorch(rois_batch[:, :, 1:5], gt_rois_batch[:, :, :4])
        bbox_targets, bbox_inside_weights = \
            self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)
        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights, rois_batch_for_3d.view(-1, 5), \
               gt_rois_batch_for_3d.view(-1, 5), gt_3d_info_batch_for_3d.view(-1, 34)