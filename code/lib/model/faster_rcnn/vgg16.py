# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN
import pdb


def conv3x3(in_, out):
  return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
  def __init__(self, in_, out):
    super().__init__()
    self.conv = conv3x3(in_, out)
    self.activation = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.conv(x)
    x = self.activation(x)
    return x

class DecoderBlockV2(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
    super(DecoderBlockV2, self).__init__()
    self.in_channels = in_channels

    if is_deconv:
      """
          Paramaters for Deconvolution were chosen to avoid artifacts, following
          link https://distill.pub/2016/deconv-checkerboard/
      """

      self.block = nn.Sequential(
        ConvRelu(in_channels, middle_channels),
        nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2,
                           padding=1),
        nn.ReLU(inplace=True)
      )
    else:
      self.block = nn.Sequential(
        Interpolate(scale_factor=2, mode='bilinear'),
        ConvRelu(in_channels, middle_channels),
        ConvRelu(middle_channels, out_channels),
      )

  def forward(self, x):
    return self.block(x)


class vgg16(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False, is_deconv=False, num_filters=32):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.is_deconv = is_deconv
    self.num_filters = num_filters

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})
    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    self.pool = nn.MaxPool2d(2, 2)
    self.relu = nn.ReLU(inplace=True)

    self.conv1 = nn.Sequential(self.RCNN_base[0],
                               self.relu,
                               self.RCNN_base[2],
                               self.relu)
    #
    self.conv2 = nn.Sequential(self.RCNN_base[5],
                               self.relu,
                               self.RCNN_base[7],
                               self.relu)
    #
    self.conv3 = nn.Sequential(self.RCNN_base[10],
                               self.relu,
                               self.RCNN_base[12],
                               self.relu,
                               self.RCNN_base[14],
                               self.relu)
    #
    self.conv4 = nn.Sequential(self.RCNN_base[17],
                               self.relu,
                               self.RCNN_base[19],
                               self.relu,
                               self.RCNN_base[21],
                               self.relu)
    #
    self.conv5 = nn.Sequential(self.RCNN_base[24],
                               self.relu,
                               self.RCNN_base[26],
                               self.relu,
                               self.RCNN_base[28],
                               self.relu)

    self.conv3_2d = nn.Sequential(self.RCNN_base[10],
                               self.relu,
                               self.RCNN_base[12],
                               self.relu,
                               self.RCNN_base[14],
                               self.relu)
    #
    self.conv4_2d = nn.Sequential(self.RCNN_base[17],
                               self.relu,
                               self.RCNN_base[19],
                               self.relu,
                               self.RCNN_base[21],
                               self.relu)
    #
    self.conv5_2d = nn.Sequential(self.RCNN_base[24],
                               self.relu,
                               self.RCNN_base[26],
                               self.relu,
                               self.RCNN_base[28],
                               self.relu)
    #
    self.center = DecoderBlockV2(512, self.num_filters * 8 * 2, self.num_filters * 8, self.is_deconv)

    self.dec5 = DecoderBlockV2(512 + self.num_filters * 8, self.num_filters * 8 * 2, self.num_filters * 8, self.is_deconv)
    self.dec4 = DecoderBlockV2(512 + self.num_filters * 8, self.num_filters * 8 * 2, self.num_filters * 8, self.is_deconv)

    self.xyz_feature = nn.Conv2d(256, 96, kernel_size=1)
    self.local_xyz_ = nn.Conv2d(96, 3, kernel_size=1)
    self.final_mask = nn.Conv2d(96, 1, kernel_size=1)
    self.HWL_local_fc = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 1024),
        nn.ReLU(True),
        nn.Dropout())
    self.DR_four = nn.Linear(1024, 4)
    self.dimension = nn.Linear(1024, 3)


    ## self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

