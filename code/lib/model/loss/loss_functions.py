from __future__ import division
import torch
import pdb
from torch import nn
import math
from torch.autograd import Variable
import numpy as np
from IPython import embed
import torch.nn.functional as F
import scipy
import cv2

def L1_Sum(pred, gt):
    L1_sum_func = nn.L1Loss(reduce=True, size_average=False).cuda()
    loss = L1_sum_func(pred, gt)
    return loss
