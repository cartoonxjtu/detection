# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb, TestFolder, rank_roidb_ratio
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from tqdm import tqdm
from model.homegraphy.homegraphy_test import homegraphy_for_3d_test
from model.three_module.three_model_test import three_model_test
import pdb
import torch.nn.functional as F
import torchvision.transforms as transforms

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--root_path', default='/data/kitti', help='path to dataset')
    parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
    parser.add_argument('--norm_m', default=(0.485, 0.456, 0.406), type=float, help=' transform norm -- m')
    parser.add_argument('--norm_s', default=(0.229, 0.224, 0.225), type=float, help=' transform norm -- s')

    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='vgg16', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--index',
                        default=-1, type=int)
    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    args = parse_args()

    np.random.seed(cfg.RNG_SEED)

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.TRAIN.USE_FLIPPED = False

    test_img_set = TestFolder(args, args.root_path, args.seed, flip=True)
    test_ratio_list, test_ratio_index = rank_roidb_ratio(test_img_set)

    print('{:d} roidb entries'.format(len(test_img_set)))

    classes = ('__background__', 'Car')
    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(('__background__', 'Car'), pretrained=True, class_agnostic=args.class_agnostic,
                           is_deconv=True)
    elif args.net == 'res101':
        fasterRCNN = resnet(classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    input_dir = cfg.TRAIN.out_path
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)

    if not os.path.isdir(input_dir):
        os.mkdir(input_dir)
    model_lst = [x for x in sorted(os.listdir(input_dir)) if x.endswith('.pth')]

    if len(model_lst) != 0:
        print('Find previous model %s' % model_lst[args.index])
    print("load checkpoint %s" % (model_lst[args.index]))

    checkpoint = torch.load(input_dir + '/' + model_lst[args.index])
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    del checkpoint
    torch.cuda.empty_cache()
    print('load model successfully!')
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_Inistric_info = torch.FloatTensor(1)
    gt_boxes = torch.FloatTensor(1)

    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_Inistric_info = gt_Inistric_info.cuda()
    gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_Inistric_info = Variable(gt_Inistric_info)
    gt_boxes = Variable(gt_boxes)

    cfg.CUDA = True

    fasterRCNN.cuda()

    start = time.time()
    max_per_image = 100

    vis = args.vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = 'faster_rcnn_10'
    num_images = len(test_ratio_index)
    all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(2)]

    train_transform = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
         transforms.ToTensor(),
         transforms.Normalize(args.norm_m, args.norm_s)])

    dataset = roibatchLoader(test_img_set, test_ratio_list, test_ratio_index, 1, \
                             2, training=False, normalize=False, transform=train_transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=False, num_workers=0,
                                             pin_memory=True)

    data_iter = iter(dataloader)

    _t = {'im_detect': time.time(), 'misc': time.time()}

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    dirs = ['det%d' % args.index, 'det%d/pred' % args.index, 'det%d/pred/data' % args.index]
    if os.path.exists(dirs[0]):
        print('det%d exist' % args.index)
        from IPython import embed
        embed()

    else:
        os.makedirs(dirs[0])
        for i in range(1, 3):
            os.makedirs(dirs[i])
    fid = open("det%d/distance.txt" % args.index, 'a')
    img_path = '/data/kitti/testing/test/image'

    IDLst = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    ID_index = 0
    temp_name = IDLst[ID_index] + '.txt'

    for i in tqdm(range(num_images)):
        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        num_boxes.data.resize_(data[3].size()).copy_(data[3])
        gt_Inistric_info.data.resize_(data[4].size()).copy_(data[4])
        image_name = data[5][0]
        txt_png = image_name[-10:]
        txt_name = txt_png[:-4] + '.txt'

        if temp_name != txt_name:
            if os.path.exists(input_dir + '/det%d/distance/data/' % args.index + temp_name):
                inID_indexdex = ID_index + 1
                temp_name = IDLst[ID_index] + '.txt'
            while temp_name != txt_name:
                f = open("det%d/pred/data/%s" % (args.index, temp_name), 'a')
                f.close()
                ID_index = ID_index + 1
                temp_name = IDLst[ID_index] + '.txt'
        print('txt_name')
        print(txt_name)
        print('temp_name')
        print(temp_name)
        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, conv1, conv2, conv3, conv4, conv5, conv5_withpooling \
            = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if vis:
            im = cv2.imread(imdb.image_path_at(i))
            im2show = np.copy(im)
        num_classes = 2
        for j in xrange(1, num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        tosave_thresh = 0.1
        for i in range(cls_dets.shape[0]):
            bbox = cls_dets[i, :4]
            bbox_enlarge = bbox * data[1][0][2].item()

            score = cls_dets[i, -1]
            if score > tosave_thresh:
                rois_for_3d = torch.zeros(1, 5).cuda()
                rois_for_3d[:, 1:] = bbox
                rois_for_3d_enlarge = torch.zeros(1, 5).cuda()
                rois_for_3d_enlarge[:, 1:] = bbox_enlarge
                param_for_txtytz_batch, eight_vertex_tensor_batch = homegraphy_for_3d_test(cfg.THREEDIM.CROP_LENGTH, rois_for_3d, gt_Inistric_info)

                conv3_begin = fasterRCNN.pool(conv2)
                conv3_rois = fasterRCNN.RCNN_roi_align_4(conv3_begin, rois_for_3d_enlarge.view(-1, 5))
                conv3_rois = fasterRCNN.conv3(conv3_rois)
                conv4_rois = fasterRCNN.conv4(fasterRCNN.pool(conv3_rois))
                conv5_rois = fasterRCNN.conv5(fasterRCNN.pool(conv4_rois))
                center_rois = fasterRCNN.center(fasterRCNN.pool(conv5_rois))
                dec5_rois = fasterRCNN.dec5(torch.cat([center_rois, conv5_rois], 1))
                dec4_rois = fasterRCNN.dec4(torch.cat([dec5_rois, conv4_rois], 1))

                xyz_feature = fasterRCNN.xyz_feature(dec4_rois)
                local_xyz_0 = fasterRCNN.local_xyz_(xyz_feature)
                batch, _, _, _ = xyz_feature.shape
                mask_0 = F.softmax(fasterRCNN.final_mask(xyz_feature).view(batch, -1), dim=1).view(batch, cfg.THREEDIM.CROP_LENGTH, cfg.THREEDIM.CROP_LENGTH)

                HWLRy_feature_view = fasterRCNN.pool(conv5_rois).view(-1, 512 * 7 * 7)
                HWL_local_feature = fasterRCNN.HWL_local_fc(HWLRy_feature_view)
                HWL_bias_0 = fasterRCNN.dimension(HWL_local_feature)
                DR_four_0 = fasterRCNN.DR_four(HWL_local_feature)
                pred_location, pred_dimension, pred_ry = \
                    three_model_test(local_xyz_0, mask_0, HWL_bias_0, DR_four_0, rois_for_3d, gt_Inistric_info,
                                     param_for_txtytz_batch, eight_vertex_tensor_batch)
                f = open("det%d/pred/data/%s" % (args.index, txt_name), 'a')
                Ray = torch.atan2(pred_location[0][2], pred_location[0][0])
                Local = Ray + pred_ry - 0.5 * np.pi
                f.write('%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f' % (
                    'Car', -1, -1, Local,
                    bbox[0], bbox[1], bbox[2],
                    bbox[3], pred_dimension[0][0], pred_dimension[0][1], pred_dimension[0][2],
                    pred_location[0][0], pred_location[0][1], pred_location[0][2], pred_ry, cls_dets[i, -1]))
                f.write("\n")