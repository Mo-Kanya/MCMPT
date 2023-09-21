import os
import sys
import shutil
import argparse
import datetime
import random
from distutils.dir_util import copy_tree

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from multiview_detector.datasets import *
from multiview_detector.models.mvdetr import MVDeTr
from multiview_detector.utils.nms import nms
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.str2bool import str2bool
from multiview_detector.trainer import PerspectiveTrainer
from multiview_detector.utils.decode import ctdet_decode, mvdet_decode

from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.utils import visualization
from deep_sort.detection import Detection


NMS_THS = 12
TOP_K = 500

def main(args):
    # check if in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
    else:
        print('No sys.gettrace')
        is_debug = False
    
    # seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # deterministic
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.backends.cudnn.benchmark = True
    
    # dataset
    config_file = args.configfile
    train_set = MMPframeDataset(config_file, train=True, world_reduce=args.world_reduce,
                                img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                                img_kernel_size=args.img_kernel_size, sample_freq=args.sample_freq,
                                dropout=args.dropcam, augmentation=args.augmentation)
    test_set = MMPframeDataset(config_file, train=False, world_reduce=args.world_reduce,
                               img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                               img_kernel_size=args.img_kernel_size, sample_freq=args.sample_freq)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, 
                              num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, 
                             num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)
    
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)
    print('Settings:')
    print(vars(args))
    
    # detection_model
    detector_path = os.path.join(
        args.detection_model_path_home, 
        args.resume,
        "MultiviewDetector.pth"
    )
    detector = MVDeTr(train_set, args.arch, world_feat_arch=args.world_feat,
                   bottleneck_dim=args.bottleneck_dim, outfeat_dim=args.outfeat_dim, 
                   droupout=args.dropout).cuda()
    detector.load_state_dict(torch.load(detector_path))
    detector.eval()
    print('Loaded detection model...')
    
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", args.max_cosine_distance, args.nn_budget)
    tracker = Tracker(metric)
    print('Loaded tracking model...')
    
    for batch_idx, (data, map_gt, imgs_gt, affine_mats, frame) in enumerate(train_loader):
        data = data.cuda()
        with torch.no_grad():
            (world_heatmap, world_offset), \
            (imgs_heatmap, imgs_offset, imgs_wh) = detector(data, affine_mats)
        
        xys = mvdet_decode(torch.sigmoid(world_heatmap.detach().cpu()), 
                           world_offset.detach().cpu(),
                           reduce=train_loader.dataset.world_reduce)
        positions, scores = xys[:, :, :2], xys[:, :, 2:3]
        ids = scores.squeeze() > args.cls_thres
        pos, s = positions[0, ids], scores[0, ids, 0]
        ids, count = nms(pos, s, 12.5, top_k=np.inf)
        count = min(count, 20)
        ids = pos[ids[:count]]/2.  # [::-1, :]
        
        ## project back to camera frames
        
        ## aggregate features of multiple views
        
        ## preprocess for tracker
        detections = []
        for (xy, id, score) in zip(pos, ids, s):
            detections.append(Detection(xy, score, feature))
        
        tracker.predict()
        tracker.update(detections)
        
        # with torch.no_grad():
            # map_res, imgs_res = detector(data, affine_mats)
        # map_grid_res = map_res.detach().cpu().squeeze()
        # v_s = map_grid_res[map_grid_res > args.cls_thres].unsqueeze(1)
        # grid_ij = (map_grid_res > args.cls_thres).nonzero()
        # if train_loader.dataset.indexing == 'xy':
        #     grid_xy = grid_ij[:, [1, 0]]
        # else:
        #     grid_xy = grid_ij
        
        # frame_id = torch.ones_like(v_s) * frame
        # world_xy = grid_xy * train_loader.dataset.grid_reduce
        # scores = v_s
        # res = torch.cat([frame_id, world_xy.float(), scores], dim=1)
        
        # ## NMS before tracking
        # ids, count = nms(grid_xy.float(), scores[:,0], NMS_THS, TOP_K)
        # # ids, count = nms(grid_xy.float(), scores[:,0], NMS_THS, np.inf)
        # grid_xy = grid_xy[ids[:count], :]
        # scores = scores[ids[:count]]
    
    
    
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='Multiview detector')
    
    # trainer
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--id_ratio', type=float, default=0)
    
    # dataset
    parser.add_argument('--configfile', type=str, 
                        default='/home/kanyamo/MVDeTr/multiview_detector/datasets/configs/retail.yaml')
    parser.add_argument('-d', '--dataset', type=str, default='MMP', choices=['MMP'])
    parser.add_argument('--world_reduce', type=int, default=2)
    parser.add_argument('--world_kernel_size', type=int, default=5)
    parser.add_argument('--img_reduce', type=int, default=4)
    parser.add_argument('--img_kernel_size', type=int, default=10)
    parser.add_argument('--sample_freq', type=int, default=5, help='sample part of frames to save time')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='input batch size for training')
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    
    # model
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18', 'mobilenet'])
    parser.add_argument('--detection_model_path_home', type=str, default='/home/kanyamo/MVDeTr/logs/MMP')
    parser.add_argument('--resume', type=str, default="aug_deform_trans_lr0.0005_baseR0.1_neck128_out0_alpha1.0_id0_drop0.0_dropcam0.0_worldRK2_5_imgRK4_10_2023-04-12_20-59-36")
    parser.add_argument('--world_feat', type=str, default='deform_trans',
                        choices=['conv', 'trans', 'deform_conv', 'deform_trans', 'aio'])
    parser.add_argument('--bottleneck_dim', type=int, default=128)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--semi_supervised', type=float, default=0)
    
    # lr
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--base_lr_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train')

    # config
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    parser.add_argument('--augmentation', type=str2bool, default=True)

    # misc
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--dropcam', type=float, default=0.0)
    parser.add_argument('--logdir', type=str, default='./logs_deepsort')
    
    # DeepSORT
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)