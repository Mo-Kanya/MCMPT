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
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.str2bool import str2bool
from multiview_detector.trainer import PerspectiveTrainer

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
    if 'wildtrack' in args.dataset:
        base = Wildtrack(os.path.expanduser('~/Data/Wildtrack'))
    elif 'multiviewx' in args.dataset:
        base = MultiviewX(os.path.expanduser('~/Data/MultiviewX'))
    
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
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)
    
    logdir = args.logdir
    os.makedirs(logdir, exist_ok=True)
    print('Settings:')
    print(vars(args))
    
    # detection_model
    detection_model_path = os.path.join(
        args.detection_model_path_home, 
        args.resume,
        "MultiviewDetector.pth"
    )
    detection_model = MVDeTr(train_set, args.arch, world_feat_arch=args.world_feat,
                   bottleneck_dim=args.bottleneck_dim, outfeat_dim=args.outfeat_dim, droupout=args.dropout).cuda()
    detection_model.load_state_dict(torch.load(detection_model_path))
    detection_model.eval()
    print('Test loaded detection model...')
    
    param_dicts = [{"params": [p for n, p in detection_model.named_parameters() if 'base' not in n and p.requires_grad], },
                   {"params": [p for n, p in detection_model.named_parameters() if 'base' in n and p.requires_grad],
                    "lr": args.lr * args.base_lr_ratio, }, ]
    # optimizer = optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optimizer = optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)
    detector = PerspectiveTrainer(detection_model, logdir, args.cls_thres, args.alpha, args.use_mse, args.id_ratio)
    # detector.test(None, test_loader, None, visualize=True)
    
    world_reduce = test_loader.dataset.world_reduce
    for data, world_gt, imgs_gt, affine_mats, frame in test_loader:
        pos, id, cnt = detector.inference(data, world_gt, imgs_gt, affine_mats, frame, world_reduce)
        print(pos) #1x34556x2
        print(id) #7x2
        print(cnt) #7
        import pdb; pdb.set_trace()
    
    
def parse_args():
    parser = argparse.ArgumentParser(description='Multiview detector')
    
    # trainer
    parser.add_argument('--cls_thres', type=float, default=0.6)
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
        
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)