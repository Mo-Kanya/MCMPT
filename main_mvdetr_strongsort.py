import os
import sys
import shutil
import argparse
from datetime import datetime
import random
from distutils.dir_util import copy_tree

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
plt.axis('off')
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

from multiview_detector.datasets import *
from multiview_detector.models.mvdetr import MVDeTr
from multiview_detector.utils.nms import nms
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.str2bool import str2bool
from multiview_detector.trainer import PerspectiveTrainer
from multiview_detector.utils.decode import ctdet_decode, mvdet_decode

from strong_sort.strong_sort import StrongSORT
from strong_sort.utils.parser import get_config

from mmp_tracking_helper.mmp_mapping3D_2D_script import *

BODY_HEIGHT = 1600
BODY_WIDTH = 150

def get_bbox_lbrt(coord_mapper, foot_point, cam_id):
    x_raw, y_raw = [], []
    for offset in [[BODY_WIDTH, BODY_WIDTH, 0], [-BODY_WIDTH, -BODY_WIDTH, 0],
                    [BODY_WIDTH, -BODY_WIDTH, 0], [-BODY_WIDTH, BODY_WIDTH, 0],
                    [BODY_WIDTH, BODY_WIDTH, BODY_HEIGHT], [-BODY_WIDTH, -BODY_WIDTH, BODY_HEIGHT],
                    [BODY_WIDTH, -BODY_WIDTH, BODY_HEIGHT], [-BODY_WIDTH, BODY_WIDTH, BODY_HEIGHT]]:
        x_cam, y_cam = coord_mapper.projection(foot_point, cam_id+1, body_offset=offset)
        x_raw.append(x_cam)
        y_raw.append(y_cam)
    l, r = min(x_raw), max(x_raw)
    b, t = min(y_raw), max(y_raw)
    
    return l, b, r, t

def compute_iou(boxes1, boxes2):
    """
    Input:
        boxes1: Nx4 ndarray, representing N bounding boxes coordinates
        boxes2: Mx4 ndarray, representing M bounding boxes coordinates
    Output:
        iou_mat: NxM ndarray, with iou_mat[i, j] = iou(boxes1[i], boxes2[j])
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    iou_mat = torch.zeros((N, M))
    box2areas = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    # print( box2areas.shape)

    for i in range(N):
        box1 = boxes1[i]

        Iwidth = torch.minimum(box1[2], boxes2[:, 2]) - torch.maximum(box1[0], boxes2[:, 0])
        Iwidth = torch.maximum(Iwidth, torch.zeros(Iwidth.shape[0]).cuda())
        Iheight = torch.minimum(box1[3], boxes2[:, 3]) - torch.maximum(box1[1], boxes2[:, 1])
        Iheight = torch.maximum(Iheight, torch.zeros(Iheight.shape[0]).cuda())
        I = Iwidth * Iheight

        U = (box1[2] - box1[0]) * (box1[3] - box1[1]) + box2areas - I

        iou_mat[i, :] = I / U

    return iou_mat

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
    train_set.get_raw = True
    test_set.get_raw = True
    NUM_CAM = train_set.num_cam
    
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
    
    date_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    outdir = os.path.join(args.outdir, date_time)
    os.makedirs(outdir, exist_ok=True)
    print('Settings:')
    print(vars(args))
    with open(os.path.join(outdir, "settings.json"), "w") as f:
        f.write(json.dumps(vars(args), indent=4))
    
    # detection_model
    detector_path = os.path.join(
        args.detection_model_path_home, 
        args.resume,
        "MultiviewDetector.pth"
    )
    detector = MVDeTr(train_set, args.arch, world_feat_arch=args.world_feat,
                   bottleneck_dim=args.bottleneck_dim, outfeat_dim=args.outfeat_dim, 
                   droupout=args.dropout)
    detector.cuda()
    detector.load_state_dict(torch.load(detector_path))
    detector.eval()
    print('Loaded multi-view detection model...')
    
    yolo = YOLO('yolov8n.pt').to("cuda")
    print('Loaded single-view detection model...')
    
    strongsort = StrongSORT(
        args.strong_sort_weights,
        torch.device('cuda'),
        fp16=False,
        max_dist=args.max_dist,
        max_iou_distance=args.max_iou_distance,
        max_age=args.max_age,
        n_init=args.n_init,
        nn_budget=args.nn_budget,
        mc_lambda=args.mc_lambda,
        ema_alpha=args.ema_alpha,
    )
    strongsort.model.warmup()
    print('Loaded tracking model...')
    
    coord_mapper = CoordMapper(test_set.calibration_path)
    if args.visualize:
        colours = np.random.rand(32, 3)

    for batch_idx, (data, world_gt, imgs_gt, affine_mats, frame, raw_data) in enumerate(test_loader):
        data = data.cuda()
        with torch.no_grad():
            (world_heatmap, world_offset), \
            (imgs_heatmap, imgs_offset, imgs_wh) = detector(data, affine_mats)
            cam_det_results = yolo(raw_data[0])
            
        ## post-processing
        xys = mvdet_decode(torch.sigmoid(world_heatmap.detach().cpu()), 
                           world_offset.detach().cpu(),
                           reduce=train_loader.dataset.world_reduce)
        positions, scores = xys[:, :, :2], xys[:, :, 2:3]
        ids = scores.squeeze() > args.cls_thres
        pos, s = positions[0, ids], scores[0, ids, 0]
        ids, count = nms(pos, s, args.nms_ths, np.inf)
        # ids, count = nms(pos, s, 12.5, top_k=np.inf)
        count = min(count, 20)
        grid_xy = pos[ids[:count]] / 2.
        scores = s[ids[:count]]
        # ids = pos[ids[:count]]/2.  # [::-1, :]
        
        # ids, count = nms(grid_xy.float(), scores[:,0], NMS_THS, TOP_K)
        # grid_xy = grid_xy[ids[:count], :]
        # scores = scores[ids[:count]]
        
        ## save YOLO detection outputs of each camera views
        cam_dets = []
        for cam_id, cam_result in enumerate(cam_det_results):
            bboxes = cam_result.boxes.xywh
            classes = cam_result.boxes.cls
            cam_dets.append(bboxes[classes == 0])
        
        if args.visualize:
            fig_cam = plt.figure(figsize=(6.4 * 4, 4.8 * 3))
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0.05)
            fig_cam_subplots = []
            for cam_id_viz in range(NUM_CAM):
                fig_cam_subplots.append(fig_cam.add_subplot(2, NUM_CAM//2, cam_id_viz + 1, title=f"Camera {cam_id_viz + 1}"))
                img_cam = raw_data[0][cam_id_viz].permute(1, 2, 0).detach().cpu().numpy()
                fig_cam_subplots[cam_id_viz].imshow(img_cam)
                fig_cam_subplots[cam_id_viz].set_axis_off()
                
                
        # detections = []
        # for i, (xy, score) in enumerate(zip(grid_xy, scores)):
        #     x, y = xy[0].item(), xy[1].item()
        #     detections.append(list([x, y, score.item()]))
        
        bbox_xywhs_all = [] # (NUM_CAM, num_bboxes, 2)
        for i, (xy, score) in enumerate(zip(grid_xy, scores)):
            x, y = xy[0].item(), xy[1].item()
            foot_point = {'X': x * train_loader.dataset.world_reduce,
                          'Y': y * train_loader.dataset.world_reduce}
            
            bbox_xywhs = []
            for cam_id in range(NUM_CAM):
                l, b, r, t = get_bbox_lbrt(coord_mapper, foot_point, cam_id)
                w, h = r - l, t - b
                bbox_xywhs.append([l, b, w, h])
                
                VIZ_MVDETR_RESULTS = True
                if args.visualize and VIZ_MVDETR_RESULTS:
                    fig_cam_subplots[cam_id].add_patch(
                        patches.Rectangle((l, b), w, h, fill=False, lw=7, ec=colours[i % len(colours)]))
                    fig_cam_subplots[cam_id].annotate(f"mvdetr_{i}", (l, b), color='white',
                                                        weight='bold', fontsize=10, ha='center', va='center')
            
                VIZ_YOLO_RESULTS = True
                if args.visualize and VIZ_YOLO_RESULTS:
                    for ii, box_xywh in enumerate(cam_dets[cam_id]):
                        ll, tt, ww, hh = box_xywh.detach().cpu().numpy()
                        bb = tt + hh
                        fig_cam_subplots[cam_id].add_patch(
                            patches.Rectangle((ll, bb), ww, hh, fill=False, lw=3, ec=(1,1,1)))
                        fig_cam_subplots[cam_id].annotate(f"yolo_{ii}", (l, b), color='white',
                                                            weight='bold', fontsize=10, ha='center', va='center')
            bbox_xywhs_all.append(torch.Tensor(bbox_xywhs))
        
        # ## only include unoccluded images using YOLO results
        # for bbox_xywh, cam_det in zip(bbox_xywhs_all, cam_dets):
        #     costs = -compute_iou(bbox_xywh, cam_det)
        #     row_ind, col_ind = linear_sum_assignment(costs)
            
        #     for rid, cid in zip(row_ind, col_ind):
        #         if -costs[rid][cid] >= 0.3:
        #             bev2cam[rid] = cid
            
        # cam_imgs = raw_data[0].permute(0,2,3,1).detach().cpu().numpy()
        # confs = scores.repeat(6,1)
        # clss = torch.zeros((NUM_CAM, scores.shape[0]))
        # out_bboxs = strongsort.update(xy, bbox_xywhs, confs, clss, cam_imgs, args.bbox_len)
        
        # if len(out_bboxs) > 0:
        #     for j, (output, conf) in enumerate(zip(out_bboxs, scores)):
        #         bboxes = output[0:4]
        #         id = int(output[4])
        #         cls = output[5]
                
        #         for cam_id in range(NUM_CAM):
        #             l, b, r, t = get_bbox_lbrt(coord_mapper, foot_point, cam_id)
        #             w, h = r - l, t - b
            
        #             if args.visualize:
        #                 fig_cam_subplots[cam_id].add_patch(
        #                     patches.Rectangle((l, b), w, h, fill=False, lw=7, ec=colours[i % len(colours)]))
        #                 fig_cam_subplots[cam_id].annotate(f"{id}", (l, b), color='white',
        #                                                     weight='bold', fontsize=15, ha='center', va='center')
 
        if args.visualize:
            cam_filename = os.path.join(outdir, str(batch_idx).zfill(5) + '.jpg')
            fig_cam.savefig(cam_filename, bbox_inches='tight')
            plt.close(fig_cam)
            print(f"saved to {cam_filename}")
                    

def parse_args():
    parser = argparse.ArgumentParser(description='Multiview detector')
    
    # trainer
    # parser.add_argument('--cls_thres', type=float, default=0.01) #debug purpose
    parser.add_argument('--cls_thres', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--id_ratio', type=float, default=0)
    
    # dataset
    parser.add_argument('--configfile', type=str, 
                        # default='/home/kanya/MVDeTr/multiview_detector/datasets/configs/retail.yaml')
                        default='/home/kanyamo/MVDeTr/multiview_detector/datasets/configs/retail.yaml')
    parser.add_argument('-d', '--dataset', type=str, default='MMP', choices=['MMP'])
    parser.add_argument('--world_reduce', type=int, default=2)
    parser.add_argument('--world_kernel_size', type=int, default=5)
    parser.add_argument('--img_reduce', type=int, default=4)
    parser.add_argument('--img_kernel_size', type=int, default=10)
    parser.add_argument('--sample_freq', type=int, default=3, help='sample part of frames to save time')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='input batch size for training')
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    
    # model
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18', 'mobilenet'])
    parser.add_argument('--detection_model_path_home', type=str, 
                        # default='/home/kanya/MVDeTr/logs/MMP')
                        default='/home/kanyamo/MVDeTr/logs/MMP')
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
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    parser.add_argument('--augmentation', type=str2bool, default=True)

    # misc
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--dropcam', type=float, default=0.0)
    parser.add_argument('--outdir', type=str, default='./strongsort_outputs')
    
    # post-processing
    parser.add_argument('--nms_ths', type=float, default=20)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--bbox_len', type=int, default=5)
    parser.add_argument('--yolo_ths', type=float, default=0.3)
    
    # StrongSORT
    parser.add_argument('--strong_sort_weights', type=str, default='strong_sort/osnet_x0_25_msmt17.pt')
    parser.add_argument('--mc_lambda', type=float, default=0.995, help='matching with both appearance (1-mc_lambda) and motion cost')
    parser.add_argument('--ema_alpha', type=float, default=0.9, help='updates appearance state in an exponential moving average manner')
    parser.add_argument('--max_dist', type=float, default=0.2, help='matching threshold; samples with larger distances are considered an invalid match')
    parser.add_argument('--max_iou_distance', type=float, default=0.7, help='gating threshold; associations with cost larger than this value are disregarded')
    parser.add_argument('--max_age', type=float, default=30, help='max number of misses before a track is deleted')
    parser.add_argument('--n_init', type=int, default=3, help='number of frames that a track remains in initialization phase')
    parser.add_argument('--nn_budget', type=int, default=100, help='max size of apperance descriptors gallery')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
