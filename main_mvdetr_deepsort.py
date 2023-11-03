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

from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.utils import visualization
from deep_sort.detection import Detection
from deep_sort.generate_detections import create_box_encoder

from mmp_tracking_helper.mmp_mapping3D_2D_script import *

BODY_HEIGHT = 1600
BODY_WIDTH = 150

def compute_iou(boxes1, boxes2):
    """
    Input:
        boxes1: Nx4 ndarray, representing N bounding boxes coordinates
        boxes2: Nx4 ndarray, representing N bounding boxes coordinates
    Output:
        iou_mat: NxM ndarray, with iou_mat[i, j] = iou(boxes1[i], boxes2[j])
    """
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    iou_mat = np.zeros((N, M))
    box2areas = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    # print( box2areas.shape)

    for i in range(N):
        box1 = boxes1[i]

        Iwidth = np.minimum(box1[2], boxes2[:, 2]) - np.maximum(box1[0], boxes2[:, 0])
        Iwidth = np.maximum(Iwidth, 0)
        Iheight = np.minimum(box1[3], boxes2[:, 3]) - np.maximum(box1[1], boxes2[:, 1])
        Iheight = np.maximum(Iheight, 0)
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
    num_cam = train_set.num_cam
    
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
    print('Loaded detection model...')
    
    yolo = YOLO('yolov8n.pt').to("cuda")
    
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", args.max_cosine_distance, args.nn_budget)
    tracker = Tracker(metric)
    encoder = create_box_encoder(args.encoder, batch_size=args.batch_size)
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
        
        cam_dets = []
        for cam_id, cam_det_res in enumerate(cam_det_results):
            bboxes = cam_det_res.boxes.xyxy.cpu().numpy()
            cats = cam_det_res.boxes.cls.cpu().numpy()
            cam_dets.append(bboxes[cats == 0])
        
        detections = []
        
        features = []
        if len(ids) > 0:
            
            
            
            
            ## DEBUGGING
            fig_cam = plt.figure(figsize=(6.4 * 4, 4.8 * 3))
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0.05)
            fig_cam_subplots = []
            for cam_id_ in range(num_cam):
                fig_cam_subplots.append(fig_cam.add_subplot(2, num_cam//2, cam_id_ + 1, title=f"Camera {cam_id_ + 1}"))
                img_cam = raw_data[0][cam_id_].permute(1, 2, 0).detach().cpu().numpy()
                fig_cam_subplots[cam_id_].imshow(img_cam)
                fig_cam_subplots[cam_id_].set_axis_off()
                    
                    
                   
            print("@@@ detected!")
            for xy in grid_xy:
                xy = xy.to(int)
                foot_point = {'X': xy[0].item() * train_loader.dataset.world_reduce,
                                'Y': xy[1].item() * train_loader.dataset.world_reduce}
                x_raw, y_raw = [], []

                ## project back to camera frames
                projected_bbox = []
                for cam_id_ in range(num_cam):
                    for offset in [[BODY_WIDTH, BODY_WIDTH, 0], [-BODY_WIDTH, -BODY_WIDTH, 0],
                                    [BODY_WIDTH, -BODY_WIDTH, 0], [-BODY_WIDTH, BODY_WIDTH, 0],
                                    [BODY_WIDTH, BODY_WIDTH, BODY_HEIGHT], [-BODY_WIDTH, -BODY_WIDTH, BODY_HEIGHT],
                                    [BODY_WIDTH, -BODY_WIDTH, BODY_HEIGHT], [-BODY_WIDTH, BODY_WIDTH, BODY_HEIGHT]]:
                        x_cam, y_cam = coord_mapper.projection(foot_point, cam_id_+1, body_offset=offset)
                        x_raw.append(x_cam)
                        y_raw.append(y_cam)
                    l, r = min(x_raw), max(x_raw)
                    b, t = min(y_raw), max(y_raw)
                    w, h = r-l, t-b
                    
                    
                    ## DEBUGGING
                    print([l,b,r,t])
                    fig_cam_subplots[cam_id_].add_patch(
                        patches.Rectangle((l, b), w, h, fill=False, lw=3, ec=colours[batch_idx % len(colours)]))
                    fig_cam_subplots[cam_id_].annotate(f"{batch_idx}", (l, b), color='white',
                                                        weight='bold', fontsize=10, ha='center', va='center')

                    
                    projected_bbox.append(np.array([l, b, w, h]))
                projected_bbox = np.array(projected_bbox)
                
                # TODO: cam_id should be within the for loop?
                ## extract features
                cam_image = raw_data[0][cam_id].permute(1,2,0).detach().cpu().numpy()
                bbox_features = encoder(cam_image, projected_bbox.copy())
                
                
                ## aggregate features of multiple views
                mean_feature = np.mean(bbox_features, axis=0)
                features.append(mean_feature)
            
            
            ## DEBUGGING
            cam_filename = os.path.join("debug", str(batch_idx).zfill(5) + '_init.jpg')
            fig_cam.savefig(cam_filename, bbox_inches='tight')
            plt.close(fig_cam)
            print(f"saved to {cam_filename}")
        
        ## preprocess for tracker
        assert len(scores) == len(grid_xy) == len(features)
        for (xy, id, score, feature) in zip(grid_xy, ids, scores, features):
            x, y = xy[0].item(), xy[1].item()
            bbox = [x-args.bbox_len, y-args.bbox_len, x+args.bbox_len, y+args.bbox_len]
            detections.append(Detection(bbox, score.item(), feature))
        
        tracker.predict()
        tracker.update(detections)
        
        fig_cam = plt.figure(figsize=(6.4 * 4, 4.8 * 3))
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0.05)
        fig_cam_subplots = []
        for cam_id in range(num_cam):
            fig_cam_subplots.append(fig_cam.add_subplot(2, num_cam//2, cam_id + 1, title=f"Camera {cam_id + 1}"))
            img_cam = raw_data[0][cam_id].permute(1, 2, 0).detach().cpu().numpy()
            fig_cam_subplots[cam_id].imshow(img_cam)
            fig_cam_subplots[cam_id].set_axis_off()
        
        for cam_id in range(num_cam):
            bev_det = []
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                
                bbox = track.to_tlwh()
                track_id = track.track_id
                
                foot_point = {'X': bbox[0] * train_loader.dataset.world_reduce,
                                'Y': bbox[1] * train_loader.dataset.world_reduce}
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
                w, h = r-l, t-b
                # print(l ,b, w, h)
                bev_det.append(np.array([l, b, r, t]))
            
            costs = -compute_iou(np.stack(bev_det), cam_dets[cam_id])
            row_ind, col_ind = linear_sum_assignment(costs)
            
            bev2cam = {}
            for rid, cid in zip(row_ind, col_ind):
                if -costs[rid][cid] >= args.yolo_ths:
                    bev2cam[rid] = cid
            for i, det in enumerate(bev_det):
                if i in bev2cam:
                    box = cam_dets[cam_id][bev2cam[i]]
                    l, b, r, t = box
                else:
                    l, b, r, t = det
                
                fig_cam_subplots[cam_id].add_patch(
                    patches.Rectangle((l, b), r-l, t-b, fill=False, lw=3, ec=colours[track_id % len(colours)]))
                fig_cam_subplots[cam_id].annotate(f"{track_id}", (l, b), color='white',
                                                    weight='bold', fontsize=10, ha='center', va='center')

        cam_filename = os.path.join(outdir, str(batch_idx).zfill(5) + '_cam.jpg')
        fig_cam.savefig(cam_filename, bbox_inches='tight')
        plt.close(fig_cam)
        print(f"saved to {cam_filename}")
        
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
    parser.add_argument('--outdir', type=str, default='./deepsort_outputs')
    
    # post-processing
    parser.add_argument('--nms_ths', type=float, default=20)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--bbox_len', type=int, default=5)
    parser.add_argument('--yolo_ths', type=float, default=0.3)
    
    # DeepSORT
    parser.add_argument('--encoder', type=str, default='./deep_sort/mars-small128.pb')
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
