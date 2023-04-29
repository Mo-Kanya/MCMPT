# will work under MVDetr
import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import time
import shutil
from distutils.dir_util import copy_tree
import datetime
import random
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from matplotlib import pyplot as plt
from PIL import Image
import csv

plt.axis('off')

from multiview_detector.datasets import *
from multiview_detector.models.mvdetr import MVDeTr

# from SORT.sort import *
from psort import *

from mmp_tracking_helper.mmp_mapping3D_2D_script import *
from multiview_detector.utils.decode import ctdet_decode, mvdet_decode
from multiview_detector.utils.nms import nms

L2_THS = 8
NMS_THS = 5
BBOX_LEN = 10
AGE = 3  # same for max age and min hits for simplicity
HITS = 5
END_FRAME = -1

def sim_decode(scoremap):
    B, C, H, W = scoremap.shape
    xy = torch.nonzero(torch.ones_like(scoremap[:, 0])).view([B, H * W, 3])[:, :, [2, 1]].float()
    scores = scoremap.permute(0, 2, 3, 1).reshape(B, H * W, 1)

    return torch.cat([xy, scores], dim=2)


def project_topdown2camera(topdown_coords, cam_id, cam_intrinsic, cam_extrinsic, worldgrid2worldcoord_mat):
    worldcoord2imgcoord_mat = cam_intrinsic @ cam_extrinsic  # np.delete(, 2, 1)
    worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
    worldgrid2imgcoord_mat = np.delete(worldgrid2imgcoord_mat, 2, 1)

    # worldcoord2imgcoord_mat = cam_intrinsic @ np.delete(cam_extrinsic, 2, 1)
    # worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat

    rot_m = cam_extrinsic[:, :3]
    tran_m = np.linalg.inv(-rot_m) @ cam_extrinsic[:, -1]
    # self.extrinsic_matrices[cam_id][:, :3] = rot_m
    # self.extrinsic_matrices[cam_id][:, -1] = -rot_m @ np.asarray(camera_param['ExtrinsicParameters']['Translation'])

    Fx = cam_intrinsic[0, 0]
    Fy = cam_intrinsic[1, 1]
    FInv = np.asarray([1 / Fx, 1 / Fy, 1])
    # self.intrinsic_matrices[cam_id][0, 0] = camera_param['IntrinsicParameters']['Fx']
    # self.intrinsic_matrices[cam_id][1, 1] = camera_param['IntrinsicParameters']['Fy']
    # self._camera_parameters[cam_id]['FInv'] = np.asarray([
    #             1 / camera_param['IntrinsicParameters']['Fx'],
    #             1 / camera_param['IntrinsicParameters']['Fy'], 1
    #         ])
    Cx = cam_intrinsic[0, 2]
    Cy = cam_intrinsic[1, 2]
    C = np.asarray([Cx, Cy, 1])

    world_coord = np.asarray([topdown_coords[0], topdown_coords[1], 0])
    uvw = np.linalg.inv(rot_m) @ (world_coord - tran_m)
    pixel_coords = uvw / FInv / uvw[2] + C

    return pixel_coords[0], pixel_coords[1]
    # topdown_coords = np.transpose(
    #     np.asarray([[person_center['X'], person_center['Y'], 0]]))
    # world_coord = topdown_coords / self._discretization_factor[:, np.newaxis] + self._min_volume[:, np.newaxis]
    # uvw = np.linalg.inv(self._camera_parameters[camera_id]['Rotation']) @ (
    #     world_coord - self._camera_parameters[camera_id]['Translation'][:, np.newaxis])
    # pixel_coords = uvw / self._camera_parameters[camera_id]['FInv'][:, np.newaxis] / uvw[
    #     2, :] + self._camera_parameters[camera_id]['C'][:, np.newaxis]
    # return pixel_coords[0][0], pixel_coords[1][0]


def detect_and_track(model, log_fpath, data_loader,
                     res_dir=None, visualize=False):
    t0 = time.time()

    mot_tracker = Sort(max_age=AGE, min_hits=HITS, l2_threshold=L2_THS)
    colours = ['white', 'lightcoral', 'red', 'gray', 'pink', 'cyan', 'mediumorchid', 'orange',
               'gold']  # used only for display

    for batch_idx, (data, world_gt, imgs_gt, affine_mats, frame) in enumerate(data_loader):
        if frame[0] == END_FRAME:
            exit()

        B, N = imgs_gt['heatmap'].shape[:2]
        data = data.cuda()
        for key in imgs_gt.keys():
            imgs_gt[key] = imgs_gt[key].view([B * N] + list(imgs_gt[key].shape)[2:])
        # with autocast():
        with torch.no_grad():
            (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = model(data, affine_mats)

        if visualize:
            fig = plt.figure()
            subplt0 = fig.add_subplot(121, title=f"output_{batch_idx}")
            subplt1 = fig.add_subplot(122, title=f"target_{batch_idx}")
            subplt1.imshow(world_gt['heatmap'].squeeze())
            subplt0.imshow(world_heatmap.cpu().detach().numpy().squeeze())
            subplt0.set_axis_off()
            subplt1.set_axis_off()

        # post-processing
        xys = mvdet_decode(torch.sigmoid(world_heatmap.detach().cpu()), world_offset.detach().cpu(),
                           reduce=data_loader.dataset.world_reduce)
        positions, scores = xys[:, :, :2], xys[:, :, 2:3]
        ids = scores.squeeze() > args.cls_thres
        pos, s = positions[0, ids], scores[0, ids, 0]
        ids, count = nms(pos, s, 20, top_k=np.inf)
        count = min(count, 20)
        grid_xy = pos[ids[:count]] / 2.  # [::-1, :]
        scores = s[ids[:count]]
        # ids = torch.stack([ids[:, 1], ids[:, 0]], dim=-1)
        # ids = torch.stack([ids % 212, ids // 212]).T

        # xys_gt = sim_decode(world_gt["heatmap"] == torch.max(world_gt["heatmap"])).detach().cpu()
        # positions_gt, scores_gt = xys_gt[:, :, :2], xys_gt[:, :, 2:3]
        # ids_gt = scores_gt.squeeze() == 1
        # pos_gt = positions_gt[0, ids_gt]

        detections = []
        for i, (xy, score) in enumerate(zip(grid_xy, scores)):
            x, y = xy[0].item(), xy[1].item()
            detections.append(list([x, y, score.item()]))

        detections = np.array(detections)

        ## Update tracking with current detections
        track_bbs_ids = mot_tracker.update(detections)

        if not visualize:
            res_file = os.path.join(res_dir, 'topdown_' + str(batch_idx).zfill(5)+ '.csv')
            with open(res_file, 'w', newline='') as csvfile:
                res_writer = csv.writer(csvfile, delimiter=',')
                for trk in track_bbs_ids:
                    res_writer.writerow([int(trk[2])]+[trk[1]*2, trk[0]*2])

        if visualize:
            for d in track_bbs_ids:
                d = d.astype(np.int32)
                subplt0.add_patch(patches.Rectangle((d[0] - BBOX_LEN // 2, d[1] - BBOX_LEN // 2),
                                                    BBOX_LEN, BBOX_LEN,
                                                    fill=False, lw=1, ec=colours[d[2] % len(colours)]))

            topdown_filename = log_fpath + f'_nms{NMS_THS}_l2{L2_THS}_' + str(batch_idx).zfill(5) + '_bev.jpg'
            fig.savefig(topdown_filename)
            plt.close(fig)

    return


def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    config_file = "/mnt/Data/MVDeTr/multiview_detector/datasets/configs/retail_eval.yaml"
    test_set = MMPframeDataset(config_file, train=False, world_reduce=args.world_reduce,
                               img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                               img_kernel_size=args.img_kernel_size, sample_freq=args.sample_freq)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)

    # model_path = '/home/kanya/MVDet/logs/MMP_frame/default/ep10/MultiviewDetector.pth'
    model_path = '/mnt/Data/MVDeTr/logs/MMP/ep10_wk5/MultiviewDetector.pth'
    model = MVDeTr(test_set, args.arch, world_feat_arch=args.world_feat,
                   bottleneck_dim=128, outfeat_dim=0, droupout=0.0).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    logdir = 'SORT_outputs'
    os.makedirs(logdir, exist_ok=True)
    logfp = os.path.join(logdir, os.path.basename(os.path.dirname(model_path)))

    print('Test loaded model...')
    detect_and_track(model=model,
                     log_fpath=logfp,
                     data_loader=test_loader,
                     res_dir=os.path.join(logdir),
                     visualize=args.visualize)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Point Tracker')
    parser.add_argument('--world_reduce', type=int, default=2)
    parser.add_argument('--world_kernel_size', type=int, default=5)
    parser.add_argument('--img_reduce', type=int, default=4)
    parser.add_argument('--img_kernel_size', type=int, default=10)
    parser.add_argument('--world_feat', type=str, default='deform_trans',
                        choices=['conv', 'trans', 'deform_conv', 'deform_trans', 'aio'])
    parser.add_argument('--cls_thres', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18'])
    parser.add_argument('-d', '--dataset', type=str, default='MMP', choices=['MMP', 'wildtrack', 'multiviewx'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    parser.add_argument('--sample_freq', type=int, default=1, help='sample part of frames to save time')
    args = parser.parse_args()

    main(args)
