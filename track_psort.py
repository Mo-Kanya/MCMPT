import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import time
import shutil
from distutils.dir_util import copy_tree
import datetime

import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from matplotlib import pyplot as plt
from PIL import Image

plt.axis('off')

from multiview_detector.datasets import *
from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.models.image_proj_variant import ImageProjVariant
from multiview_detector.models.res_proj_variant import ResProjVariant
from multiview_detector.models.no_joint_conv_variant import NoJointConvVariant
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.image_utils import img_color_denormalize, add_heatmap_to_image
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.nms import nms
from multiview_detector.trainer import PerspectiveTrainer
from multiview_detector.evaluation.evaluate import evaluate

# from SORT.sort import *
from psort import *

from mmp_tracking_helper.mmp_mapping3D_2D_script import *

L2_THS = 8
NMS_THS = 5
PERSON_WIDTH = 30
PERSON_HEIGHT = 160
BBOX_LEN = 10


def mvdet_decode(scoremap):
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
                     res_fpath=None, gt_fpath=None,
                     visualize=False,
                     num_cam=0, camera_intrinsic=None, camera_extrinsic=None,
                     worldgrid2worldcoord_mat=None,
                     coord_mapper=None):
    losses = 0
    precision_s, recall_s = AverageMeter(), AverageMeter()
    all_res_list = []
    t0 = time.time()

    mot_tracker = Sort(max_age=5, min_hits=5, l2_threshold=L2_THS)  # FIXME: finetune threshold value
    colours = ['white', 'lightcoral', 'red', 'gray', 'pink', 'cyan', 'mediumorchid', 'orange',
               'gold']  # used only for display
    # colours = np.random.rand(32, 3) #used only for display

    criterion = GaussianMSE().cuda()
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if res_fpath is not None:
        # assert gt_fpath is not None
        if gt_fpath:
            raise Warning("No gt.txt provided for moda and modp evaluation.")

    for batch_idx, (data, map_gt, imgs_gt, frame) in enumerate(data_loader):
        if frame[0] == 75:
            exit()
        with torch.no_grad():
            map_res, imgs_res = model(data)

        if res_fpath is not None:
            # if res_fpath is not None and gt_fpath is not None:
            map_grid_res = map_res.detach().cpu().squeeze()
            v_s = map_grid_res[map_grid_res > args.cls_thres].unsqueeze(1)
            grid_ij = (map_grid_res > args.cls_thres).nonzero()
            if data_loader.dataset.indexing == 'xy':
                grid_xy = grid_ij[:, [1, 0]]
            else:
                grid_xy = grid_ij

            frame_id = torch.ones_like(v_s) * frame
            world_xy = grid_xy * data_loader.dataset.grid_reduce
            scores = v_s
            res = torch.cat([frame_id, world_xy.float(), scores], dim=1)
            all_res_list.append(res)

        ## Evaluation Metrics
        # loss = 0
        # for img_res, img_gt in zip(imgs_res, imgs_gt):
        #     loss += criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.img_kernel)
        # loss = criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel) + \
        #         loss / len(imgs_gt) * args.alpha
        # losses += loss.item()
        # pred = (map_res > args.cls_thres).int().to(map_gt.device)
        # true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
        # false_positive = pred.sum().item() - true_positive
        # false_negative = map_gt.sum().item() - true_positive
        # precision = true_positive / (true_positive + false_positive + 1e-4)
        # recall = true_positive / (true_positive + false_negative + 1e-4)
        # precision_s.update(precision)
        # recall_s.update(recall)

        if visualize:
            fig = plt.figure()
            subplt0 = fig.add_subplot(121, title=f"output_{batch_idx}")
            subplt1 = fig.add_subplot(122, title=f"target_{batch_idx}")
            subplt0.imshow(map_res.cpu().detach().numpy().squeeze())
            subplt1.imshow(criterion._traget_transform(map_res, map_gt,
                                                       data_loader.dataset.map_kernel).cpu().detach().numpy().squeeze())
            subplt0.set_axis_off()
            subplt1.set_axis_off()
            # plt.savefig('./vis/map'+ str(batch_idx)+'.jpg')
            # plt.close(fig)

        ## NMS before tracking
        pred = map_res
        # TODO: try if a sigmoid is needed
        xys = mvdet_decode(torch.sigmoid(pred.detach().cpu()))
        positions, scores = xys[:, :, :2], xys[:, :, 2:3]
        ids = scores.squeeze() > 0.4
        pos, s = positions[0, ids], scores[0, ids, 0]
        ids, count = nms(pos, s, 12.5, top_k=800)
        count = min(count, 20)
        ids = ids[:count]
        grid_xy = torch.stack([ids % 212, ids // 212]).T

        # ids, count = nms(grid_xy.float(), scores[:,0], NMS_THS, np.inf)
        # grid_xy = grid_xy[ids[:count], :]
        scores = scores[0, ids, 0]

        ## Pre-process detection results for tracking
        ## FIXME: arbitrary bounding box created from point with width=10px, height=10px
        detections = []
        for i, (fid, xy, xy_world, score) in enumerate(zip(frame_id, grid_xy, world_xy, scores)):
            # res[1:] /= dataloader.dataset.world_reduce
            # res[1:] -= offset[i]
            fid = int(fid.item())
            x, y = xy[0].item(), xy[1].item()
            # print(score.shape)
            detections.append(list([x, y, score.item()]))

        detections = np.array(detections)

        ## Update tracking with current detections
        track_bbs_ids = mot_tracker.update(detections)
        for d in track_bbs_ids:
            ## track_bbs_idx: Nx5 (bb_x, bb_y, bb2_x, bb2_y, track_id)
            # print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame[0], d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]))
            if visualize:
                d = d.astype(np.int32)
                subplt0.add_patch(patches.Rectangle((d[0] - BBOX_LEN // 2, d[1] - BBOX_LEN // 2),
                                                    BBOX_LEN, BBOX_LEN,
                                                    fill=False, lw=1, ec=colours[d[2] % len(colours)]))
                # subplt0.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=1,ec=colours[d[4]%32,:]))

        if visualize:
            topdown_filename = log_fpath + f'_nms{NMS_THS}_l2{L2_THS}_' + str(batch_idx).zfill(5) + '_bev.jpg'
            fig.savefig(topdown_filename)
            plt.close(fig)

    t1 = time.time()
    t_epoch = t1 - t0

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
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([360, 640]), T.ToTensor(), normalize, ])
    # if 'wildtrack' in args.dataset:
    #     data_path = os.path.expanduser('../Data/Wildtrack')
    #     base = Wildtrack(data_path)
    # elif 'multiviewx' in args.dataset:
    #     data_path = os.path.expanduser('../Data/MultiviewX')
    #     base = MultiviewX(data_path)
    # else:
    #     raise Exception('must choose from [wildtrack, multiviewx]')
    # train_set = frameDataset(base, train=True, transform=train_trans, grid_reduce=4)
    # test_set = frameDataset(base, train=False, transform=train_trans, grid_reduce=4)

    config_file = "/home/kanya/MVDet/multiview_detector/datasets/configs/retail.yaml"
    # train_set = MMPframeDataset(config_file, train=True, sample_freq=args.sample_freq)
    # test_set = MMPframeDataset(config_file, train=False, sample_freq=args.sample_freq)
    test_set = MMPframeDataset(config_file, train=False, sample_freq=args.sample_freq)
    # test_set = MMPframeDataset(config_file, train=False, sample_freq=args.sample_freq, img_reduce=4)

    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
    #                                            num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    model_path = '/home/kanya/MVDet/logs/MMP_frame/default/ep10/MultiviewDetector.pth'
    # model_path = '/home/kanya/MVDet/logs/MMP_frame/default/2023-03-26_03-18-03/MultiviewDetector.pth'
    model = PerspTransDetector(test_set, args.arch)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # trainer = PerspectiveTrainer(model, criterion, logdir, denormalize, args.cls_thres, args.alpha)

    coordmap = CoordMapper(test_set.calibration_path)

    logdir = 'SORT_outputs'
    os.makedirs(logdir, exist_ok=True)
    logfp = os.path.join(logdir, os.path.basename(os.path.dirname(model_path)))

    print('Test loaded model...')
    detect_and_track(model=model,
                     log_fpath=logfp,
                     data_loader=test_loader,
                     res_fpath=os.path.join(logdir, 'test.txt'),
                     gt_fpath=test_set.gt_fpath,
                     visualize=True,
                     num_cam=test_set.num_cam,
                     camera_intrinsic=test_set.intrinsic_matrices,
                     camera_extrinsic=test_set.extrinsic_matrices,
                     worldgrid2worldcoord_mat=test_set.worldgrid2worldcoord_mat,
                     coord_mapper=coordmap)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18'])
    parser.add_argument('-d', '--dataset', type=str, default='MMP', choices=['MMP', 'wildtrack', 'multiviewx'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    parser.add_argument('--sample_freq', type=int, default=1, help='sample part of frames to save time')
    args = parser.parse_args()

    main(args)
