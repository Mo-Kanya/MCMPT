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

from SORT.sort import *

IOU_THS = 0.001
NMS_THS = 5

def detect_and_track(model, logfp, data_loader, res_fpath=None, gt_fpath=None, visualize=False):
    losses = 0
    precision_s, recall_s = AverageMeter(), AverageMeter()
    all_res_list = []
    t0 = time.time()
    
    mot_tracker = Sort(iou_threshold=IOU_THS)  #FIXME: finetune threshold value
    colours = ['white', 'lightcoral', 'red', 'gray', 'pink', 'cyan', 'mediumorchid', 'orange', 'gold'] #used only for display
    # colours = np.random.rand(32, 3) #used only for display
    
    criterion = GaussianMSE().cuda()
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    if res_fpath is not None:
        # assert gt_fpath is not None
        if gt_fpath:
            raise Warning("No gt.txt provided for moda and modp evaluation.")
    for batch_idx, (data, map_gt, imgs_gt, frame) in enumerate(data_loader):
        #######
        # if batch_idx not in [20, 60, 100, 140, 180]:
        #     continue
        # if batch_idx>200:
        #     return
        #######

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
        loss = 0
        for img_res, img_gt in zip(imgs_res, imgs_gt):
            loss += criterion(img_res, img_gt.to(img_res.device), data_loader.dataset.img_kernel)
        loss = criterion(map_res, map_gt.to(map_res.device), data_loader.dataset.map_kernel) + \
                loss / len(imgs_gt) * args.alpha
        losses += loss.item()
        pred = (map_res > args.cls_thres).int().to(map_gt.device)
        true_positive = (pred.eq(map_gt) * pred.eq(1)).sum().item()
        false_positive = pred.sum().item() - true_positive
        false_negative = map_gt.sum().item() - true_positive
        precision = true_positive / (true_positive + false_positive + 1e-4)
        recall = true_positive / (true_positive + false_negative + 1e-4)
        precision_s.update(precision)
        recall_s.update(recall)

        if visualize:
            fig = plt.figure()
            subplt0 = fig.add_subplot(121, title=f"output_{batch_idx}")
            subplt1 = fig.add_subplot(122, title=f"target_{batch_idx}")
            subplt0.imshow(map_res.cpu().detach().numpy().squeeze())
            subplt1.imshow(criterion._traget_transform(map_res, map_gt, data_loader.dataset.map_kernel).cpu().detach().numpy().squeeze())
            # plt.savefig('./vis/map'+ str(batch_idx)+'.jpg')
            # plt.close(fig)
            
            # # visualizing the heatmap for per-view estimation
            # heatmap0_head = imgs_res[0][0, 0].detach().cpu().numpy().squeeze()
            # heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
            # cvgt = criterion._traget_transform(imgs_res[0], imgs_gt[0], data_loader.dataset.img_kernel)
            # gold_head = cvgt[0, 0].detach().cpu().numpy().squeeze()
            # gold_head = Image.fromarray((gold_head * 255).astype('uint8'))
            # gold_foot = cvgt[0, 1].detach().cpu().numpy().squeeze()
            # gold_foot = Image.fromarray((gold_foot * 255).astype('uint8'))
            # gold_foot.save('./vis/foot_label' + str(batch_idx) + '.jpg')
            # gold_head.save('./vis/head_label' + str(batch_idx) + '.jpg')
            # img0 = denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
            # img0 = Image.fromarray((img0 * 255).astype('uint8'))
            # head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
            # head_cam_result.save('./vis/cam1_head' + str(batch_idx) + '.jpg')
            # foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
            # foot_cam_result.save('./vis/cam1_foot' + str(batch_idx) + '.jpg')
            ######
        
        ids, count = nms(grid_xy.float(), scores[:,0], NMS_THS, np.inf)
        grid_xy = grid_xy[ids[:count], :]
        scores = scores[ids[:count]]
        
        detections = []
        for i, (fid, xy, score) in enumerate(zip(frame_id, grid_xy, scores)):
            # res[1:] /= dataloader.dataset.world_reduce
            # res[1:] -= offset[i]
            fid = int(fid.item())
            x, y = xy[0].item(), xy[1].item()
            detections.append(list([x-5,y-5,x+5,y+5,score.item()]))
            # subplt0.add_patch(patches.Rectangle((int(x),int(y)),5,5,fill=False,lw=3,ec=colours[fid%32,:]))
        detections = np.array(detections)
        
        track_bbs_ids = mot_tracker.update(detections)
        
        for d in track_bbs_ids:
            # track_bbs_idx: Nx5 (bb_x, bb_y, bb2_x, bb2_y, track_id)
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame[0],d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]))
            if visualize:
                d = d.astype(np.int32)
                subplt0.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=1,ec=colours[d[4]%len(colours)]))
                # subplt0.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=1,ec=colours[d[4]%32,:]))

        if visualize:
            filename = logfp + f'_nms{NMS_THS}_iou{IOU_THS}_' + str(batch_idx).zfill(5) +'.jpg'
            plt.savefig(filename)
            plt.close(fig)
            

    t1 = time.time()
    t_epoch = t1 - t0

    # if visualize:
    #     fig = plt.figure()
    #     subplt0 = fig.add_subplot(211, title="output")
    #     subplt1 = fig.add_subplot(212, title="target")
    #     subplt0.imshow(map_res.cpu().detach().numpy().squeeze())
    #     subplt1.imshow(criterion._traget_transform(map_res, map_gt, data_loader.dataset.map_kernel)
    #                     .cpu().detach().numpy().squeeze())
        # plt.savefig(os.path.join(logdir, 'map.jpg'))
        # plt.close(fig)

        # # visualizing the heatmap for per-view estimation
        # heatmap0_head = imgs_res[0][0, 0].detach().cpu().numpy().squeeze()
        # heatmap0_foot = imgs_res[0][0, 1].detach().cpu().numpy().squeeze()
        # img0 = denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
        # img0 = Image.fromarray((img0 * 255).astype('uint8'))
        # head_cam_result = add_heatmap_to_image(heatmap0_head, img0)
        # head_cam_result.save(os.path.join(logdir, 'cam1_head.jpg'))
        # foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
        # foot_cam_result.save(os.path.join(logdir, 'cam1_foot.jpg'))

    moda = 0
    if res_fpath is not None and gt_fpath is not None:
        all_res_list = torch.cat(all_res_list, dim=0)
        np.savetxt(os.path.abspath(os.path.dirname(res_fpath)) + '/all_res.txt', all_res_list.numpy(), '%.8f')
        res_list, score_list = [], []
        for frame in np.unique(all_res_list[:, 0]):
            res = all_res_list[all_res_list[:, 0] == frame, :]
            positions, scores = res[:, 1:3], res[:, 3]
            ids, count = nms(positions, scores, 20, np.inf)
            res_list.append(torch.cat([torch.ones([count, 1]) * frame, positions[ids[:count], :]], dim=1))
            score_list.append(scores[ids[:count]])
        res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
        np.savetxt(res_fpath, res_list, '%d')

        recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath), os.path.abspath(gt_fpath), "MMP")

        # If you want to use the unofiicial python evaluation tool for convenient purposes.
        # recall, precision, modp, moda = python_eval(os.path.abspath(res_fpath), os.path.abspath(gt_fpath),
        #                                             data_loader.dataset.base.__name__)

        print('moda: {:.1f}%, modp: {:.1f}%, precision: {:.1f}%, recall: {:.1f}%'.
                format(moda, modp, precision, recall))
        
        # detections = []
        # for i, (res, score) in enumerate(zip(res_list[batch_idx], score_list[batch_idx])):
        #     # res[1:] /= dataloader.dataset.world_reduce
        #     # res[1:] -= offset[i]
        #     frame_id, x, y = int(res[0].item()), res[1].item(), res[2].item()
        #     detections.append(list([x-1,y-1,x+1,y+1,score.item()]))
        #     # ax1.add_patch(patches.Rectangle((int(x),int(y)),5,5,fill=False,lw=3,ec=colours[frame_id%32,:]))
        # detections = np.array(detections)
        
        # mot_tracker = Sort() 
        # track_bbs_ids = mot_tracker.update(detections)
        
        # for d in track_bbs_ids:
        #     # track_bbs_idx: 37x5 (bb_x, bb_y, bb2_x, bb2_y, track_id)
        #     print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame[0],d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]))
        #     if visualize:
        #         d = d.astype(np.int32)
        #         subplt0.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
        
        # if visualize:
        #     # fig.canvas.flush_events()
        #     # plt.draw()
        #     # ax1.cla()
        #     # ax2.cla()
        #     filename = f'vis/track{batch_idx if batch_idx else ""}.jpg'
        #     plt.savefig(filename)
        #     plt.close(fig)

    print('Test, Loss: {:.6f}, Precision: {:.3f}%, Recall: {:.3f}, \tTime: {:.3f}'.format(
        losses / (len(data_loader) + 1), precision_s.avg * 100, recall_s.avg * 100, t_epoch))

    return losses / len(data_loader), precision_s.avg * 100, moda

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
    
    model_path = '/home/kanya/MVDet/logs/MMP_frame/default/2023-03-16_ep5_transfer/MultiviewDetector.pth'
    # model_path = '/home/kanya/MVDet/logs/MMP_frame/default/2023-03-26_03-18-03/MultiviewDetector.pth'
    model = PerspTransDetector(test_set, args.arch)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # trainer = PerspectiveTrainer(model, criterion, logdir, denormalize, args.cls_thres, args.alpha)
    
    logdir = 'SORT_outputs'
    os.makedirs(logdir, exist_ok=True)
    
    logfp = os.path.join(logdir, os.path.basename(os.path.dirname(model_path)))
    
    print('Test loaded model...')
    detect_and_track(model, logfp, test_loader, os.path.join(logdir, 'test.txt'), test_set.gt_fpath, True)
    
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
