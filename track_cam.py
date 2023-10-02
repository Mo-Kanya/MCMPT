# This version is based on MVDetr
import os

os.environ['OMP_NUM_THREADS'] = '1'
import random
import torch
from matplotlib import pyplot as plt
from PIL import Image
import csv
from ultralytics import YOLO

plt.axis('off')

from multiview_detector.datasets import *
from multiview_detector.models.mvdetr import MVDeTr

# from SORT.sort import *
from psort import *

from mmp_tracking_helper.mmp_mapping3D_2D_script import *
from multiview_detector.utils.decode import ctdet_decode, mvdet_decode
from multiview_detector.utils.nms import nms
from scipy.optimize import linear_sum_assignment

L2_THS = 8
NMS_THS = 5
BBOX_LEN = 10
AGE = 3  # same for max age and min hits for simplicity
HITS = 5
START_FRAME = 50
END_FRAME = 51
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


def sim_decode(scoremap):
    B, C, H, W = scoremap.shape
    xy = torch.nonzero(torch.ones_like(scoremap[:, 0])).view([B, H * W, 3])[:, :, [2, 1]].float()
    scores = scoremap.permute(0, 2, 3, 1).reshape(B, H * W, 1)

    return torch.cat([xy, scores], dim=2)


def detect_and_track_cam(model, detector, log_fpath, data_loader,
                     res_dir=None, visualize=False, coord_mapper=None):
    t0 = time.time()

    mot_tracker = Sort(max_age=AGE, min_hits=HITS, l2_threshold=L2_THS)
    colours = ['white', 'lightcoral', 'red', 'gray', 'pink', 'cyan', 'mediumorchid', 'orange',
               'gold']  # used only for display

    for batch_idx, (data, world_gt, imgs_gt, affine_mats, frame, raw_data) in enumerate(data_loader):
        if frame[0] < START_FRAME:
            continue
        if frame[0] == END_FRAME:
            exit()

        B, N = imgs_gt['heatmap'].shape[:2]
        data = data.cuda()
        for key in imgs_gt.keys():
            imgs_gt[key] = imgs_gt[key].view([B * N] + list(imgs_gt[key].shape)[2:])
        # with autocast():
        with torch.no_grad():
            (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = model(data, affine_mats)
            cam_det_results = detector(raw_data[0])

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
        track_bbs_ids = mot_tracker.update(detections)

        if not visualize:
            res_file = os.path.join(res_dir, 'topdown_' + str(batch_idx).zfill(5)+ '.csv')
            with open(res_file, 'w', newline='') as csvfile:
                res_writer = csv.writer(csvfile, delimiter=',')
                for trk in track_bbs_ids:
                    res_writer.writerow([int(trk[2])]+[trk[1]*2, trk[0]*2])

        cam_dets = []
        for cam_id, cam_det_res in enumerate(cam_det_results):
            bboxes = cam_det_res.boxes.xyxy.cpu().numpy()
            cats = cam_det_res.boxes.cls.cpu().numpy()
            cam_dets.append(bboxes[cats == 0])

        if visualize:
            fig_cam = plt.figure(figsize=(6.4 * 4, 4.8 * 3))
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0.05)
            fig_cam_subplots = []
            for cam_id in range(6):
                fig_cam_subplots.append(fig_cam.add_subplot(2, 3, cam_id + 1, title=f"Camera {cam_id + 1}"))
                img_cam = raw_data[0][cam_id].permute(1, 2, 0).detach().cpu().numpy()
                fig_cam_subplots[cam_id].imshow(img_cam)
                fig_cam_subplots[cam_id].set_axis_off()

            for cam_id in range(6):
                bev_det = []
                track_id = []
                for d in track_bbs_ids:
                    d = d.astype(np.int32)
                    foot_point = {'X': d[0] * data_loader.dataset.world_reduce,
                                  'Y': d[1] * data_loader.dataset.world_reduce}
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
                    track_id.append(d[-1])
                    bev_det.append(np.array([l, b, r, t]))

                costs = -compute_iou(np.stack(bev_det), cam_dets[cam_id])
                row_ind, col_ind = linear_sum_assignment(costs)

                bev2cam = {}
                for rid, cid in zip(row_ind, col_ind):
                    if -costs[rid][cid] >= 0.3:
                        bev2cam[rid] = cid
                for i, det in enumerate(bev_det):
                    if False: # i in bev2cam
                        # print(len(cam_dets), cam_dets[cam_id].shape, costs.shape)
                        box = cam_dets[cam_id][bev2cam[i]]
                        l, b, r, t = box[0], box[1], box[2], box[3]
                    else:
                        l, b, r, t = det[0], det[1], det[2], det[3]
                    
                    fig_cam_subplots[cam_id].add_patch(
                        patches.Rectangle((l, b), r-l, t-b, fill=False, lw=1, ec=colours[track_id[i] % len(colours)]))
                    fig_cam_subplots[cam_id].annotate(f"{track_id[i]}", (l, b), color='white',
                                                      weight='bold', fontsize=10, ha='center', va='center')

            # YOLO results
            # fig_cam_subplots.append(fig_cam.add_subplot(2, 3, cam_id + 1, title=f"Camera {cam_id + 1}"))
            # im_array = cam_det_results[cam_id].plot()  # plot a BGR numpy array of predictions
            # img_cam = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            # fig_cam_subplots[cam_id].imshow(img_cam)
            # fig_cam_subplots[cam_id].set_axis_off()

            cam_filename = log_fpath + f'_nms{NMS_THS}_bbox{BBOX_LEN}_' + str(
                batch_idx).zfill(5) + '_cam.jpg'
            fig_cam.savefig(cam_filename, bbox_inches='tight')
            plt.close(fig_cam)

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
    config_file = "/home/kanya/MVDeTr/multiview_detector/datasets/configs/retail.yaml"
    test_set = MMPframeDataset(config_file, train=False, world_reduce=args.world_reduce,
                               img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                               img_kernel_size=args.img_kernel_size, sample_freq=args.sample_freq)
    test_set.get_raw = True

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)

    model_path = '/home/kanya/MVDeTr/logs/MMP/aug_deform_trans_lr0.0005_baseR0.1_neck128_out0_alpha1.0_id0_drop0.0_dropcam0.0_worldRK2_5_imgRK4_10_2023-04-12_20-59-36/MultiviewDetector.pth'
    # model_path = '/home/kanyamo/MVDeTr/logs/MMP/aug_deform_trans_lr0.0005_baseR0.1_neck128_out0_alpha1.0_id0_drop0.0_dropcam0.0_worldRK2_5_imgRK4_10_2023-04-12_20-59-36/MultiviewDetector.pth'
    model = MVDeTr(test_set, args.arch, world_feat_arch=args.world_feat,
                   bottleneck_dim=128, outfeat_dim=0, droupout=0.0).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    detector = YOLO('yolov8n.pt').to("cuda")

    logdir = 'SORT_outputs'
    os.makedirs(logdir, exist_ok=True)
    logfp = os.path.join(logdir, os.path.basename(os.path.dirname(model_path)))

    coordmap = CoordMapper(test_set.calibration_path)

    print('Test loaded model...')
    detect_and_track_cam(model=model,
                         detector=detector,
                         log_fpath=logfp,
                         data_loader=test_loader,
                         res_dir=os.path.join(logdir),
                         visualize=args.visualize,
                         coord_mapper=coordmap)


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
