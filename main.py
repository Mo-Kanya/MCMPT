# This version is based on MVDetr
import os
import json
import random
import warnings
from datetime import datetime
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings("ignore")

import torch
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
plt.axis('off')

from ultralytics import YOLO

from multiview_detector.datasets import *
from multiview_detector.models.mvdetr import MVDeTr

# from SORT.sort import *
from psort import *

from multiview_detector.utils.decode import ctdet_decode, mvdet_decode
from multiview_detector.utils.nms import nms
from mmp_tracking_helper.mmp_mapping3D_2D_script import *

from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from reid.reid_multibackend import ReIDDetectMultiBackend

START_FRAME = 0
END_FRAME = -1

def crop_pixel(x, limit):
    return min(max(round(x), 0), limit)

def get_features(bbox_xyxy, ori_img, reid_model):
    im_crops = []
    for box in bbox_xyxy:
        x1, y1, x2, y2 = box
        im = ori_img[:, crop_pixel(y1,320):crop_pixel(y2,320), crop_pixel(x1,640):crop_pixel(x2,640)]
        im_crops.append(im)
    if im_crops:
        features = reid_model(im_crops)
    else:
        features = np.array([])
    return features

def project_box3d(foot_point, coord_mapper, cam_id, body_width=150, body_height=1600):
    x_raw, y_raw = [], []
    for offset in [[body_width, body_width, 0], [-body_width, -body_width, 0],
                   [body_width, -body_width, 0], [-body_width, body_width, 0],
                   [body_width, body_width, body_height], [-body_width, -body_width, body_height],
                   [body_width, -body_width, body_height], [-body_width, body_width, body_height]]:
        x_cam, y_cam = coord_mapper.projection(foot_point, cam_id + 1, body_offset=offset)
        x_raw.append(x_cam)
        y_raw.append(y_cam)
    l, r = min(x_raw), max(x_raw)
    b, t = min(y_raw), max(y_raw)

    return l, r, b, t


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


def detect_and_track_cam(model, detector, tracker, data_loader, res_dir,
                         visualize=False, coord_mapper=None):
    NUM_CAM = data_loader.dataset.num_cam
    
    reid_model = ReIDDetectMultiBackend(weights=args.reid_modelpath, device='cuda:0', fp16=False)
    colours = ['white', 'lightcoral', 'red', 'gray', 'pink', 'cyan', 'mediumorchid', 'orange', 'gold']  # used only for display
    cnt_ma, cnt_mb = 0,0

    for batch_idx, (data, world_gt, imgs_gt, affine_mats, frame, raw_data) in enumerate(data_loader):
        if frame[0] < START_FRAME:
            continue
        if frame[0] == END_FRAME:
            exit()

        B, N = imgs_gt['heatmap'].shape[:2]
        data = data.cuda()
        for key in imgs_gt.keys():
            imgs_gt[key] = imgs_gt[key].view([B * N] + list(imgs_gt[key].shape)[2:])
        
        with torch.no_grad():
            (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = model(data, affine_mats)
            cam_det_results = detector(raw_data[0])

        # post-processing
        xys = mvdet_decode(torch.sigmoid(world_heatmap.detach().cpu()), world_offset.detach().cpu(),
                           reduce=data_loader.dataset.world_reduce)
        positions, scores = xys[:, :, :2], xys[:, :, 2:3]
        ids = scores.squeeze() > args.cls_thres
        pos, s = positions[0, ids], scores[0, ids, 0]
        ids, count = nms(pos, s, args.nms_ths, top_k=np.inf)
        count = min(count, args.min_count)
        grid_xy = pos[ids[:count]] / 2.  # [::-1, :]
        scores = s[ids[:count]]

        cam_dets = []
        for cam_id, cam_det_res in enumerate(cam_det_results):
            bboxes = cam_det_res.boxes.xyxy.cpu().numpy()
            cats = cam_det_res.boxes.cls.cpu().numpy()
            cam_dets.append(bboxes[cats == 0])

        # single view refinement: remove false positives
        detections = []
        for i, (xy, score) in enumerate(zip(grid_xy, scores)):
            x, y = xy[0].item(), xy[1].item()
            foot_point = {'X': x * data_loader.dataset.world_reduce,
                          'Y': y * data_loader.dataset.world_reduce}
            
            max_iou = 0
            mp_feature = []
            for i in range(NUM_CAM):
                if len(cam_dets[i]) == 0: 
                    continue
                
                l, r, b, t = project_box3d(foot_point, coord_mapper, i, args.body_width, args.body_height)
                bev_det_raw = np.array([[l, b, r, t]])
                
                ious = compute_iou(bev_det_raw, cam_dets[i])
                max_iou = max(max_iou, np.max(ious))
                
                if np.max(ious) >= args.yolo_iou_ths:
                    with torch.no_grad():
                        appearance_feature = get_features(bev_det_raw, raw_data[0,i,:,:,:], reid_model)[0].cpu().numpy()
                        mp_feature.append(appearance_feature)
            
            if len(mp_feature) == 0: 
                continue
            
            mp_feature = np.stack(mp_feature) # (# valid views, 512)
            mp_feature = np.mean(mp_feature, axis=0) # TODO
            detections.append( Detection([x, y, 1, 150], score.item(), mp_feature) )

        tracker.predict()
        num_ma, num_mb = tracker.update(detections)
        cnt_ma += num_ma
        cnt_mb += num_mb

        if not visualize:
            res_file = os.path.join(res_dir, 'topdown_' + str(batch_idx).zfill(5) + '.csv')
            with open(res_file, 'w', newline='') as csvfile:
                res_writer = csv.writer(csvfile, delimiter=',')
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.mean[:4]
                    res_writer.writerow([track.track_id] + [bbox[1]*2, bbox[0]*2])

        if visualize:
            fig_cam = plt.figure(figsize=(6.4 * 4, 4.8 * 3))
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0.05)
            
            fig_cam_subplots = []
            for cam_id in range(NUM_CAM):
                fig_cam_subplots.append(fig_cam.add_subplot(2, 3, cam_id + 1, title=f"Camera {cam_id + 1}"))
                img_cam = raw_data[0][cam_id].permute(1, 2, 0).detach().cpu().numpy()
                fig_cam_subplots[cam_id].imshow(img_cam)
                fig_cam_subplots[cam_id].set_axis_off()

            res_dicts = {}
            for cam_id in range(NUM_CAM):
                bev_det, track_ids = [], []                    
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    
                    d = track.mean[:4]
                    d = d.astype(np.int32)
                    foot_point = {'X': d[0] * data_loader.dataset.world_reduce,
                                  'Y': d[1] * data_loader.dataset.world_reduce}
                    l, r, b, t = project_box3d(foot_point, coord_mapper, cam_id, args.body_width, args.body_height)

                    track_ids.append(track.track_id)
                    bev_det.append(np.array([l, b, r, t]))
                
                if len(bev_det) == 0:
                    # res_file = os.path.join(res_dir, 'rgb_' + str(batch_idx).zfill(5) + '_' + str(cam_id + 1) + '.json')
                    # with open(res_file, 'w') as fp:
                    #     json.dump({}, fp)
                    continue

                costs = -compute_iou(np.stack(bev_det), cam_dets[cam_id])
                row_ind, col_ind = linear_sum_assignment(costs)

                bev2cam, res_dict = {}, {}
                for rid, cid in zip(row_ind, col_ind):
                    if -costs[rid][cid] >= args.cost_ths:
                        bev2cam[rid] = cid
                for i, det in enumerate(bev_det):
                    if i in bev2cam:
                        box = cam_dets[cam_id][bev2cam[i]]
                        l, b, r, t = box[0], box[1], box[2], box[3]
                    else:
                        l, b, r, t = det[0], det[1], det[2], det[3]
                    res_dict["track_id " + str(track_ids[i])] = [round(l), round(b), round(r), round(t)]

                    fig_cam_subplots[cam_id].add_patch(
                        patches.Rectangle((l, b), r - l, t - b, fill=False, lw=3,
                                          ec=colours[track_ids[i] % len(colours)]))
                    fig_cam_subplots[cam_id].annotate(f"{track_ids[i]}", (l, b), color='white',
                                                      weight='bold', fontsize=20, ha='center', va='center')
                res_dicts["cam " + str(cam_id)] = res_dict
            
            res_file = os.path.join(res_dir, 'rgb_' + str(batch_idx).zfill(5) + '.json')
            with open(res_file, 'w') as fp:
                json.dump(res_dicts, fp)
                    
            cam_filename = os.path.join(res_dir, str(batch_idx).zfill(5) + '_cam.jpg')
            fig_cam.savefig(cam_filename, bbox_inches='tight')
            plt.close(fig_cam)
    
    print(cnt_ma*100./(cnt_mb+cnt_ma), "% of the matching are done w/ appearance info")
    return


def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    print('Load dataset...')
    test_set = MMPframeDataset(args.dataset_config_file, train=False, world_reduce=args.world_reduce,
                               img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                               img_kernel_size=args.img_kernel_size, sample_freq=args.sample_freq)
    test_set.get_raw = True

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers,
                                              pin_memory=True, worker_init_fn=seed_worker)

    print('Load multi-view detector...')
    model = MVDeTr(test_set, args.arch, world_feat_arch=args.world_feat,
                   bottleneck_dim=128, outfeat_dim=0, droupout=0.0).cuda()
    model.load_state_dict(torch.load(args.mvdetr_modelpath))
    model.eval()

    print('Load single-view detector...')
    detector = YOLO(args.yolo_modelpath).to("cuda")

    print('Set up coordinate mapper...')
    coordmap = CoordMapper(test_set.calibration_path)
    
    print('Load tracker...')
    metric = nn_matching.NearestNeighborDistanceMetric(
        metric='cosine',
        matching_threshold=1, # max_cosine_distance
        budget=100
    )
    tracker = Tracker(metric)
    
    print('Set up output directory...')
    date_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    outdir = os.path.join(args.outdir, date_time)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "settings.json"), "w") as f:
        f.write(json.dumps(vars(args), indent=4))

    print('Start tracking...')
    detect_and_track_cam(model=model,
                         detector=detector,
                         tracker=tracker,
                         data_loader=test_loader,
                         res_dir=outdir,
                         visualize=args.visualize,
                         coord_mapper=coordmap)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multi-camera multi-people tracking using MVDeTr and DeepSORT')
    
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
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    parser.add_argument('--sample_freq', type=int, default=1, help='sample part of frames to save time')
    
    parser.add_argument('--min_count', type=int, default=20)
    parser.add_argument('--cost_ths', type=float, default=0.1)
    parser.add_argument('--nms_ths', type=float, default=20)
    parser.add_argument('--yolo_iou_ths', type=float, default=0.25)
    parser.add_argument('--body_width', type=float, default=150)
    parser.add_argument('--body_height', type=float, default=1600)
    
    parser.add_argument('--dataset_config_file', type=str, default='dataset_config/mmptrack_retail.yaml') #sample
    parser.add_argument('--mvdetr_modelpath', type=str, default='model_weights/MultiviewDetector.pth')
    parser.add_argument('--yolo_modelpath', type=str, default='model_weights/yolo_finetuned.pt')
    parser.add_argument('--reid_modelpath', type=str, default='model_weights/osnet_x0_25_market1501.pt')
    parser.add_argument('--outdir', type=str, default='sample_outputs')
    
    args = parser.parse_args()

    main(args)
