import os
import json

import numpy as np
import yaml
import csv
from scipy.stats import multivariate_normal
from PIL import Image
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from torchvision.transforms import ToTensor
from multiview_detector.utils.projection import *
import math


class frameDataset(VisionDataset):
    def __init__(self, base, train=True, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=4, img_reduce=4, train_ratio=0.9, force_download=True):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        img_sigma, img_kernel_size = 10 / img_reduce, 10
        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce

        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))

        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.map_gt = {}
        self.imgs_head_foot_gt = {}
        self.download(frame_range)

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

        x, y = np.meshgrid(np.arange(-img_kernel_size, img_kernel_size + 1),
                           np.arange(-img_kernel_size, img_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        img_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * img_sigma)
        img_kernel = img_kernel / img_kernel.max()
        kernel_size = img_kernel.shape[0]
        self.img_kernel = torch.zeros([2, 2, kernel_size, kernel_size], requires_grad=False)
        self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        self.img_kernel[1, 1] = torch.from_numpy(img_kernel)
        pass

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, frame_range):
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                i_s, j_s, v_s = [], [], []
                head_row_cam_s, head_col_cam_s = [[] for _ in range(self.num_cam)], \
                                                 [[] for _ in range(self.num_cam)]
                foot_row_cam_s, foot_col_cam_s, v_cam_s = [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)]
                for single_pedestrian in all_pedestrians:
                    x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
                    for cam in range(self.num_cam):
                        x = max(min(int((single_pedestrian['views'][cam]['xmin'] +
                                         single_pedestrian['views'][cam]['xmax']) / 2), self.img_shape[1] - 1), 0)
                        y_head = max(single_pedestrian['views'][cam]['ymin'], 0)
                        y_foot = min(single_pedestrian['views'][cam]['ymax'], self.img_shape[0] - 1)
                        if x > 0 and y > 0:
                            head_row_cam_s[cam].append(y_head)
                            head_col_cam_s[cam].append(x)
                            foot_row_cam_s[cam].append(y_foot)
                            foot_col_cam_s[cam].append(x)
                            v_cam_s[cam].append(single_pedestrian['personID'] + 1 if self.reID else 1)
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape)
                self.map_gt[frame] = occupancy_map
                self.imgs_head_foot_gt[frame] = {}
                for cam in range(self.num_cam):
                    img_gt_head = coo_matrix((v_cam_s[cam], (head_row_cam_s[cam], head_col_cam_s[cam])),
                                             shape=self.img_shape)
                    img_gt_foot = coo_matrix((v_cam_s[cam], (foot_row_cam_s[cam], foot_col_cam_s[cam])),
                                             shape=self.img_shape)
                    self.imgs_head_foot_gt[frame][cam] = [img_gt_head, img_gt_foot]

    def __getitem__(self, index):
        frame = list(self.map_gt.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        map_gt = self.map_gt[frame].toarray()
        if self.reID:
            map_gt = (map_gt > 0).int()
        if self.target_transform is not None:
            map_gt = self.target_transform(map_gt)
        imgs_gt = []
        for cam in range(self.num_cam):
            img_gt_head = self.imgs_head_foot_gt[frame][cam][0].toarray()
            img_gt_foot = self.imgs_head_foot_gt[frame][cam][1].toarray()
            img_gt = np.stack([img_gt_head, img_gt_foot], axis=2)
            if self.reID:
                img_gt = (img_gt > 0).int()
            if self.target_transform is not None:
                img_gt = self.target_transform(img_gt)
            imgs_gt.append(img_gt.float())
        return imgs, map_gt.float(), imgs_gt, frame

    def __len__(self):
        return len(self.map_gt.keys())


class MMPframeDataset(VisionDataset):
    # TODO: MMP consists of several videos. For tracking, we cannot simply combine them together.
    def __init__(self, config_file, train=True, sample_freq=1, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=2, img_reduce=1, train_ratio=0.9, force_download=False):
        config = yaml.safe_load(open(config_file, 'r'))
        root = config["train_root"] if train else config["valid_root"]
        super().__init__(root, transform=transform, target_transform=target_transform)
        env = config["env_name"]
        self.calibration_path = os.path.join(root, f"calibrations/{env}/calibrations.json")
        self.image_path = os.path.join(root, "images/")
        self.label_path = os.path.join(root, "labels/")
        self.topdown_label_path = os.path.join(root, "topdown_labels/")
        self.img_shape = [320, 640]
        self.bins = []
        self.num_cam = config["num_views"]
        self.grid_reduce, self.img_reduce = grid_reduce, img_reduce
        self.indexing = 'xy'  # might be different with MVDet in persp proj
        self.gt_fpath = None  # TODO: parse labels to write a gt.txt

        # TODO: tune this Gaussian kernel size
        # map_sigma, map_kernel_size = 20 / grid_reduce, 20
        # img_sigma, img_kernel_size = 10 / img_reduce, 10
        map_sigma, map_kernel_size = 16 / grid_reduce, 16
        img_sigma, img_kernel_size = 8 / img_reduce, 8

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

        x, y = np.meshgrid(np.arange(-img_kernel_size, img_kernel_size + 1),
                           np.arange(-img_kernel_size, img_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        img_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * img_sigma)
        img_kernel = img_kernel / img_kernel.max()
        kernel_size = img_kernel.shape[0]
        self.img_kernel = torch.zeros([2, 2, kernel_size, kernel_size], requires_grad=False)
        self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        self.img_kernel[1, 1] = torch.from_numpy(img_kernel)

        total_frames = 0
        sample_list = config["train_list"] if train else config["valid_list"]
        for k in sample_list:
            total_frames += sample_list[k]//6
            self.bins.append( (k, total_frames) )
        self.total_frames = total_frames//sample_freq
        self.sample_freq = sample_freq

        self.grid_size = config["grid_size"]
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.grid_size))
        # self.reducedgrid_shape = self.grid_size  # assuming no reduction
        camera_configs = json.load(open(os.path.join(self.calibration_path)))
        self.num_cam = len(camera_configs['Cameras'])
        self.intrinsic_matrices = [np.zeros((3, 3)) for i in range(self.num_cam)]
        self.extrinsic_matrices = [np.zeros((3, 4)) for i in range(self.num_cam)]
        for camera_param in camera_configs['Cameras']:
            cam_id = camera_param['CameraId']-1
            rot_m = np.linalg.inv(np.asarray(camera_param['ExtrinsicParameters']['Rotation']).reshape((3, 3)))
            self.extrinsic_matrices[cam_id][:, :3] = rot_m
            self.extrinsic_matrices[cam_id][:, -1] = -rot_m @ np.asarray(camera_param['ExtrinsicParameters']['Translation'])
            self.intrinsic_matrices[cam_id][0, 0] = camera_param['IntrinsicParameters']['Fx']
            self.intrinsic_matrices[cam_id][1, 1] = camera_param['IntrinsicParameters']['Fy']
            self.intrinsic_matrices[cam_id][0, 2] = camera_param['IntrinsicParameters']['Cx']
            self.intrinsic_matrices[cam_id][1, 2] = camera_param['IntrinsicParameters']['Cy']
            self.intrinsic_matrices[cam_id][2, 2] = 1

        factorX = 1.0 / (
                (camera_configs['Space']['MaxU'] - camera_configs['Space']['MinU']) / (math.floor(
            (camera_configs['Space']['MaxU'] - camera_configs['Space']['MinU']) /
            camera_configs['Space']['VoxelSizeInMM']) - 1))
        factorY = 1.0 / (
                (camera_configs['Space']['MaxV'] - camera_configs['Space']['MinV']) / (math.floor(
            (camera_configs['Space']['MaxV'] - camera_configs['Space']['MinV']) /
            camera_configs['Space']['VoxelSizeInMM']) - 1))
        self.worldgrid2worldcoord_mat = np.array([[1/factorX, 0, 0, camera_configs['Space']['MinU']],
                                                 [0, 1/factorY, 0, camera_configs['Space']['MinV']],
                                                 [0, 0, 1, camera_configs['Space']['MinW']],
                                                 [0, 0, 0, 1]])


    def __len__(self):
        return self.total_frames

    def __getitem__(self, index):
        index *= self.sample_freq

        new_index = None
        img_dir = None
        for i, (bin_path, num) in enumerate(self.bins):
            if num > index:
                new_index = index if i == 0 else index-self.bins[i-1][1]
                img_dir = os.path.join(bin_path)
                break

        imgs = []
        imgs_gt = []
        for cam in range(1, self.num_cam+1):
            fpath = os.path.join(self.image_path, img_dir, 'rgb_' + str(new_index).zfill(5) + '_' + str(cam) + '.jpg')
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)

            label = json.load(
                open(os.path.join(self.label_path, img_dir, 'rgb_' + str(new_index).zfill(5) + '_' + str(cam) + '.json'))
            )
            v, hr, fr, c = [], [], [], []
            # print(label)
            for pid in label:
                bbox = label[pid]
                if bbox[0]>self.img_shape[1]-1 or bbox[1]>self.img_shape[0]-1 or bbox[2]<0 or bbox[3]<0:
                    continue
                x = max(min( int((bbox[0] + bbox[2]) / 2), self.img_shape[1] - 1), 0)
                y_head = max(bbox[1], 0)
                y_foot = min(bbox[3], self.img_shape[0] - 1)
                v.append(1)
                hr.append(y_head)
                fr.append(y_foot)
                c.append(x)
            # print(hr)
            # print(c)
            img_gt_head = coo_matrix((v, (hr, c)), shape=self.img_shape).toarray()
            img_gt_foot = coo_matrix((v, (fr, c)), shape=self.img_shape).toarray()
            img_gt = np.stack([img_gt_head, img_gt_foot], axis=2)
            if self.target_transform is not None:
                img_gt = self.target_transform(img_gt)
            imgs_gt.append(img_gt.float())
        imgs = torch.stack(imgs)

        map_gt = None
        with open(os.path.join(self.topdown_label_path, img_dir, 'topdown_' + str(new_index).zfill(5) + '.csv'), 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            v, i_s, j_s = [], [], []
            for row in csv_reader:
                v.append(1)
                i_s.append( min(int(float(row[1]) / self.grid_reduce), self.reducedgrid_shape[0])-1 )
                j_s.append( min(int(float(row[2]) / self.grid_reduce), self.reducedgrid_shape[1])-1 )
            map_gt = coo_matrix((v, (i_s, j_s)), shape=self.reducedgrid_shape).toarray()
            if self.target_transform is not None:
                map_gt = self.target_transform(map_gt)
        return imgs, map_gt.float(), imgs_gt, new_index


def test():
    from multiview_detector.datasets.MultiviewX import MultiviewX
    # from multiview_detector.datasets.MultiviewX import MultiviewX
    from multiview_detector.utils.projection import get_worldcoord_from_imagecoord
    dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')))
    # test projection
    # world_grid_maps = []
    # xx, yy = np.meshgrid(np.arange(0, 1920, 20), np.arange(0, 1080, 20))
    # H, W = xx.shape
    # image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])
    # import matplotlib.pyplot as plt
    # for cam in range(dataset.num_cam):
    #     world_coords = get_worldcoord_from_imagecoord(image_coords.transpose(), dataset.base.intrinsic_matrices[cam],
    #                                                   dataset.base.extrinsic_matrices[cam])
    #     world_grids = dataset.base.get_worldgrid_from_worldcoord(world_coords).transpose().reshape([H, W, 2])
    #     world_grid_map = np.zeros(dataset.worldgrid_shape)
    #     for i in range(H):
    #         for j in range(W):
    #             x, y = world_grids[i, j]
    #             if dataset.base.indexing == 'xy':
    #                 if x in range(dataset.worldgrid_shape[1]) and y in range(dataset.worldgrid_shape[0]):
    #                     world_grid_map[int(y), int(x)] += 1
    #             else:
    #                 if x in range(dataset.worldgrid_shape[0]) and y in range(dataset.worldgrid_shape[1]):
    #                     world_grid_map[int(x), int(y)] += 1
    #     world_grid_map = world_grid_map != 0
    #     plt.imshow(world_grid_map)
    #     plt.show()
    #     world_grid_maps.append(world_grid_map)
    #     pass
    # plt.imshow(np.sum(np.stack(world_grid_maps), axis=0))
    # plt.show()
    # pass
    imgs, map_gt, imgs_gt, _ = dataset.__getitem__(0) # (6, 3, 1080, 1920); (1, 160, 250); [(2, 1080, 1920)]*6
    print(torch.sum(map_gt), torch.max(map_gt), torch.min(map_gt))
    print(torch.max(imgs_gt[0][0]), torch.min(imgs_gt[0][0]))
    print(torch.max(imgs_gt[0][1]), torch.min(imgs_gt[0][1]))

    pass


if __name__ == '__main__':
    # test()
    dataset = MMPframeDataset("/home/kanya/MVDet/multiview_detector/datasets/configs/retail.yaml", train=False)
    # print(len(dataset))
    imgs, map_gt, imgs_gt, _ = dataset.__getitem__(5000)
    print(_)
    print(imgs.shape)
    print(map_gt.shape)
    print(len(imgs_gt))
    print(imgs_gt[0].shape)
    print(torch.sum(map_gt), torch.max(map_gt), torch.min(map_gt))
    print(torch.max(imgs_gt[0][0]), torch.min(imgs_gt[0][0]))
    print(torch.max(imgs_gt[0][1]), torch.min(imgs_gt[0][1]))
    print(dataset.reducedgrid_shape)
