import os
import yaml
import re
import json
import math
import csv

import cv2
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision.datasets import VisionDataset
from scipy.stats import multivariate_normal
from scipy.sparse import coo_matrix
from torchvision.transforms import ToTensor

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
        self.img_shape, self.worldgrid_shape = [320, 640], [160, 250] #TODO: check worldgrid_shape
        self.bins = []
        self.num_cam = config["num_views"]
        self.grid_reduce, self.img_reduce = grid_reduce, img_reduce
        self.indexing = 'xy'  # might be different with MVDet in persp proj
        self.gt_fpath = None  # TODO: parse labels to write a gt.txt
        self.world_reduce = self.grid_reduce
        
        self.Rworld_shape = list(map(lambda x: x // self.grid_reduce, self.worldgrid_shape))
        self.Rimg_shape = np.ceil(np.array(self.img_shape) / self.img_reduce).astype(int).tolist()

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

        