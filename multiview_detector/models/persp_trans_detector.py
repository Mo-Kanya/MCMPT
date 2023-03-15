import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from torchvision.models.vgg import vgg11
from multiview_detector.models.resnet import resnet18

import matplotlib.pyplot as plt


class PerspTransDetector(nn.Module):
    def __init__(self, dataset, arch='resnet18'):
        super().__init__()
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape
        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(dataset.intrinsic_matrices,
                                                                           dataset.extrinsic_matrices,
                                                                           dataset.worldgrid2worldcoord_mat)
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])
        # img
        # self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        self.upsample_shape = list(map(lambda x: int(x/1), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        # map
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        # projection matrices: img feat -> map feat
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                          for cam in range(self.num_cam)]
        # print(self.proj_mats)
        # sdfsfds
        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split].to('cuda:0')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        elif arch == 'resnet18':
            base = nn.Sequential(*list(resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2])
            split = 7
            self.base_pt1 = base[:split].to('cuda:0')
            self.base_pt2 = base[split:].to('cuda:0')
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')
        # 2.5cm -> 0.5m: 20x
        self.img_classifier = nn.Sequential(nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
                                            nn.Conv2d(64, 2, 1, bias=False)).to('cuda:0')
        self.map_classifier = nn.Sequential(nn.Conv2d(out_channel * self.num_cam + 2, 512, 3, padding=1), nn.ReLU(),
                                            # nn.Conv2d(512, 512, 5, 1, 2), nn.ReLU(),
                                            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
                                            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False)).to('cuda:0')
        pass

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        imgs_result = []
        for cam in range(self.num_cam):
            img_feature = self.base_pt1(imgs[:, cam].to('cuda:0'))
            img_feature = self.base_pt2(img_feature.to('cuda:0'))
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature.to('cuda:0'))
            imgs_result.append(img_res)
            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to('cuda:0')
            world_feature = kornia.geometry.transform.imgwarp.warp_perspective(img_feature.to('cuda:0'), proj_mat, self.reducedgrid_shape)  # warp_perspective3d
            # print(world_feature.shape)
            if visualize:
                plt.imshow(torch.norm(img_feature[0].detach(), dim=0).cpu().numpy())
                plt.show()
                plt.imshow(torch.norm(world_feature[0].detach(), dim=0).cpu().numpy())
                plt.show()
            world_features.append(world_feature.to('cuda:0'))

        world_features = torch.cat(world_features + [self.coord_map.repeat([B, 1, 1, 1]).to('cuda:0')], dim=1)
        if visualize:
            plt.imsave("./vis_persp.jpg", torch.norm(world_features[0].detach(), dim=0).cpu().numpy())
            # plt.show()
        map_result = self.map_classifier(world_features.to('cuda:0'))
        map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')

        if visualize:
            plt.imshow(torch.norm(map_result[0].detach(), dim=0).cpu().numpy())
            plt.show()
        return map_result, imgs_result

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices, worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ extrinsic_matrices[cam] #np.delete(, 2, 1)
            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            worldgrid2imgcoord_mat = np.delete(worldgrid2imgcoord_mat, 2, 1)
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            # image of shape C,H,W (C,N_row,N_col); indexed as x,y,w,h (x,y,n_col,n_row)
            # matrix of shape N_row, N_col; indexed as x,y,n_row,n_col
            # TODO: not sure why permutation twice (first time at worldgrid2worldcoord_mat)
            # permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]); permutation_mat @
            projection_matrices[cam] = imgcoord2worldgrid_mat
            pass
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret


def test():
    from multiview_detector.datasets.frameDataset import frameDataset, MMPframeDataset
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    # transform = T.Compose([T.Resize([360, 640]),  # H,W
    #                        T.ToTensor(),
    #                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), transform=transform)
    # dataloader = DataLoader(dataset, 1, False, num_workers=0)
    # imgs, map_gt, imgs_gt, frame = next(iter(dataloader))
    # model = PerspTransDetector(dataset)
    # map_res, img_res = model(imgs, visualize=True)

    dataset = MMPframeDataset("/home/kanya/MVDet/multiview_detector/datasets/configs/retail.yaml")
    dataloader = DataLoader(dataset, 1, False, num_workers=4)

    with torch.no_grad():
        imgs, map_gt, imgs_gt, frame = next(iter(dataloader))
        # imgs, map_gt, imgs_gt, frame = dataset[15]
        model = PerspTransDetector(dataset)
        map_res, img_res = model(imgs, visualize=True)

    # wp = np.array([2511.79245283, 392.26993865,  100., 1.])
    # gp = np.array([336.39237111763816, 189.10957723734649, 1.])
    # res = model.proj_mats[2] @ gp
    # print(res[0]/res[2], res[1]/res[2]) # 250, 200
    # mid = np.array([212.77647348,  50.1102489, 3785.99952084])
    # print(dataset.intrinsic_matrices[2] @ dataset.extrinsic_matrices[2] @ wp ) #


if __name__ == '__main__':
    test()
