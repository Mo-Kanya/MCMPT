import numpy as np
import torch
import sys
import cv2
import gdown
from os.path import exists as file_exists, join
import torchvision.transforms as transforms

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker
from .deep.reid_model_factory import show_downloadeable_models, get_model_url, get_model_name

from torchreid.utils import FeatureExtractor
from torchreid.utils.tools import download_url
from .reid_multibackend import ReIDDetectMultiBackend

__all__ = ['StrongSORT']


class StrongSORT(object):
    def __init__(self, 
                 model_weights,
                 device,
                 fp16,
                 max_dist=0.2,
                 max_iou_distance=0.7,
                 max_age=70, n_init=3,
                 nn_budget=100,
                 mc_lambda=0.995,
                 ema_alpha=0.9
                ):
        
        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)
        
        self.max_dist = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def _point_to_bbox(self, xy, bbox_len): #TODO: replace this with point tracker
        x, y = xy[0].item(), xy[1].item()
        return np.array([x-bbox_len, y-bbox_len, bbox_len*2, bbox_len*2])
    
    def update(self, point_loc_topview, valid_bboxs, bbox_xywh_imgs, confidences, classes, ori_imgs, bbox_len):
        self.height, self.width = ori_imgs[0].shape[:2]
        num_cam, num_bbox = bbox_xywh_imgs.shape[:-1]
        
        # generate detections
        features = []
        for i in range(num_bbox):
            valid_bbox = valid_bboxs[:,i]
            bbox_xywh = bbox_xywh_imgs[:,i,:]
            
            feats = self._get_features(bbox_xywh[valid_bbox == 1], ori_imgs)
            if len(feats) > 0:
                feature = torch.mean(feats, axis=0)
            else:
                feature = torch.tensor([])
            features.append(feature.unsqueeze(0))
        features = torch.cat(features)
        
        # features = []
        # for (valid_bbox, bbox_xywh) in zip(valid_bboxs, bbox_xywh_imgs): #iterate over each camera
        #     valid_xywhs = bbox_xywh[valid_bbox == 1]
            
        #     features_allcam = torch.zeros((6, 512))
        #     for i, (valid_xywh, ori_img) in enumerate(zip(valid_xywhs, ori_imgs)):
        #         feat = self._get_features([valid_xywh], ori_img)
        #         features_allcam[i] = feat[0]
        #     feature = torch.mean(features_allcam, axis=0)
        #     features.append(feature.unsqueeze(0))
        # features = torch.cat(features)
        
        bbox_tlwh = self._point_to_bbox(point_loc_topview, bbox_len)
        detections = [Detection(bbox_tlwh, conf, feat) for (conf, feat) in zip(confidences, features)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, confidences)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            
            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs
    
    def update_original(self, bbox_xywh, confidences, classes, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences)]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, confidences)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            
            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_imgs):
        im_crops = []
        for box, ori_img in zip(bbox_xywh, ori_imgs):
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features
    
    def _get_features_orig(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features

