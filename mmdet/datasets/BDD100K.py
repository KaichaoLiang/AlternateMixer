# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import io
import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
import json
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class BDD100K(CustomDataset):

    CLASSES = ('traffic light','traffic sign','car','pedestrian','bus','truck','rider','bicycle','motorcycle','train')

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30)]


    def load_annotations(self, ann_file):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        data_infos = []
        with open(ann_file,'r') as f:
            annotations = json.load(f)

        for anns in annotations:
            filename = self.img_prefix+'/'+anns['name']
            image = mmcv.imread(filename)
            height, width = image.shape[:2]
    
            data_info = dict(filename=anns['name'], width=width, height=height)
            bbox_names = []
            bboxes = []
            for box in anns['labels']:
                bbox_names.append(box['category'])
                bboxes.append([box['box2d']['x1'],box['box2d']['y1'],box['box2d']['x2'],box['box2d']['y2']])

            gt_bboxes = []
            gt_labels = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
    
            # filter 'DontCare'
            for bbox_name, bbox in zip(bbox_names, bboxes):
                if bbox_name in cat2label:
                    gt_labels.append(cat2label[bbox_name])
                    gt_bboxes.append(bbox)
                else:
                    gt_labels_ignore.append(-1)
                    gt_bboxes_ignore.append(bbox)

            data_anno = dict(
                bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(gt_labels, dtype=np.long),
                bboxes_ignore=np.array(gt_bboxes_ignore,
                                       dtype=np.float32).reshape(-1, 4),
                labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

            data_info.update(ann=data_anno)
            data_infos.append(data_info)

        return data_infos
        
    
         