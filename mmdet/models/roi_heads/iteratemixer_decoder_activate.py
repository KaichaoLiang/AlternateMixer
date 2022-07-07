#-------------------------------------------------------------------------
# file: iteratemixer_decoder.py
# author: kaichao liang
# date: 2022.06.16
# discription: alternative iteration for feature paramid and queries.
# most adapted from adamixer_decoder.py of mmdetection
#--------------------------------------------------------------------------

import queue
from pandas import concat
import torch
import torch.nn.functional as F
from zmq import device
F.conv2d
from torch.nn import Linear
import torch.nn as nn
from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.samplers import PseudoSampler
from ..builder import HEADS
from .cascade_roi_head import CascadeRoIHead
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
import os
import time
DEBUG = 'DEBUG' in os.environ


@HEADS.register_module()
class IterateMixerDecoderActivate(CascadeRoIHead):
    _DEBUG = -1

    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 content_dim=256,
                 featmap_strides=[4, 8, 16, 32],
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 query_detach=False,
                 feat_norm='GN',
                 init_cfg=None):
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.featmap_strides = featmap_strides
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.content_dim = content_dim
        self.query_detach = query_detach
        assert feat_norm in ['GN','BN2d'], "not supported normalization"
        self.feat_norm = feat_norm
        super(IterateMixerDecoderActivate, self).__init__(
            num_stages,
            stage_loss_weights,
            bbox_roi_extractor=dict(
                # This does not mean that our method need RoIAlign. We put this
                # as a placeholder to satisfy the argument for the parent class
                # 'CascadeRoIHead'.
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self._init_layers()
        #self.init_weights()
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler)
    
    def _init_layers(self):
        """Initialize layers of the head."""
        self.query_projection_stages = nn.ModuleList()

        self.conv_generate_stages = nn.ModuleList()
        self.conv_norm_stages = nn.ModuleList()
        self.conv_activation_stages = nn.ModuleList()

        self.mixing_generate_stages = nn.ModuleList()
        self.mixing_norm_stages = nn.ModuleList()
        self.mixing_activation_stages = nn.ModuleList()
        SCALE = len(self.featmap_strides)
        for stage in range(self.num_stages):

            for s in range(SCALE):
                self.query_projection_stages.append(Linear(self.content_dim, self.content_dim))
                
                self.conv_generate_stages.append(Linear(self.content_dim, 3*3*self.content_dim))
                if self.feat_norm=='BN2d':
                    self.conv_norm_stages.append(build_norm_layer(dict(type=self.feat_norm), self.content_dim)[1]) 
                elif self.feat_norm=='GN':
                    self.conv_norm_stages.append(build_norm_layer(dict(type=self.feat_norm,num_groups=8),self.content_dim)[1]) 
                self.conv_activation_stages.append(build_activation_layer(dict(type='ReLU', inplace=True)))

                self.mixing_generate_stages.append(Linear(self.content_dim, 1*1*self.content_dim*self.content_dim))
                if self.feat_norm=='BN2d':
                    self.mixing_norm_stages.append(build_norm_layer(dict(type=self.feat_norm), self.content_dim)[1]) 
                elif self.feat_norm=='GN':
                    self.mixing_norm_stages.append(build_norm_layer(dict(type=self.feat_norm, num_groups=8),self.content_dim)[1])  
                self.mixing_activation_stages.append(build_activation_layer(dict(type='ReLU', inplace=True)))
            
    def init_weights(self):
        super().init_weights()
        SCALE = len(self.featmap_strides)
        for stage in range(self.num_stages):
            for s in range(SCALE):
                module = self.conv_generate_stages[stage*SCALE+s]
                module.reset_parameters()
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                #module.cuda()

                module = self.mixing_generate_stages[stage*SCALE+s]
                module.reset_parameters()
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                #module.cuda()

    def _bbox_forward(self, stage, img_feat, query_xyzr, query_content, img_metas):
        num_imgs = len(img_metas)
        bbox_head = self.bbox_head[stage]
        img_feat_update = self._update_img_feat(stage=stage,img_feat=img_feat,query_content=query_content)
        cls_score, delta_xyzr, query_content = bbox_head(img_feat_update, query_xyzr,
                                                         query_content,
                                                         featmap_strides=self.featmap_strides)
        query_xyzr, decoded_bboxes = self.bbox_head[stage].refine_xyzr(
            query_xyzr,
            delta_xyzr)
        bboxes_list = [bboxes for bboxes in decoded_bboxes]
        bbox_results = dict(
            cls_score=cls_score,
            query_xyzr=query_xyzr,
            decode_bbox_pred=decoded_bboxes,
            query_content=query_content,
            detach_cls_score_list=[
                cls_score[i].detach() for i in range(num_imgs)
            ],
            detach_bboxes_list=[item.detach() for item in bboxes_list],
            bboxes_list=bboxes_list,
            img_feat_update=img_feat_update
        )
        if DEBUG:
            with torch.no_grad():
                torch.save(
                    bbox_results, 'demo/bbox_results_{}.pth'.format(IterateMixerDecoderActivate._DEBUG))
                IterateMixerDecoderActivate._DEBUG += 1
        return bbox_results
    
    def _update_img_feat(self, stage, img_feat, query_content):
        img_feat_out = []
        batchsize = img_feat[0].size(0)
        num_query = query_content.size(1)
        SCALE=len(img_feat)
        for s in range(SCALE):
            img_in = img_feat[s]
            img_batch = img_in
            h=img_batch.size(2)
            w=img_batch.size(3)
            query_seed = query_content
            if self.query_detach:
                query_seed = query_seed.detach()
            
            query_activate = self.query_projection_stages[stage*SCALE+s](query_seed)
            query_activate = F.sigmoid(query_activate)
            query_activate = query_activate.view(query_seed.size())
            query_seed = query_seed * query_activate
            query_seed = torch.sum(query_seed, dim=1)

            conv_kernel = self.conv_generate_stages[stage*SCALE+s](query_seed.view(batchsize,self.content_dim))
            conv_kernel = conv_kernel.view(batchsize*self.content_dim,1,3,3)
            img_batch = img_batch.view(1, batchsize*self.content_dim, h, w)
            img_batch = F.conv2d(img_batch,conv_kernel,stride=1,padding=1, groups=batchsize*self.content_dim)
            img_batch = img_batch.view(batchsize, self.content_dim, h,w)
            img_batch = self.conv_norm_stages[stage*SCALE+s](img_batch)
            img_batch = self.conv_activation_stages[stage*SCALE+s](img_batch)

            mixing_kernel = self.mixing_generate_stages[stage*SCALE+s](query_seed.view(batchsize,self.content_dim))
            mixing_kernel = mixing_kernel.view(batchsize*self.content_dim,self.content_dim,1,1)
            img_batch = img_batch.view(1, batchsize*self.content_dim, h, w)
            img_batch = F.conv2d(img_batch,mixing_kernel,stride=1,padding=0, groups=batchsize)
            img_batch = img_batch.view(batchsize, self.content_dim, h,w)
            img_batch = self.mixing_norm_stages[stage*SCALE+s](img_batch)
            img_batch = self.mixing_activation_stages[stage*SCALE+s](img_batch)
            img_batch=img_batch+img_in
            img_feat_out.append(img_batch)
        return img_feat_out

    def forward_train(self,
                      x,
                      query_xyzr,
                      query_content,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_masks=None):

        num_imgs = len(img_metas)
        num_queries = query_xyzr.size(1)
        imgs_whwh = imgs_whwh.repeat(1, num_queries, 1)
        all_stage_bbox_results = []
        all_stage_loss = {}

        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(stage, x, query_xyzr, query_content,
                                              img_metas)
            all_stage_bbox_results.append(bbox_results)
            if gt_bboxes_ignore is None:
                # TODO support ignore
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            cls_pred_list = bbox_results['detach_cls_score_list']
            bboxes_list = bbox_results['detach_bboxes_list']

            query_xyzr = bbox_results['query_xyzr'].detach()
            query_content = bbox_results['query_content']
            x = bbox_results['img_feat_update']

            if self.stage_loss_weights[stage] <= 0:
                continue

            for i in range(num_imgs):
                normalize_bbox_ccwh = bbox_xyxy_to_cxcywh(bboxes_list[i] /
                                                          imgs_whwh[i])
                assign_result = self.bbox_assigner[stage].assign(
                    normalize_bbox_ccwh, cls_pred_list[i], gt_bboxes[i],
                    gt_labels[i], img_metas[i])
                sampling_result = self.bbox_sampler[stage].sample(
                    assign_result, bboxes_list[i], gt_bboxes[i])
                sampling_results.append(sampling_result)
            bbox_targets = self.bbox_head[stage].get_targets(
                sampling_results, gt_bboxes, gt_labels, self.train_cfg[stage],
                True)

            cls_score = bbox_results['cls_score']
            decode_bbox_pred = bbox_results['decode_bbox_pred']

            single_stage_loss = self.bbox_head[stage].loss(
                cls_score.view(-1, cls_score.size(-1)),
                decode_bbox_pred.view(-1, 4),
                *bbox_targets,
                imgs_whwh=imgs_whwh)
            for key, value in single_stage_loss.items():
                all_stage_loss[f'stage{stage}_{key}'] = value * \
                    self.stage_loss_weights[stage]

        return all_stage_loss

    def simple_test(self,
                    x,
                    query_xyzr,
                    query_content,
                    img_metas,
                    imgs_whwh,
                    rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        if DEBUG:
            torch.save(img_metas, 'demo/img_metas.pth')

        num_imgs = len(img_metas)

        for stage in range(self.num_stages):
            bbox_results = self._bbox_forward(
                stage, x, query_xyzr, query_content, img_metas)
            query_content = bbox_results['query_content']
            cls_score = bbox_results['cls_score']
            bboxes_list = bbox_results['detach_bboxes_list']
            query_xyzr = bbox_results['query_xyzr']
            x = bbox_results['img_feat_update']

        num_classes = self.bbox_head[-1].num_classes
        det_bboxes = []
        det_labels = []

        if self.bbox_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        for img_id in range(num_imgs):
            cls_score_per_img = cls_score[img_id]
            scores_per_img, topk_indices = cls_score_per_img.flatten(
                0, 1).topk(
                    self.test_cfg.max_per_img, sorted=False)
            labels_per_img = topk_indices % num_classes
            bbox_pred_per_img = bboxes_list[img_id][topk_indices //
                                                    num_classes]
            if rescale:
                scale_factor = img_metas[img_id]['scale_factor']
                bbox_pred_per_img /= bbox_pred_per_img.new_tensor(scale_factor)
            det_bboxes.append(
                torch.cat([bbox_pred_per_img, scores_per_img[:, None]], dim=1))
            det_labels.append(labels_per_img)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], num_classes)
            for i in range(num_imgs)
        ]

        return bbox_results

    def aug_test(self, x, bboxes_list, img_metas, rescale=False):
        raise NotImplementedError()

    def forward_dummy(self, x,
                      query_xyzr,
                      query_content,
                      img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_bbox_results = []

        num_imgs = len(img_metas)
        if self.with_bbox:
            for stage in range(self.num_stages):
                bbox_results = self._bbox_forward(stage, x,
                                                  query_xyzr,
                                                  query_content,
                                                  img_metas)
                all_stage_bbox_results.append(bbox_results)
                query_content = bbox_results['query_content']
                query_xyzr = bbox_results['query_xyzr']
        return all_stage_bbox_results
