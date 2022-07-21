import torch
import torch.nn as nn
from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh
from mmdet.core.bbox.samplers import PseudoSampler
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from ..builder import HEADS
from .cascade_roi_head import CascadeRoIHead
import os

DEBUG = 'DEBUG' in os.environ


@HEADS.register_module()
class AdaMixerDecoderQueryBank(CascadeRoIHead):
    _DEBUG = -1

    def __init__(self,
                 num_stages=6,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 content_dim=256,
                 featmap_strides=[4, 8, 16, 32],
                 bank_size=300,
                 bbox_head=None, 
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        self.featmap_strides = featmap_strides
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.content_dim = content_dim
        self.bank_size = bank_size
        super(AdaMixerDecoderQueryBank, self).__init__(
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
        # train_cfg would be None when run the test.py
        if train_cfg is not None:
            for stage in range(num_stages):
                assert isinstance(self.bbox_sampler[stage], PseudoSampler)
        self._init_layers()
    
    def _init_layers(self):
        self.init_content_features = nn.Embedding(
            self.bank_size, self.content_dim)
        self.crossattention = MultiheadAttention(self.content_dim, 8, 0.0, batch_first=True)
        self.attention_norm = build_norm_layer(dict(type='LN'), self.content_dim)[1]
    
    def _bbox_forward(self, stage, img_feat, query_xyzr, query_content, img_metas):
        num_imgs = len(img_metas)
        init_content_features = self.init_content_features.weight.clone()
        init_content_features = init_content_features[None].expand(
            num_imgs, *init_content_features.size())
        query_crossatt = self.crossattention(query=query_content, key=init_content_features, value=init_content_features)
        query_content = query_content + query_crossatt
        query_content = self.attention_norm(query_content)

        
        bbox_head = self.bbox_head[stage]

        cls_score, delta_xyzr, query_content = bbox_head(img_feat, query_xyzr,
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
        )
        if DEBUG:
            with torch.no_grad():
                torch.save(
                    bbox_results, 'demo/bbox_results_{}.pth'.format(AdaMixerDecoderQueryBank._DEBUG))
                AdaMixerDecoderQueryBank._DEBUG += 1
        return bbox_results

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
