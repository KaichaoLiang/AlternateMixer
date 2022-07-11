#-------------------------------------------------------------------------
# file: fcos_query_generator_withbox.py
# author: kaichao liang
# date: 2022.06.08
# discription: the query generator of Fcos-Mixer with Fcos proposed box, 
# most adapted from fcos_head.py of mmdetection
#--------------------------------------------------------------------------
import warnings
import math
import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, reduce_mean
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from mmcv.cnn import build_norm_layer
INF = 1e8


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, scale):
        B, C, H, W = x.size()
        not_mask = torch.ones(B, H, W).to(x.device)*scale
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        print('position_embedding shape is', pos.shape)
        return pos


@HEADS.register_module()
class MaxkGenerator(AnchorFreeHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 maxk_num = 100,
                 norm_on_bbox=False,
                 cls_stack_conv=True,
                 reg_stack_conv=True,
                 position_embedding=False,
                 constant_bbox_seed=False,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.maxk_num = maxk_num
        self.norm_on_bbox = norm_on_bbox
        self.constant_bbox_seed = constant_bbox_seed
        self.position_embedding = position_embedding
        
        super().__init__(
            num_classes,
            in_channels,
            cls_stack_conv=cls_stack_conv,
            reg_stack_conv=reg_stack_conv,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        
        if(self.position_embedding):
            self.position_embedding = PositionEmbeddingSine(self.feat_channels, normalize=True)
            self.pyramid_position_embedding = nn.Parameter(torch.ones(len(self.strides), self.feat_channels*2), requires_grad=True)


    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        #self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.feat_projector = nn.Linear(2*self.feat_channels, self.feat_channels)
        self.feat_norm = build_norm_layer(dict(type='LN'),self.feat_channels)[1]
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        self.init_proposal_bboxes = nn.Embedding(self.maxk_num, 4)
    
    def init_weights(self):
        super(MaxkGenerator, self).init_weights()
        nn.init.constant_(self.init_proposal_bboxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_bboxes.weight[:, 2:], 1)

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        '''
        *bbox define in outs:
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        '''
        
        outs = self(x)
        cls_scores = outs[0]
        bbox_preds = outs[1]
        cat_features = outs[2]
        
        if self.norm_on_bbox:
            for s in range(len(self.strides)):
                bbox_preds[s]*=bbox_preds[s]*self.strides[s]
        xyzr, init_content_features, imgs_whwh = self.fcos_feature_proposal(cat_features, img_metas, cls_scores, bbox_preds)

        return {}, xyzr, init_content_features, imgs_whwh
       
    def simple_test_rpn(self, x, img_metas, rescale=False):
        outs = self(x)
        cls_scores = outs[0]
        bbox_preds = outs[1]
        cat_features = outs[2]

        if self.norm_on_bbox:
            for s in range(len(self.strides)):
                bbox_preds[s]*=bbox_preds[s]*self.strides[s]
        xyzr, init_content_features, imgs_whwh = self.fcos_feature_proposal(cat_features, img_metas, cls_scores, bbox_preds)

        return xyzr, init_content_features, imgs_whwh

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        #if self.centerness_on_reg:
            #centerness = self.conv_centerness(reg_feat)
        #else:
            #centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)
        else:
            bbox_pred = bbox_pred.exp()
        out_feat = torch.cat((cls_feat,reg_feat),1)

        if(self.position_embedding):
            B,C,H,W = out_feat.size()
            xy_embedding = self.position_embedding(out_feat, stride)
            fpn_level = self.strides.index(stride)
            z_embedding = self.pyramid_position_embedding[fpn_level].clone().view(1,-1,1,1).repeat(B,1,H,W)
            print('z embedding shape is ', z_embedding.size())
            pos_embedding = xy_embedding + z_embedding
            out_feat = out_feat+pos_embedding
        return cls_score, bbox_pred, out_feat

    def bbox_cxcywh_to_xyxy(self,bbox):
        """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

        Args:
            bbox (Tensor): Shape (n, 4) for bboxes.

        Returns:
        T   ensor: Converted bboxes.
        """
        cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
        bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
        return torch.cat(bbox_new, dim=-1)
    
    def fcos_feature_proposal(self, x, img_metas, cls_scores, bbox_preds):
        num_imgs = cls_scores[0].size(0)
        batch_w = img_metas[0]['batch_shape'][0]
        batch_h = img_metas[0]['batch_shape'][1]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        
        flatten_features = [
            feature.permute(0, 2, 3, 1).reshape(num_imgs, -1, x[0].size(1))
            for feature in x
        ]
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_features = torch.cat(flatten_features,dim=1)
        flatten_cls_scores = torch.cat(flatten_cls_scores,dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds,dim=1)

        flatten_points = torch.cat(
            [points.view(1,-1,2).repeat(num_imgs,1, 1) for points in all_level_points],dim=1)
        sigm = torch.nn.Sigmoid()
        flatten_cls = sigm(flatten_cls_scores.max(-1)[0])
        flatten_totalscore = flatten_cls
        
        # max_k selection, image wise
        sort_score, sort_id = torch.sort(flatten_totalscore,dim=-1,descending=True) 
        select_features = []
        select_points = []
        select_bboxes = []
        for id in range(num_imgs):
            ind_select = sort_id[id,:self.maxk_num]
            flatten_feature = flatten_features[id,:,:]
            flatten_point = flatten_points[id,:,:]
            flatten_bbox_pred = flatten_bbox_preds[id,:,:]
            select_point = flatten_point[ind_select].clone().view(1,self.maxk_num,2)
            select_bbox = flatten_bbox_pred[ind_select].clone().view(1,self.maxk_num,4)
            #select_point = select_point.flip(2) #point height/width
            #normalize to image size
            select_feature = flatten_feature[ind_select].clone().view(1,self.maxk_num,x[0].size(1))

            select_points.append(select_point)
            select_features.append(select_feature)
            select_bboxes.append(select_bbox)
        
        select_points = torch.cat(select_points,dim=0)
        select_features = torch.cat(select_features,dim=0)
        select_bboxes = torch.cat(select_bboxes,dim=0)
        
        if(self.constant_bbox_seed):
            xyzr, imgs_whwh = self._decode_init_proposals(x, img_metas)
        else:
            xs = select_points[:,:,0]
            ys = select_points[:,:,1]
        
            x1 = xs - select_bboxes[...,0]
            x1[x1<0]=0
            x1[x1>batch_w-2]=batch_w-2
            y1 = ys - select_bboxes[...,1]
            y1[y1<0]=0
            y1[y1>batch_h-2]=batch_h-2
            x2 = xs + select_bboxes[...,2]
            x2 = torch.max(x1+1,x2)
            x2[x2>batch_w-1]=batch_w-1
            y2 = ys + select_bboxes[...,3]
            y2 = torch.max(y1+1,y2)
            y2[y2>batch_h-1]=batch_h-1
            proposals = torch.stack((x1, y1, x2, y2), -1)

            imgs_whwh = []
            for meta in img_metas:
                h, w, _ = meta['img_shape']
                imgs_whwh.append(x[0].new_tensor([[w, h, w, h]]))
            imgs_whwh = torch.cat(imgs_whwh, dim=0)
            imgs_whwh = imgs_whwh[:, None, :]

            xy = 0.5 * (proposals[..., 0:2] + proposals[..., 2:4])
            wh = proposals[..., 2:4] - proposals[..., 0:2]
            z = (wh).prod(-1, keepdim=True).sqrt().log2()
            r = (wh[..., 1:2]/wh[..., 0:1]).log2()

            # NOTE: xyzr **not** learnable
            xyzr = torch.cat([xy, z, r], dim=-1).detach()

        select_features = self.feat_projector(select_features)
        select_features = self.feat_norm(select_features)
        return xyzr, select_features, imgs_whwh
    
    def _decode_init_proposals(self, imgs, img_metas):
        """
        Hacks based on 'sparse_roi_head.py'.
        For the positional vector, we first compute (x, y, z, r) that fully covers an image. 
        """
        proposals = self.init_proposal_bboxes.weight.clone()
        proposals = self.bbox_cxcywh_to_xyxy(proposals)
        num_imgs = len(imgs[0])
        imgs_whwh = []
        for meta in img_metas:
            h, w, _ = meta['img_shape']
            imgs_whwh.append(imgs[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        proposals = proposals * imgs_whwh

        xy = 0.5 * (proposals[..., 0:2] + proposals[..., 2:4])
        wh = proposals[..., 2:4] - proposals[..., 0:2]
        z = (wh).prod(-1, keepdim=True).sqrt().log2()
        r = (wh[..., 1:2]/wh[..., 0:1]).log2()

        # NOTE: xyzr **not** learnable
        xyzr = torch.cat([xy, z, r], dim=-1).detach()
        return xyzr, imgs_whwh

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        return None
    
    def loss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas, gt_bboxes_ignore=None):
        return None
