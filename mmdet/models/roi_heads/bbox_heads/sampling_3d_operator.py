import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F


def sampling_each_level(sample_points: torch.Tensor,
                        value: torch.Tensor,
                        weight=None,
                        n_points=1):
    B1, n_queries, _t, n_groups_points, _ = sample_points.shape
    assert _t == 1
    B2, C_feat, H_feat, W_feat = value.shape
    assert B1 == B2
    B = B1

    n_groups = n_groups_points//n_points
    n_channels = C_feat//n_groups

    sample_points = sample_points \
        .view(B, n_queries, n_groups, n_points, 2) \
        .permute(0, 2, 1, 3, 4).flatten(0, 1)
    sample_points = sample_points*2.0-1.0

    # `sampling_points` now has the shape [B*n_groups, n_queries, n_points, 2]

    value = value.view(B*n_groups, n_channels, H_feat, W_feat)
    out = F.grid_sample(
        value, sample_points,
        mode='bilinear', padding_mode='zeros', align_corners=False,
    )

    # `out`` now has the shape [B*n_groups, C, n_queries, n_points]

    if weight is not None:
        weight = weight.view(B, n_queries, n_groups, n_points) \
            .permute(0, 2, 1, 3).flatten(0, 1).unsqueeze(1)
        # `weight`` has the shape [B*n_groups, 1, n_queries, n_points]
        out *= weight

    return out \
        .view(B, n_groups, n_channels, n_queries, n_points) \
        .permute(0, 3, 1, 4, 2)

    # `out`` has shape [B, n_queries, n_groups, n_points, n_channels]

def sampling_each_level_alternate(sample_points: torch.Tensor,
                        value: torch.Tensor,
                        query: torch.Tensor,
                        weight=None,
                        n_points=1):
    B1, n_queries, _t, n_groups_points, _ = sample_points.shape
    assert _t == 1
    B2, C_feat, H_feat, W_feat = value.shape
    assert B1 == B2
    B = B1

    n_groups = n_groups_points//n_points
    n_channels = C_feat//n_groups

    sample_points = sample_points \
        .view(B, n_queries, n_groups, n_points, 2) \
        .permute(0, 2, 1, 3, 4).flatten(0, 1)
    sample_points = sample_points*2.0-1.0

    # `sampling_points` now has the shape [B*n_groups, n_queries, n_points, 2]

    value = value.view(B*n_groups, n_channels, H_feat, W_feat)
    out = F.grid_sample(
        value, sample_points,
        mode='bilinear', padding_mode='zeros', align_corners=False,
    )

    # `out`` now has the shape [B*n_groups, C, n_queries, n_points]

    if weight is not None:
        weight = weight.view(B, n_queries, n_groups, n_points) \
            .permute(0, 2, 1, 3).flatten(0, 1).unsqueeze(1)
        # `weight`` has the shape [B*n_groups, 1, n_queries, n_points]
        out *= weight

    out = out \
        .view(B, n_groups, n_channels, n_queries, n_points) \
        .permute(0, 3, 1, 4, 2)
    
    # inverse sampling modified by kaichao liang
    #sample_points back to position index level. 
    sample_points = (sample_points+1.0)/2.0
    sample_points = sample_points.view(B,n_groups,n_queries,n_points,2)
    sample_points = sample_points.permute(0,2,1,3,4)
    print('mark sample shape',sample_points.size(),B,n_queries,n_groups,n_points,sample_points.is_contiguous())
    sample_points = sample_points.contiguous()
    print('mark sample shape',sample_points.size(),B,n_queries,n_groups,n_points,sample_points.is_contiguous())
    sample_points = sample_points.view(4,100,128,2)
    
    weight = weight.view(B, n_groups, n_queries, n_points).permute(0,2,1,3).view(B, n_queries,-1)
    sample_points = sample_points[:,:,:,0] * (H_feat)
    sample_points = sample_points[:,:,:,1] * (W_feat)
    inverse_feats = inverse_sample(sample_points, query, weight, H_feat, W_feat)

    return out, inverse_feats

# The inverse sampling sparse feature by kaichao liang
def inverse_sample(sample_points, query, weight, H_feat, W_feat):
    B, Nq, Np, _ =sample_points.size()
    sample_points=sample_points.int()
    sample_points[:,:,:,0] = torch.clip(sample_points[:,:,:,0], 0, H_feat-1)
    sample_points[:,:,:,1] = torch.clip(sample_points[:,:,:,1], 0, W_feat-1)
    
    sample_points_flatten = sample_points[:,:,:,0]*W_feat+sample_points[:,:,:,1]
    sample_points_flatten = sample_points_flatten.view(B, Nq, Np)
    
    query_index_val = weight
    
    feat_index_map = sample_points.new_zeros(B,Nq,H_feat*W_feat)
    feat_index_map.scatter_(-1,sample_points_flatten,query_index_val)
    feat_index_map = feat_index_map.permute(0,2,1)

    feat_update = []
    for id in range(B):
        feat_index_tmp = feat_index_map[id,:,:]
        query_tmp = query[id,:,:]
        feat_tmp = torch.matmul(feat_index_tmp, query_tmp)
        feat_update.append(feat_tmp)
    feat_update = torch.cat(feat_update, dim=0)
    feat_update = feat_update.permute(0,2,1).view(B,-1,H_feat,W_feat)
    return feat_update
       

def translate_to_linear_weight(ref: torch.Tensor, num_total,
                               tau=2.0, featmap_strides=None):
    if featmap_strides is None:
        grid = torch.arange(num_total, device=ref.device, dtype=ref.dtype).view(
            *[len(ref.shape)*[1, ]+[-1, ]])
    else:
        grid = torch.as_tensor(
            featmap_strides, device=ref.device, dtype=ref.dtype)
        grid = grid.log2().view(*[len(ref.shape)*[1, ]+[-1, ]])

    ref = ref.unsqueeze(-1).clone()
    l2 = (ref-grid).pow(2.0).div(tau).abs().neg()
    weight = torch.softmax(l2, dim=-1)

    return weight


def sampling_3d(
    sample_points: torch.Tensor,
    multi_lvl_values,
    featmap_strides,
    n_points: int = 1,
    num_levels: int = None,
    tau=2.0,
):
    B, n_queries, _t, n_groups_points, _ = sample_points.shape
    assert _t == 1
    B, C_feat, _, _ = multi_lvl_values[0].shape

    n_groups = n_groups_points//n_points
    n_channels = C_feat//n_groups

    if num_levels is None:
        num_levels = len(featmap_strides)

    sample_points_xy = sample_points[..., 0:2]

    sample_points_z = sample_points[..., 2].clone()
    sample_points_lvl_weight = translate_to_linear_weight(
        sample_points_z, num_levels,
        tau=tau, featmap_strides=featmap_strides)

    sample_points_lvl_weight_list = sample_points_lvl_weight.unbind(-1)

    out = sample_points.new_zeros(
        B, n_queries, n_groups, n_points, n_channels)

    for i in range(num_levels):
        value = multi_lvl_values[i]
        lvl_weights = sample_points_lvl_weight_list[i]

        stride = featmap_strides[i]

        mapping_size = value.new_tensor(
            [value.size(3), value.size(2)]).view(1, 1, 1, 1, -1) * stride
        normalized_xy = sample_points_xy/mapping_size

        out += sampling_each_level(normalized_xy, value,
                                   weight=lvl_weights, n_points=n_points)

    return out, None


def sampling_3d_alternate(
    sample_points: torch.Tensor,
    multi_lvl_values,
    query,
    featmap_strides,
    n_points: int = 1,
    num_levels: int = None,
    tau=2.0,
):
    B, n_queries, _t, n_groups_points, _ = sample_points.shape
    assert _t == 1
    B, C_feat, _, _ = multi_lvl_values[0].shape

    n_groups = n_groups_points//n_points
    n_channels = C_feat//n_groups

    if num_levels is None:
        num_levels = len(featmap_strides)

    sample_points_xy = sample_points[..., 0:2]

    sample_points_z = sample_points[..., 2].clone()
    sample_points_lvl_weight = translate_to_linear_weight(
        sample_points_z, num_levels,
        tau=tau, featmap_strides=featmap_strides)

    sample_points_lvl_weight_list = sample_points_lvl_weight.unbind(-1)

    out_query = sample_points.new_zeros(
        B, n_queries, n_groups, n_points, n_channels)

    out_feats = []
    for i in range(num_levels):
        value = multi_lvl_values[i]
        lvl_weights = sample_points_lvl_weight_list[i]

        stride = featmap_strides[i]

        mapping_size = value.new_tensor(
            [value.size(3), value.size(2)]).view(1, 1, 1, 1, -1) * stride
        normalized_xy = sample_points_xy/mapping_size

        out_query_lvl,  out_feat_lvl = sampling_each_level_alternate(normalized_xy, value, query,
                                   weight=lvl_weights, n_points=n_points)
        out_feats.append(out_feat_lvl)
        out_query += out_query_lvl

    return out_query, out_feats