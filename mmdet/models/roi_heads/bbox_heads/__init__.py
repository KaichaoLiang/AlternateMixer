# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_head import BBoxHead
from .convfc_bbox_head import (ConvFCBBoxHead, Shared2FCBBoxHead,
                               Shared4Conv1FCBBoxHead)
from .dii_head import DIIHead
from .double_bbox_head import DoubleConvFCBBoxHead
from .sabl_head import SABLHead
from .scnet_bbox_head import SCNetBBoxHead
from .adamixer_decoder_stage import AdaMixerDecoderStage
from .adamixer_decoder_stage_gcn import AdaMixerDecoderGCNStage
from .adamixer_decoder_stage_gcndense import AdaMixerDecoderGCNDenseStage
from .adamixer_decoder_stage_withpe import AdaMixerDecoderStageWithPe
from .adamixer_decoder_stage_withpeabsolute import AdaMixerDecoderStageWithPeAbsolute
from .adamixer_decoder_stage_cat import AdaMixerDecoderStageCat
from .adamixer_decoder_stage_catpe import AdaMixerDecoderStageCatPe
from .adamixer_decoder_stage_catpeffn import AdaMixerDecoderStageCatPeFFN
from .adamixer_decoder_stage_catpeffn_headtail import AdaMixerDecoderStageCatPeFFNHeadTail
from .adamixer_decoder_stage_dualsample import AdaMixerDecoderStageDualSample
from .adamixer_decoder_stage_dualsample_test import AdaMixerDecoderStageDualSampleTest
from .alternatemixer_decoder_stage import AlternateMixerDecoderStage
from .adaptive_mixing_operator import AdaptiveMixing

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead', 'SABLHead', 'DIIHead',
    'SCNetBBoxHead', 'AdaMixerDecoderStage','AdaptiveMixing','AdaMixerDecoderStageWithPe','AdaMixerDecoderStageWithPeAbsolute'
    ,'AlternateMixerDecoderStage','AdaMixerDecoderStageCat', 'AdaMixerDecoderStageCatPe'
    ,'AdaMixerDecoderStageCatPeFFN','AdaMixerDecoderStageCatPeFFNHeadTail','AdaMixerDecoderStageDualSample','AdaMixerDecoderStageDualSampleTest'
    ,'AdaMixerDecoderGCNStage','AdaMixerDecoderGCNDenseStage'
]
