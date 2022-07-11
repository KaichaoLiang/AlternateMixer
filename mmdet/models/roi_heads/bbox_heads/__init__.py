# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_head import BBoxHead
from .convfc_bbox_head import (ConvFCBBoxHead, Shared2FCBBoxHead,
                               Shared4Conv1FCBBoxHead)
from .dii_head import DIIHead
from .double_bbox_head import DoubleConvFCBBoxHead
from .sabl_head import SABLHead
from .scnet_bbox_head import SCNetBBoxHead
from .adamixer_decoder_stage import AdaMixerDecoderStage
from .alternatemixer_decoder_stage import AlternateMixerDecoderStage
from .adaptive_mixing_operator import AdaptiveMixing
from .alternate_mixing_operator import AlternateMixing

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead', 'SABLHead', 'DIIHead',
    'SCNetBBoxHead', 'AdaMixerDecoderStage','AdaptiveMixing','AlternateMixing'
    ,'AlternateMixerDecoderStage'
]
