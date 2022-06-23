from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import set_random_seed
import mmcv
import os.path as osp

cfg = mmcv.Config.fromfile('configs/fcos_mixer/fcosmixer_r50_1x_coco_improvefcoshead_clsfeat.py')

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = ['bbox']
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 1
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 4
 
# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device='cuda'
cfg.work_dir='Train_FcosMixer_BestFcosConfig_ClsFeat'

cfg.data_root = '/home/lkc20/DetectionTransformer/Coco/'
cfg.data.train.ann_file = cfg.data_root + 'annotations/instances_train2017.json',
cfg.data.train.img_prefix = cfg.data_root + 'train2017/'
cfg.data.val.ann_file = cfg.data_root + 'annotations/instances_val2017.json',
cfg.data.val.img_prefix = cfg.data_root + 'val2017/'
cfg.data.test.ann_file = cfg.data_root + 'annotations/instances_val2017.json',
cfg.data.test.img_prefix = cfg.data_root + 'val2017/'

# We can also use tensorboard to log the training process
cfg.log_config.interval=200
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    #dict(type='TensorboardLoggerHook')
    ]

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
model.init_weights()

# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)