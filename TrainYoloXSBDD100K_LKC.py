from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.apis import set_random_seed
import mmcv
import os.path as osp

cfg = mmcv.Config.fromfile('configs/yolox/yolox_s_8x8_300e_coco.py')

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = ['bbox']
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 1
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 40
 
# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device='cuda'
cfg.work_dir='YoloXS_BDD100K'

cfg.dataset_type='CocoBDDDataset'

cfg.data_root = '/home/lkc20/mmdetection/datasets/bdd100s/'

cfg.data.train.dataset.type=cfg.dataset_type
cfg.data.train.dataset.ann_file = cfg.data_root + 'annotations/labels_train.json',
cfg.data.train.dataset.img_prefix = cfg.data_root + 'train/'
cfg.data.train.dataset.data_root=cfg.data_root
cfg.data.val.type= cfg.dataset_type
cfg.data.val.ann_file = cfg.data_root + 'annotations/labels_val.json',
cfg.data.val.img_prefix = cfg.data_root + 'valc/'
cfg.data.val.data_root=cfg.data_root
cfg.data.test.type= cfg.dataset_type
cfg.data.test.ann_file = cfg.data_root + 'annotations/labels_val.json',
cfg.data.test.img_prefix = cfg.data_root + 'val/'
cfg.data.test.data_root=cfg.data_root

cfg.model.bbox_head.num_classes = 10
# We can also use tensorboard to log the training process
cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]
cfg.log_config.interval = 2
print(f'Config:\n{cfg.pretty_text}')
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
