"""
在backbone最后一处加上CBAM块
"""

_base_ = '../yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'

# dataset settings
data_root = 'C:/app/daima/object_detection/VOCdevkit/'
dataset_type = 'YOLOv5VOCDataset'

# parameters that often need to be modified
num_classes = 20
img_scale = (640, 640)  # width, height
max_epochs = 150
train_batch_size_per_gpu = 8
train_num_workers = 4
val_batch_size_per_gpu = 1
val_num_workers = 2

# persistent_workers must be False if num_workers is 0.
persistent_workers = True

lr_factor = 0.01
affine_scale = 0.5

num_det_layers = 3

model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(
                    type='CBAM'),
                stages=(False, False, True, True))]),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

albu_train_transforms = _base_.albu_train_transforms
pre_transform = _base_.pre_transform

train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    # last_transform，与v8完全一致
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader = dict(
    _delete_=True,
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        # 我的理解是在07+12上训练
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='VOC2007/ImageSets/Main/trainval.txt',
                data_prefix=dict(sub_data_root='VOC2007/'),
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='VOC2012/ImageSets/Main/trainval.txt',
                data_prefix=dict(sub_data_root='VOC2012/'),
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=train_pipeline)
        ],
        # Use ignore_keys to avoid judging metainfo is
        # not equal in `ConcatDataset`.
        ignore_keys='dataset_type'),
    collate_fn=dict(type='yolov5_collate'))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/test.txt',  # 在07上验证
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True
        ))

test_dataloader = val_dataloader

param_scheduler = None
optim_wrapper = dict(
    optimizer=dict(
        batch_size_per_gpu=train_batch_size_per_gpu))

default_hooks = dict(
    param_scheduler=dict(
        lr_factor=lr_factor,
        max_epochs=max_epochs))

# TODO: Support using coco metric in voc dataset
val_evaluator = dict(
    _delete_=True, type='mmdet.VOCMetric', metric='mAP', eval_mode='area')

test_evaluator = val_evaluator

# 训练的配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,  # 最大训练轮次 300 轮
    val_interval=10)  # 验证间隔，每 10 个 epoch 验证一次

# tensorboard记录
# visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])

# wandb可视化
visualizer = dict(
    vis_backends = [dict(type='LocalVisBackend'),
                    dict(
                        type='WandbVisBackend',
                        init_kwargs=dict(
                        project='Yolo',
                        name='yolo_v8_CBAM1'
                        )
                    )])

