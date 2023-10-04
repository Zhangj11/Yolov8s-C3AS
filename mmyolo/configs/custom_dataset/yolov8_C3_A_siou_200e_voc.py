"""
在07+12上训练，在07上验证
"""
_base_ = '../yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'

# dataset settings
data_root = '../../VOCdevkit'
# data_root = '/home/ubuntu/E/zbc/VOCdevkit'
dataset_type = 'YOLOv5VOCDataset'

# parameters that often need to be modified
num_classes = 20
img_scale = (640, 640)  # width, height

close_mosaic_epochs = 10

max_epochs = 200
train_batch_size_per_gpu = 16
train_num_workers = 8

val_batch_size_per_gpu = 1
val_num_workers = 2

persistent_workers = True

deepen_factor = 0.33
widen_factor = 0.5
last_stage_out_channels = 1024
affine_scale = 0.5

model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(
                    type='CBAM'),
                stages=(True, True, True, False))]),
    neck=[
        dict(
            type='YOLOv8PAFPN',
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            in_channels=[256, 512, last_stage_out_channels],
            out_channels=[256, 512, last_stage_out_channels],
            num_csp_blocks=3,
            # norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True)),
        dict(
            type='ASFFNeck',
            widen_factor=widen_factor,
            use_att='ASFF_sim'),
    ],
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            act_cfg=dict(type='SiLU', inplace=True),
            ),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='siou',
            return_iou=False)
    ),
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

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
_base_.custom_hooks[1].switch_epoch = max_epochs - close_mosaic_epochs

# TODO: Support using coco metric in voc dataset
val_evaluator = dict(
    _delete_=True, type='mmdet.VOCMetric', metric='mAP', eval_mode='area')

test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=1, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))

train_cfg = dict(max_epochs=max_epochs, val_interval=10)  # 每 val_interval 轮迭代进行一次测试评估

visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(
                        type='WandbVisBackend',
                        init_kwargs=dict(project='VOC07+12', name='yolo_v8s_C3_A_siou')
                    )])
