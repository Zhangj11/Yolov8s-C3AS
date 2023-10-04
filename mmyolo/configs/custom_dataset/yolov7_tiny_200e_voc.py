_base_ = '../yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco.py'

# ========================modified parameters========================
# -----data related-----
data_root = '../../VOCdevkit'  # Root path of data
# data_root = '/home/ubuntu/E/zbc/VOCdevkit'
dataset_type = 'YOLOv5VOCDataset'

num_classes = 20

max_epochs = 200
train_batch_size_per_gpu = 8
train_num_workers = 4

val_batch_size_per_gpu = 1
val_num_workers = 2

# persistent_workers must be False if num_workers is 0
persistent_workers = True

# -----model related-----
# Data augmentation
max_translate_ratio = 0.1  # YOLOv5RandomAffine
scaling_ratio_range = (0.5, 1.6)  # YOLOv5RandomAffine
mixup_prob = 0.05  # YOLOv5MixUp
randchoice_mosaic_prob = [0.8, 0.2]
mixup_alpha = 8.0  # YOLOv5MixUp
mixup_beta = 8.0  # YOLOv5MixUp

# -----train val related-----
loss_cls_weight = 0.5
loss_obj_weight = 1.0

lr_factor = 0.01  # Learning rate scaling factor
# ===============================Unmodified in most cases====================
num_classes = _base_.num_classes
num_det_layers = _base_.num_det_layers
img_scale = _base_.img_scale
pre_transform = _base_.pre_transform
model = dict(
    backbone=dict(
        arch='Tiny', act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
    neck=dict(
        is_tiny_version=True,
        in_channels=[128, 256, 512],
        out_channels=[64, 128, 256],
        block_cfg=dict(
            _delete_=True, type='TinyDownSampleBlock', middle_ratio=0.25),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        use_repconv_outs=False),
    bbox_head=dict(
        head_module=dict(in_channels=[128, 256, 512]),
        loss_cls=dict(loss_weight=loss_cls_weight *
                      (num_classes / 80 * 3 / num_det_layers)),
        loss_obj=dict(loss_weight=loss_obj_weight *
                      ((img_scale[0] / 640)**2 * 3 / num_det_layers))))

mosiac4_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=max_translate_ratio,  # change
        scaling_ratio_range=scaling_ratio_range,  # change
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
]

mosiac9_pipeline = [
    dict(
        type='Mosaic9',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=max_translate_ratio,  # change
        scaling_ratio_range=scaling_ratio_range,  # change
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
]

randchoice_mosaic_pipeline = dict(
    type='RandomChoice',
    transforms=[mosiac4_pipeline, mosiac9_pipeline],
    prob=randchoice_mosaic_prob)

train_pipeline = [
    *pre_transform,
    randchoice_mosaic_pipeline,
    dict(
        type='YOLOv5MixUp',
        alpha=mixup_alpha,
        beta=mixup_beta,
        prob=mixup_prob,  # change
        pre_transform=[*pre_transform, randchoice_mosaic_pipeline]),
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

# TODO: Support using coco metric in voc dataset
val_evaluator = dict(
    _delete_=True, type='mmdet.VOCMetric', metric='mAP', eval_mode='area')

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=1, save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))

train_cfg = dict(max_epochs=max_epochs, val_interval=10)  # 每 val_interval 轮迭代进行一次测试评估

# visualizer = dict(
#     vis_backends=[dict(type='LocalVisBackend'),
#                   dict(
#                         type='WandbVisBackend',
#                         init_kwargs=dict(project='VOC07+12', name='yolo_v7_tiny')
#                     )])
