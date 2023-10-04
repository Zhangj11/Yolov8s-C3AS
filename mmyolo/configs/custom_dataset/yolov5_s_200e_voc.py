_base_ = '../yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

# dataset settings
data_root = '../../VOCdevkit'
# data_root = '/home/ubuntu/E/zbc/VOCdevkit'
dataset_type = 'YOLOv5VOCDataset'

# parameters that often need to be modified
num_classes = 20
img_scale = (640, 640)  # width, height

max_epochs = 200
train_batch_size_per_gpu = 16
train_num_workers = 8

val_batch_size_per_gpu = 1
val_num_workers = 2

persistent_workers = True
lr_factor = 0.15135
affine_scale = 0.5

num_det_layers = 3

# tta_img_scales = [img_scale, (416, 416), (640, 640)]

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        # loss_cls=dict(
        #     loss_weight=0.21638 * (num_classes / 80 * 3 / num_det_layers)
        # )
    )
)

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True)
]

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
    dict(
        type='YOLOv5HSVRandomAug',
        hue_delta=0.01041,
        saturation_delta=0.54703,
        value_delta=0.27739),
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
        ann_file='VOC2007/ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='VOC2007/'),
        test_mode=True))

test_dataloader = val_dataloader

default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs))

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu

# TODO: Support using coco metric in voc dataset
val_evaluator = dict(
    _delete_=True, type='mmdet.VOCMetric', metric='mAP', eval_mode='area')

test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10)

# visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])

# visualizer = dict(
#     vis_backends=[dict(type='LocalVisBackend'),
#                     dict(
#                         type='WandbVisBackend',
#                         init_kwargs=dict(
#                         project='VOC07+12',
#                         name='yolo_v5s'
#                         )
#                     )])
