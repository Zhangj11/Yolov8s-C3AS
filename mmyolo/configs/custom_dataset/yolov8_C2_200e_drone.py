"""
无人机数据集
yolov8_s_C2,不加载预训练权重，200e
stages=(False, False, True, True)
"""
_base_ = '../yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py'

data_root = './data/drone/'
class_name = ('drone', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])

close_mosaic_epochs = 10

max_epochs = 200
train_batch_size_per_gpu = 16
train_num_workers = 8

# load_from = 'work_dirs/yolov8_s_100e_drone/best_coco_bbox_mAP_epoch_100.pth'

model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(
                    type='CBAM'),
                stages=(False, False, True, True))]),
    bbox_head=dict(head_module=dict(num_classes=num_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_classes)))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/annotations_all.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
_base_.custom_hooks[1].switch_epoch = max_epochs - close_mosaic_epochs

val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
test_evaluator = val_evaluator

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=1, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)  # 每 val_interval 轮迭代进行一次测试评估

# tensorboard记录
# visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])

# wandb可视化
visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'),
                    dict(
                        type='WandbVisBackend',
                        init_kwargs=dict(
                        project='Drone_N',
                        name='yolo_v8s_C2'
                        )
                    )])
