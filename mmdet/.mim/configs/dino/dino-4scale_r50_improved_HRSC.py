_base_ = ['dino-4scale_r50_8xb2-12e_coco.py']

# from deformable detr hyper
model = dict(
    backbone=dict(frozen_stages=-1),
    bbox_head=dict(loss_cls=dict(loss_weight=2.0)),
    positional_encoding=dict(offset=-0.5, temperature=10000),
    dn_cfg=dict(group_cfg=dict(num_dn_queries=300)))

# optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.0002),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

# 36
max_epochs = 60
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30],
        gamma=0.1)
]


# 设置类别
classes = (  # 'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'ship')


data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

data_root = 'D:/PyCharm/Py_Projects/data/HRSC2016_COCO/'

load_from = r"D:\PyCharm\Py_Projects\mmdetection\configs\dino\dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth"

# b32 10 / b5 10
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    pin_memory=True,
    dataset=dict(data_root=data_root,
                 ann_file='trainval.json',
                 data_prefix=dict(img='trainval/'),
                 metainfo=dict(classes=classes)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(data_root=data_root,
                 ann_file='test.json',
                 data_prefix=dict(img='test/'),
                 metainfo=dict(classes=classes)))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'test.json')
test_evaluator = val_evaluator
evaluation = dict(save_best='auto')
