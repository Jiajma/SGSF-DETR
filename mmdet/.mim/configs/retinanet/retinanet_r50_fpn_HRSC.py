_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    './retinanet_tta.py'
]

# optimizer
optim_wrapper = dict(
    # lr=0.01
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# 设置类别
classes = ('ship', )


data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

data_root = 'D:/PyCharm/Py_Projects/data/HRSC2016_COCO/'


# b32 10 / b5 10
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    pin_memory=True,
    dataset=dict(data_root=data_root,
                 ann_file='trainval.json',
                 data_prefix=dict(img='trainval/'),
                 metainfo=dict(classes=classes)))
val_dataloader = dict(
    batch_size=2,
    num_workers=1,
    dataset=dict(data_root=data_root,
                 ann_file='test.json',
                 data_prefix=dict(img='test/'),
                 metainfo=dict(classes=classes)))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'test.json')
test_evaluator = val_evaluator
evaluation = dict(save_best='auto')
