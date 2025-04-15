_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/dota(coco)_detection.py',
    '../_base_/schedules/schedule_20e.py', '../_base_/default_runtime.py'
]

# 设置参数
optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=2)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.0001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=30,
        by_epoch=True,
        milestones=[23, 29],
        gamma=0.1)
]

# 0.005
#test_cfg = dict(rcnn=dict(score_thr=0.005, nms=dict(type='nms', iou_threshold=0.5), ax_per_img=100))


# 设置类别
model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))

classes = (  # 'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    # 'small-vehicle', 'large-vehicle',
    'ship',
    # 'tennis-court',
    # 'basketball-court', 'storage-tank', 'soccer-ball-field',
    # 'roundabout', 'harbor', 'swimming-pool', 'helicopter')
)

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

train_dataloader = dict(dataset=dict(metainfo=dict(classes=classes)))
val_dataloader = dict(dataset=dict(metainfo=dict(classes=classes)))
test_dataloader = dict(dataset=dict(metainfo=dict(classes=classes)))

# 加载预训练模型
load_from = 'D:/PyCharm/Py_Projects/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
