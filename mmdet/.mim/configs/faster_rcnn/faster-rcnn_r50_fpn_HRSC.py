_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/HRSID(coco)_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


# 设置类别
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

data_root = 'D:/PyCharm/Py_Projects/data/HRSC2016_COCO/'

# b32 10 / b5 10
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
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

# 加载预训练模型
# load_from = 'checkpoints/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'
