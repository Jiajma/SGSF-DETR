_base_ = './rtmdet_l_8xb32-300e_coco.py'

model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(in_channels=320, feat_channels=320, num_classes=1))


# 设置类别
classes = (  # 'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    # 'small-vehicle', 'large-vehicle',
    # 'ship',
    # 'tennis-court',
    # 'basketball-court', 'storage-tank', 'soccer-ball-field',
    # 'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    'ship')
    # 'BulkCarrier', 'Fishing', 'Tanker', 'Unspecified')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

data_root = 'D:/PyCharm/Py_Projects/data/HRSC2016_COCO/'
    # '/home/w/jiajie/data/iSAID/'
# '/home/w/jiajie/data/SSDD/Official-SSDD-OPEN/BBox_RBox_PSeg_SSDD/coco_style/'
# '/home/w/jiajie/data/HRSID/HRSID_JPG/'

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

# 加载预训练模型
# load_from = '/home/w/jiajie/mmdetection/checkpoints/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'
# load_from = '/home/w/jiajie/mmdetection/work_dirs/rtmdet_x_dota_AP0.193_lr0.004/epoch_300.pth'
# load_from = 'D:/PyCharm/Py_Projects/mmdetection/work_dirs/rtmdet_x_HRSID_AP0.677_lr0.004/epoch_300.pth'
