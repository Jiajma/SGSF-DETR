_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


# 设置类别
classes = (  # 'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    # 'small-vehicle', 'large-vehicle',
    #'ship',
    # 'tennis-court',
    # 'basketball-court', 'storage-tank', 'soccer-ball-field',
    # 'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    'Small_Vehicle', 'Large_Vehicle', 'plane', 'storage_tank', 'ship', 'Swimming_pool', 'Harbor', 'tennis_court', 'Ground_Track_Field', 'Soccer_ball_field', 'baseball_diamond', 'Bridge', 'basketball_court', 'Roundabout', 'Helicopter')

data = dict(
    train=dict(classes=classes),
    val=dict(classes=classes),
    test=dict(classes=classes))

data_root = 'D:/PyCharm/Py_Projects/data/iSAID/'
    # '/home/w/jiajie/data/iSAID/'
# '/home/w/jiajie/data/SSDD/Official-SSDD-OPEN/BBox_RBox_PSeg_SSDD/coco_style/'
# '/home/w/jiajie/data/HRSID/HRSID_JPG/'

# b32 10 / b5 10
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    pin_memory=True,
    dataset=dict(data_root=data_root,
                 ann_file='train/Annotations/iSAID_train.json',
                 data_prefix=dict(img='train/images/'),
                 metainfo=dict(classes=classes)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(data_root=data_root,
                 ann_file='val/Annotations/iSAID_val.json',
                 data_prefix=dict(img='val/images/'),
                 metainfo=dict(classes=classes)))
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'val/Annotations/iSAID_val.json')
test_evaluator = val_evaluator
evaluation = dict(save_best='auto')

# 加载预训练模型
# load_from = '/home/w/jiajie/mmdetection/checkpoints/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth'
# load_from = '/home/w/jiajie/mmdetection/work_dirs/rtmdet_x_dota_AP0.193_lr0.004/epoch_300.pth'
