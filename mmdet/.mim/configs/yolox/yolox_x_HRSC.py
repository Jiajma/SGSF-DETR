_base_ = './yolox_s_8xb8-300e_coco.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(num_classes=1, in_channels=320, feat_channels=320))


load_from = r"D:\PyCharm\Py_Projects\mmdetection\configs\yolox\yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth"


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


evaluation = dict(save_best='auto')
