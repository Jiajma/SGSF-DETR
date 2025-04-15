_base_ = './yolox_s_8xb8-300e_coco.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25),
    neck=dict(
        in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(num_classes=1, in_channels=320, feat_channels=320))


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

#train_dataset = dict(dataset=dict(metainfo=dict(classes=classes)))
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
)
val_dataloader = dict(
    batch_size=2,
    num_workers=1,
    #dataset=dict(metainfo=dict(classes=classes))
)
test_dataloader = val_dataloader


# 加载预训练模型
load_from = 'checkpoints/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
