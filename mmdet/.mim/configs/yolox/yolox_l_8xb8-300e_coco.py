_base_ = './yolox_s_8xb8-300e_coco.py'

# model settings
model = dict(
    backbone=dict(deepen_factor=1.0, widen_factor=1.0),
    neck=dict(
        in_channels=[256, 512, 1024], out_channels=256, num_csp_blocks=3),
    bbox_head=dict(in_channels=256, feat_channels=256))


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

train_dataset = dict(dataset=dict(metainfo=dict(classes=classes)))
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
)
val_dataloader = dict(
    batch_size=2,
    num_workers=1,
    dataset=dict(metainfo=dict(classes=classes))
)
test_dataloader = val_dataloader


# 加载预训练模型
load_from = 'checkpoints/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
