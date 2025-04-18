_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
model = dict(
    type='MaskScoringRCNN',
    roi_head=dict(
        type='MaskScoringRoIHead',
        mask_iou_head=dict(
            type='MaskIoUHead',
            num_convs=4,
            num_fcs=2,
            roi_feat_size=14,
            in_channels=256,
            conv_out_channels=256,
            fc_out_channels=1024,
            num_classes=1)),
    # model training and testing settings
    train_cfg=dict(rcnn=dict(mask_thr_binary=0.5)))

classes = ('ship', )
data = dict(
    test=dict(classes=('ship', )),
    train=dict(classes=('ship', )),
    val=dict(classes=('ship', )))

evaluation = dict(interval=1, metric=['bbox', 'segm'], save_best='auto')

train_dataloader = dict(dataset=dict(metainfo=dict(classes=('ship',))))
val_dataloader = dict(dataset=dict(metainfo=dict(classes=('ship',))))
test_dataloader = dict(dataset=dict(metainfo=dict(classes=('ship',))))
