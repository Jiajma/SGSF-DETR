_base_ = [
    '../_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

classes = ('ship', )
data = dict(
    test=dict(classes=('ship', )),
    train=dict(classes=('ship', )),
    val=dict(classes=('ship', )))

evaluation = dict(interval=1, metric=['bbox', 'segm'], save_best='auto')

train_dataloader = dict(dataset=dict(metainfo=dict(classes=('ship',))))
val_dataloader = dict(dataset=dict(metainfo=dict(classes=('ship',))))
test_dataloader = dict(dataset=dict(metainfo=dict(classes=('ship',))))
