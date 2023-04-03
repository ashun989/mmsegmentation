_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py',
    '../_base_/datasets/pascal_voc12_aug.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        ann_dir='SegmentationClassAug_pse1',
    )
)

model = dict(
    decode_head=dict(num_classes=21,
                     with_prob=False),
    auxiliary_head=dict(num_classes=21,
                        with_prob=False)
)
