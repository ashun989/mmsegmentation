_base_ = ['../_base_/models/deeplabv3_r50-d8.py',
          '../_base_/datasets/diffuse_made0.py',
          '../_base_/schedules/schedule_40k.py',
          '../_base_/default_runtime.py']

model = dict(
    decode_head=dict(num_classes=21, with_prob=False),
    auxiliary_head=dict(num_classes=21, with_prob=False)
)


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (320, 320)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        ann_dir='pseudo_masks_aug',
        pipeline=train_pipeline
    )
)

