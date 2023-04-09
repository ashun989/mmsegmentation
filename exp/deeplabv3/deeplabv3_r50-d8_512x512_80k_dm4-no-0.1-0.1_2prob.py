_base_ = ['../_base_/models/deeplabv3_r50-d8.py',
          '../_base_/datasets/diffuse_made4.py',
          '../_base_/schedules/schedule_80k.py',
          '../_base_/default_runtime.py']


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (320, 320)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotationWithProbs'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='MyFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]


model = dict(
    decode_head=dict(num_classes=21,
                     with_prob=True,
                     use_prob=dict(method='two_prob')),
    auxiliary_head=dict(num_classes=21,
                        with_prob=True,
                        use_prob=dict(method='two_prob')),
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        ann_dir='pseudo-1.0-no-1.0-0.1-0.1/train',
        pipeline=train_pipeline
    )
)

