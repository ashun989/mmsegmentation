_base_ = ['../_base_/models/deeplabv3_r50-d8.py',
          '../_base_/datasets/diffuse_made4.py',
          '../_base_/schedules/schedule_80k.py',
          '../_base_/default_runtime.py']



model = dict(
    decode_head=dict(num_classes=21,
                     with_prob=False,
                     ),
    auxiliary_head=dict(num_classes=21,
                        with_prob=False,
                        ),
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        ann_dir='pseudo-0.3-dcrf-1.0-0.05-0.95/train',
    )
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
)

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)