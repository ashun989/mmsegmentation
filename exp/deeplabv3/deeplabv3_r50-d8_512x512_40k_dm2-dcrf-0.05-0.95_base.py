_base_ = ['../_base_/models/deeplabv3_r50-d8.py',
          '../_base_/datasets/diffuse_made2.py',
          '../_base_/schedules/schedule_40k.py',
          '../_base_/default_runtime.py']

model = dict(
    decode_head=dict(num_classes=21, with_prob=False),
    auxiliary_head=dict(num_classes=21, with_prob=False),
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        ann_dir='pseudo-dm2-dcrf-1.0-0.05-0.95-prob/train'
    )
)

