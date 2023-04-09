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
        ann_dir='deeplabv3-r50-d8_512x512_40k',
    )
)

