_base_ = ['../_base_/models/deeplabv3_r50-d8.py',
          '../_base_/datasets/diffuse_made5.py',
          '../_base_/schedules/schedule_80k_adamw.py',
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
        ann_dir='out-combine/out_ann_dir/pow-0.5-dcrf-0.05-0.95',
    )
)


