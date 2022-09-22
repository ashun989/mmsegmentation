# model settings
_base_ = [
    '../_base_/models/eunet-t.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

checkpoint = '/root/andrewhoux/zqs/mmsegmentation/work_dirs/eunet/model_best.pth.tar'
# checkpoint = '/home/ashun/projects/mmsegmentation/work_dirs/eunet/model_best.pth.tar'

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
    ),
    decode_head=dict(
        type='NLHead2',
        in_channels=[48, 96, 192],
        in_index=[1, 2, 3],
        channels=192,
        dropout_ratio=0.1,
        reduction=2,
        use_scale=True,
        mode='embedded_gaussian',
        num_classes=150,
        norm_cfg=norm_cfg,
        interpolate_mode='bilinear',
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

data = dict(samples_per_gpu=8, workers_per_gpu=4)
