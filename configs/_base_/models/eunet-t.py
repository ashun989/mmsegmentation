# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MixNet',
        img_size=224,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[24, 48, 96, 192],
        strides=[2, 2, 2, 1],
        mlp_ratios=[4, 4, 4, 4],
        out_indices=(0, 1, 2, 3),
        layer_scale_init_value=1e-6,
        head_init_scale=1.,
        drop_path_rate=0.,
        drop_rate=0.
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[24, 48, 96, 192],
        in_index=[0, 1, 2, 3],
        channels=192,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
        )
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
