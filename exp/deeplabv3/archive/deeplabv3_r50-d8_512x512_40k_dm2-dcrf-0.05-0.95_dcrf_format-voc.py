_base_ = ['./deeplabv3_r50-d8_512x512_40k_dm2-dcrf-0.05-0.95_base.py']

model = dict(
    post_cfg=dict(
        method='dcrf',
        kwargs=dict(iter_max=10,
                    pos_xy_std=1,
                    pos_w=3,
                    bi_xy_std=67,
                    bi_rgb_std=3,
                    bi_w=4, )
    )
)

data = dict(
    test=dict(
        type='PascalVOCDataset2',
        ann_dir='SegmentationClassAUG',
        split='ImageSets/Segmentation/train_aug.txt'
    )
)
