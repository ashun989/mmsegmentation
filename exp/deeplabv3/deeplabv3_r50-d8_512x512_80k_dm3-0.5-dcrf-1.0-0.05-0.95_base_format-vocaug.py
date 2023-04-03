_base_ = ['./deeplabv3_r50-d8_512x512_80k_dm3-0.5-dcrf-1.0-0.05-0.95_base.py']

data = dict(
    test=dict(
        type='PascalVOCDataset2',
        ann_dir='SegmentationClassAUG',
        split='ImageSets/Segmentation/train_aug.txt'
    )
)
