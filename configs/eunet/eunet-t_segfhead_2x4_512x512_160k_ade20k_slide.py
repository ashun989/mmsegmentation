_base_ = ['./eunet-t_segfhead_2x4_512x512_160k_ade20k.py']

model = dict(
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)