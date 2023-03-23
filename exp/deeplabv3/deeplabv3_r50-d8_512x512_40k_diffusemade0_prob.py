_base_ = ['./deeplabv3_r50-d8_512x512_40k_diffusemade0_base.py']


model = dict(
    decode_head=dict(
        use_prob=dict(method='direct')
    ),
    auxiliary_head=dict(
        use_prob=dict(method='direct')
    )
)