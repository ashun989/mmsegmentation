_base_ = ['./deeplabv3_r50-d8_512x512_40k_diffusemade0_base2.py']


model = dict(
    decode_head=dict(
        use_prob=dict(method='direct')
    ),
    auxiliary_head=dict(
        use_prob=dict(method='direct')
    )
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4)  # for debug