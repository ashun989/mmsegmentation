_base_ = ['../_base_/models/deeplabv3_r50-d8.py',
          '../_base_/datasets/diffuse_made0.py',
          '../_base_/schedules/schedule_40k.py',
          '../_base_/default_runtime.py']

model = dict(
    decode_head=dict(num_classes=21), auxiliary_head=dict(num_classes=21))

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4)

