# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
        dict(type='MMSegWandbHook',
             init_kwargs={
                 'entity': "ashun989",
                 'project': "dm-seg"
             },
             interval=50,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=100)
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
