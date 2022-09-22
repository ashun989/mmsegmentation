# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import NonLocal2d
import torch.nn as nn
from ..builder import HEADS
from .fcn_head import FCNHead
from .decode_head import BaseDecodeHead
from mmcv.cnn import build_norm_layer, Conv2d, ConvModule
from mmcv.runner import Sequential, ModuleList
from mmseg.ops import resize

@HEADS.register_module()
class NLHead(FCNHead):
    """Non-local Neural Networks.

    This head is the implementation of `NLNet
    <https://arxiv.org/abs/1711.07971>`_.

    Args:
        reduction (int): Reduction factor of projection transform. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            sqrt(1/inter_channels). Default: True.
        mode (str): The nonlocal mode. Options are 'embedded_gaussian',
            'dot_product'. Default: 'embedded_gaussian.'.
    """

    def __init__(self,
                 reduction=2,
                 use_scale=True,
                 mode='embedded_gaussian',
                 **kwargs):
        super(NLHead, self).__init__(num_convs=2, **kwargs)
        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        self.nl_block = NonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs[0](x)
        output = self.nl_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output


@HEADS.register_module()
class NLHead2(BaseDecodeHead):
    """Non-local Neural Networks.

    This head is the implementation of `NLNet
    <https://arxiv.org/abs/1711.07971>`_.

    Args:
        reduction (int): Reduction factor of projection transform. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            sqrt(1/inter_channels). Default: True.
        mode (str): The nonlocal mode. Options are 'embedded_gaussian',
            'dot_product'. Default: 'embedded_gaussian.'.
    """

    def __init__(self,
                 reduction=2,
                 use_scale=True,
                 mode='embedded_gaussian',
                 interpolate_mode='bilinear',
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 **kwargs):
        super(NLHead2, self).__init__(input_transform='multiple_select', **kwargs)
        self.reduction = reduction
        self.use_scale = use_scale
        self.mode = mode
        self.interpolate_mode = interpolate_mode
        self.concat_input = concat_input
        convs = []
        conv_padding = (kernel_size // 2) * dilation
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        self.convs = Sequential(*convs)

        if self.concat_input:
            self.conv_cat = ConvModule(
                2 * self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        self.nl_block = NonLocal2d(
            in_channels=self.channels,
            reduction=self.reduction,
            use_scale=self.use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode=self.mode)
        self.projs = ModuleList()
        for in_channel in self.in_channels:
            self.projs.append(
                Sequential(
                    Conv2d(in_channels=in_channel, out_channels=self.channels, kernel_size=1),
                    build_norm_layer(self.norm_cfg, self.channels)[1]
                )
            )

    def forward(self, inputs):
        """Forward function."""
        features = self._transform_inputs(inputs)

        tmps = []
        for i, feature in enumerate(features):
            proj = self.projs[i]
            tmps.append(
                resize(
                    input=proj(feature),
                    size=features[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        x = sum(tmps)
        output = self.convs[0](x)
        output = self.nl_block(output)
        output = self.convs[1](output)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output
