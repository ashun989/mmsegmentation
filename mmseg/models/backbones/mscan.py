import torch
import torch.nn as nn
import warnings

from ..builder import BACKBONES
from ..utils import AdaptivePadding
from mmcv.runner import ModuleList
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn import ConvModule
from mmcv.cnn import (Conv2d, build_activation_layer, build_norm_layer)
import torch.utils.checkpoint as cp
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init)
from torch.nn.modules.batchnorm import _BatchNorm


class FFN(nn.Module):
    def __init__(self,
                 num_channel,
                 hidden_channel,
                 act_cfg):
        self.proj1 = Conv2d(
            in_channels=num_channel,
            out_channels=hidden_channel,
            kernel_size=1,
            stride=1
        )

        self.dconv = Conv2d(
            in_channels=hidden_channel,
            out_channels=hidden_channel,
            kernel_size=3,
            stride=1,
            groups=hidden_channel,
        )

        self.proj2 = Conv2d(
            in_channels=hidden_channel,
            out_channels=num_channel,
            kernel_size=1,
            stride=1
        )
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        out = self.proj1(x)
        out = self.dconv(out)
        out = self.act(out)
        return self.proj2(out)


class MSCA(nn.Module):
    def __init__(self,
                 num_channel,
                 k1_size=5,
                 k_sizes=(7, 11, 21)):
        super(MSCA, self).__init__()
        self.adap_pad1 = AdaptivePadding(
            kernel_size=k1_size,
            stride=1,
        )
        self.conv1 = Conv2d(
            in_channels=num_channel,
            out_channels=num_channel,
            kernel_size=k1_size,
            stride=1,
            padding=0,
            groups=num_channel
        )
        self.sd_adap_pads = []
        self.sd_convs = []
        for k_size in k_sizes:
            self.sd_adap_pads.append(AdaptivePadding(kernel_size=k_size, stride=1))
            self.sd_convs.append(Conv2d(
                in_channels=num_channel, out_channels=num_channel,
                kernel_size=(1, k_size), stride=1, padding=0,
                groups=num_channel
            ))
            self.sd_convs.append(Conv2d(
                in_channels=num_channel, out_channels=num_channel,
                kernel_size=(k_size, 1), stride=1, padding=0,
                groups=num_channel
            ))
        self.channel_mix = Conv2d(
            in_channels=num_channel,
            out_channels=num_channel,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.skip_connection1 = nn.Identity()
        self.skip_connection2 = nn.Identity()

    def forward(self, x):
        x0 = self.skip_connection1(x)
        x = self.conv1(self.adap_pad1(x))
        attn = self.skip_connection2(x)
        for i in range(len(self.sd_adap_pads)):
            xi = self.sd_adap_pads[i](x)
            xi = self.sd_convs[2 * i](self.sd_convs[2 * i + 1](xi))
            attn += xi
        attn = self.channel_mix(attn)
        return torch.multiply(attn, x0)


class MultiScaleConvAttnModule(nn.Module):

    def __init__(self,
                 num_channel,
                 hidden_channel,
                 norm_cfg,
                 act_cfg
                 ):
        super(MultiScaleConvAttnModule, self).__init__()

        self.norm1 = build_norm_layer(norm_cfg, num_channel)[1]
        self.attn = MSCA(
            num_channel=num_channel
        )
        self.norm2 = build_norm_layer(norm_cfg, num_channel)[1]
        self.ffn = FFN(
            num_channel=num_channel,
            hidden_channel=hidden_channel,
            act_cfg=act_cfg
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.attn(x)
        x = identity + x
        identity = x
        x = self.norm2(x)
        x = self.ffn(x)
        return identity + x


@BACKBONES.register_module()
class MSCAN(nn.Module):
    """The MSCAN

    """

    def __init__(self,
                 in_channels=3,
                 num_channels=[32, 64, 160, 256],
                 num_blocks=[3, 3, 5, 2],
                 exp_ratios=[8, 8, 4, 4],
                 out_channel=256,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='GELU'),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False
                 ):
        super().__init__()
        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        assert len(num_channels) == len(num_blocks) == len(exp_ratios)

        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.exp_rations = exp_ratios
        self.out_channel = out_channel
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp

        self.embedding = ConvModule(
            in_channels=in_channels,
            out_channels=self.num_channels[0],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

        self.layers = ModuleList()
        for i, num_block in enumerate(self.num_blocks):
            in_c = self.num_channels[i]
            out_c = self.num_channels[i + 1] if i + 1 > len(self.num_channels) else self.out_channel
            hid_c = self.num_channels[i] * self.exp_rations[i]
            layer = ModuleList()
            for _ in range(num_block):
                layer.append(MultiScaleConvAttnModule(
                    num_channel=in_c,
                    hidden_channel=hid_c,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg
                ))
            downsample = Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False if self.norm_cfg['type'] == 'BN' else True
            ) if i < len(self.num_blocks) - 1 else nn.Identity() if in_c == out_c else Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False if self.norm_cfg['type'] == 'BN' else True
            )

            norm = build_norm_layer(self.norm_cfg, out_c)[1]
            self.layers.append(ModuleList[layer, downsample, norm])

    def forward(self, x):
        def _inner_forward(x):
            x = self.embedding(x)
            x = self.layers(x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

    def init_weights(self, pretrained=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, mode='fan_in', bias=0.)
            elif isinstance(m, (_BatchNorm , nn.GroupNorm, nn.LayerNorm)):
                constant_init(m, val=1.0, bias=0.)


