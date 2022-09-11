import torch.nn as nn
import warnings

from ..builder import BACKBONES
from ..utils import PatchEmbed
from mmcv.runner import ModuleList
from torch.nn.modules.utils import _pair as to_2tuple
from mmcv.cnn import ConvModule
from mmcv.cnn import Conv2d, build_activation_layer, build_norm_layer


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

    def forward(self, x, identity=None):
        out = self.proj1(x)
        out = self.dconv(out)
        out = self.act(out)
        if identity is None:
            identity = x
        return self.proj2(out) + identity


class MSCA(nn.Module):
    def __init__(self):
        super(MSCA, self).__init__()

    def forward(self, x, identity=None):
        pass


class MultiScaleConvAttnModule(nn.Module):

    def __init__(self,
                 num_channel,
                 hidden_channel,
                 norm_cfg,
                 act_cfg
                 ):
        super(MultiScaleConvAttnModule, self).__init__()

        self.attn = MSCA()
        self.ffn = FFN(
            num_channel=num_channel,
            hidden_channel=hidden_channel,
            act_cfg=act_cfg
        )

    def forward(self, x):
        pass


@BACKBONES.register_module()
class MSCAN(nn.Module):
    """The MSCAN

    """

    def __init__(self,
                 in_channels=3,
                 img_size=224,
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

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        self.img_size = img_size
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
        pass

    def init_weights(self, pretrained=None):
        pass
