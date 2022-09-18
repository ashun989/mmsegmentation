import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import warnings

from ..builder import BACKBONES
from mmcv.runner import BaseModule


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        # self.norm1 = nn.BatchNorm2d(dim)
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        # self.norm2 = nn.BatchNorm2d(dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)

        return x


class SpatialAttention(nn.Module):
    def __init__(self, dim, expand_ratio=2):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        # self.norm = nn.BatchNorm2d(dim)
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.ReLU6(),
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        )

        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = MLP(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.norm(x)
        a = self.a(x)
        # a = F.normalize(a, p=2.0)
        x = a * self.v(x)
        x = self.proj(x)

        return x


class Block(nn.Module):
    def __init__(self, index, dim, mlp_ratio=4., drop_path=0.):
        super().__init__()

        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        # self.layer_scale_2 = nn.Parameter(
        #     layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        # x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


@BACKBONES.register_module()
class MixNet(BaseModule):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 num_classes=1000,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 strides=[2, 2, 2, 2],
                 mlp_ratios=[4, 4, 4, 4],
                 out_indices=(0, 1, 2, 3),
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.,
                 drop_path_rate=0.,
                 drop_rate=0.,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(MixNet, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.num_classes = num_classes
        self.depths = depths
        self.drop_rate = drop_rate
        self.out_indices = out_indices

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=4, padding=3),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(len(dims) - 1):
            stride = strides[i]
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=stride, stride=stride),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[Block(index=cur + j, dim=dims[i], drop_path=dp_rates[cur + j], mlp_ratio=mlp_ratios[i]) for j in
                  range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # self.head = nn.Sequential(
        #         nn.Conv2d(dims[-1], 1280, 1),
        #         nn.ReLU6(),
        #         LayerNorm(1280, eps=1e-6, data_format="channels_first")
        # )
        # self.pred = nn.Sequential(
        #     # nn.Linear(dims[-1], 1280),
        #     # nn.GELU(),
        #     # LayerNorm(1280, eps=1e-6, data_format="channels_first")
        #     nn.Linear(dims[-1], num_classes)
        # )

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    trunc_normal_(m.weight, std=.02)
                    # nn.init.constant_(m.bias, 0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, LayerNorm)):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        else:
            super(MixNet, self).init_weights()



    def forward_features(self, x):
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # x = self.head(x)
        return x.mean([-2, -1])  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        outs = []
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                outs.append(x)
        # x = self.forward_features(x)
        # x = self.pred(x)
        # if self.drop_rate > 0.:
        #     x = F.dropout(x, p=self.drop_rate, training=self.training)
        return outs


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x




# model_urls = {
#     "mixnet_t": "/root/andrewhoux/Projects/van/output/train/20220616-220244-mixnet_t-224/model_best.pth.tar",
# }
#
#
# def load_model_weights(model, arch, kwargs):
#     # url = pretrained #model_urls[arch]
#     # checkpoint = torch.hub.load_state_dict_from_url(
#     #     url=url, map_location="cpu", check_hash=True
#     # )
#     checkpoint = torch.load(model_urls[arch], map_location="cpu")
#     strict = True
#     if "num_classes" in kwargs and kwargs["num_classes"] != 1000:
#         strict = False
#         del checkpoint["state_dict"]["head.weight"]
#         del checkpoint["state_dict"]["head.bias"]
#     model.load_state_dict(checkpoint["state_dict"], strict=strict)
#     return model
#
#
# @register_model
# def eunet_t(pretrained=False, **kwargs):  # 81.5
#     print(kwargs)
#     model = MixNet(dims=[24, 48, 96, 192], mlp_ratios=[4, 4, 4, 4], depths=[3, 3, 9, 3], strides=[2, 2, 2, 1], **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         model = load_model_weights(model, 'mixnet_t', kwargs)
#     return model
