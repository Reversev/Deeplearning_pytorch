# ！/usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2023/3/2 13:32
# @Author: 'IReverser'
# @ FileName: efficientnetv2.py
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch.nn as nn
import torch
from torch import Tensor
from torch.nn import functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    out = x.div(keep_prob) * random_tensor
    return out


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBnAct(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 act_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        super(ConvBnAct, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                  out_channels=out_planes,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  groups=groups,
                                                  bias=False),
                                        norm_layer(out_planes),
                                        act_layer())


class SqueezeExcitation(nn.Module):
    def __init__(self,
                 in_channels: int,  # block input channel
                 expand_channels: int,  # block expand channel
                 squeeze_ratio: float = 0.25):
        super(SqueezeExcitation, self).__init__()
        squeeze_channels = int(in_channels * squeeze_ratio)
        self.fc1 = nn.Conv2d(expand_channels, squeeze_channels, 1)  # reduce
        self.act1 = nn.SiLU()
        self.fc2 = nn.Conv2d(squeeze_channels, expand_channels, 1)  # expand
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.act1(self.fc1(scale))
        scale = self.act2(self.fc2(scale))
        return scale * x


class MBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        act_layer = nn.SiLU
        expanded_channels = in_channels * expand_ratio

        # must use expansion for MBconv in efficientv2, that is, expansion != 1
        assert expand_ratio != 1
        # Point-wise expansion
        self.expand_conv = ConvBnAct(in_channels,
                                     expanded_channels,
                                     kernel_size=1,
                                     norm_layer=norm_layer,
                                     act_layer=act_layer)

        # Depth-wise conv
        self.dwconv = ConvBnAct(expanded_channels,
                                expanded_channels,
                                kernel_size=kernel_size,
                                groups=expanded_channels,
                                norm_layer=norm_layer,
                                act_layer=act_layer)

        self.se = SqueezeExcitation(in_channels, expanded_channels, se_ratio) if se_ratio > 0 else nn.Identity()
        # Point-wise linear projection
        self.project_conv = ConvBnAct(expanded_channels,
                                      out_channels,
                                      kernel_size=1,
                                      norm_layer=norm_layer,
                                      act_layer=nn.Identity)

        self.out_channels = out_channels
        # while use shortcut, use dropout layer
        self.drop_rate = drop_rate
        if self.use_res_connect and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        res = self.expand_conv(x)
        res = self.dwconv(res)
        res = self.se(res)
        res = self.project_conv(res)

        if self.use_res_connect:
            if self.drop_rate > 0:
                res = self.dropout(res)
            res += x
        return res


class FusedMBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(FusedMBConv, self).__init__()

        assert stride in [1, 2]
        assert se_ratio == 0

        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        self.drop_rate = drop_rate

        self.has_expansion = expand_ratio != 1

        act_layer = nn.SiLU
        expanded_channels = in_channels * expand_ratio

        # while expand_ratio not equal 1, use expand conv
        if self.has_expansion:
            # Expansion conv
            self.expand_conv = ConvBnAct(in_channels,
                                         expanded_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         act_layer=act_layer)

            self.project_conv = ConvBnAct(expanded_channels,
                                          out_channels,
                                          kernel_size=1,
                                          norm_layer=norm_layer,
                                          act_layer=nn.Identity)  # not use act_layer

        else:
            # only use project conv
            self.project_conv = ConvBnAct(in_channels,
                                          out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          norm_layer=norm_layer,
                                          act_layer=act_layer)  # use act_layer

        self.out_channels = out_channels
        # while use shortcut, use dropout layer
        self.drop_rate = drop_rate
        if self.use_res_connect and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        if self.has_expansion:
            res = self.expand_conv(x)
            res = self.project_conv(res)
        else:
            res = self.project_conv(x)

        if self.use_res_connect:
            if self.drop_rate > 0:
                res = self.dropout(res)
            res += x
        return res


class EfficientNetv2(nn.Module):
    def __init__(self,
                 model_cfg: list,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 drop_rate: float = 0.2,
                 drop_connect_rate: float = 0.2):
        super(EfficientNetv2, self).__init__()
        self.drop_rate = drop_rate

        for cfg in model_cfg:
            assert len(cfg) == 8

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        stem_filter_num = model_cfg[0][4]

        self.stem = ConvBnAct(3,
                              stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)

        total_blocks = sum([i[0] for i in model_cfg])
        block_id = 0
        blocks = []
        for cfg in model_cfg:
            repeats = cfg[0]
            op = FusedMBConv if cfg[-2] == 0 else MBConv
            for i in range(repeats):
                blocks.append(op(kernel_size=cfg[1],
                                 in_channels=cfg[4] if i == 0 else cfg[5],
                                 out_channels=cfg[5],
                                 expand_ratio=cfg[3],
                                 stride=cfg[2] if i == 0 else 1,
                                 se_ratio=cfg[-1],
                                 drop_rate=drop_connect_rate * block_id / total_blocks,
                                 norm_layer=norm_layer))
                block_id += 1

        self.blocks = nn.Sequential(*blocks)
        # head
        self.project_conv = ConvBnAct(model_cfg[-1][-3],
                                      num_features,
                                      kernel_size=1,
                                      norm_layer=norm_layer)  # SiLU

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        classifier = []
        if drop_rate > 0:
            classifier.append(nn.Dropout(p=drop_rate, inplace=True))
        classifier.append(nn.Linear(num_features, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, is_feat=False):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.project_conv(x)
        feas = self.flatten(self.avgpool(x))
        x = self.classifier(feas)
        if is_feat:
            return feas, x
        else:
            return x


# cfg = [repeats, kernel_size, stride, expand_ratio, in_channels, out_channels, 0/1, se_ratio]  0: FusedMBConv 1:MBConv
def efficientnetv2_s(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 300, eval_size: 384

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25]]

    model = EfficientNetv2(model_cfg=model_config,
                           num_classes=num_classes,
                           drop_rate=0.2)
    return model


def efficientnetv2_m(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    model = EfficientNetv2(model_cfg=model_config,
                           num_classes=num_classes,
                           drop_rate=0.3)
    return model


def efficientnetv2_l(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    model = EfficientNetv2(model_cfg=model_config,
                           num_classes=num_classes,
                           drop_rate=0.4)
    return model


model_dict = {"effnetv2_s": efficientnetv2_s,
              "effnetv2_m": efficientnetv2_m,
              "effnetv2_l": efficientnetv2_l}

input_size_dict = {"effnetv2_s": "300_384",
                   "effnetv2_m": "384_480",
                   "effnetv2_l": "384_480"}


def create_efficientnetv2(model_name, num_classes: int = 1000):
    return model_dict[model_name](num_classes=num_classes)


if __name__ == '__main__':
    model_name = "effnetv2_s"
    net = model_dict[model_name](num_classes=100)
    input_size = int(input_size_dict[model_name].split("_")[0])
    print(input_size)
    x = torch.rand((1, 3, input_size, input_size))
    print(net(x).shape)
