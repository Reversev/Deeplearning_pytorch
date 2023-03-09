# ！/usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2023/3/2 13:31
# @Author: 'IReverser'
# @ FileName: efficientnetv1.py
import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def _make_divisible(v, divisor: Optional[int] = 8, min_value=None):
    """ Optional[X] == Union[X, None]
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_value < 0.9 * v:
        new_value += divisor
    return new_value


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
    output = x.div(keep_prob) * random_tensor
    return output


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
                 out_planes,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 act_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if act_layer is None:
            act_layer = nn.SiLU    # alias Swish  (torch>=1.7)

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
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_channels = in_channels // squeeze_factor
        self.fc1 = nn.Conv2d(expand_channels, squeeze_channels, 1)
        self.act1 = nn.SiLU()
        self.fc2 = nn.Conv2d(squeeze_channels, expand_channels, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.act1(self.fc1(scale))
        scale = self.act2(self.fc2(scale))
        return scale * x


class InvertedResidualConfig:
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    def __init__(self,
                 kernel_size: int,  # 3 or 5
                 in_channels: int,
                 out_channels: int,
                 exp_ratio: int,   # 1 or 6
                 stride: int,      # 1 or 2
                 use_se: bool,     # True
                 drop_rate: float,
                 index: str,       # 1a, 2a, 2b, ...
                 width_coefficient: float):
        self.in_channels = self.adjust_channels(in_channels, width_coefficient)
        self.kernel_size = kernel_size
        self.exp_channels = self.in_channels * exp_ratio
        self.out_channels = self.adjust_channels(out_channels, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.in_channels == cnf.out_channels)

        layers = OrderedDict()
        act_layer = nn.SiLU

        # expand
        if cnf.exp_channels != cnf.in_channels:
            layers.update({"expand_conv": ConvBnAct(cnf.in_channels,
                                                    cnf.exp_channels,
                                                    kernel_size=1,
                                                    norm_layer=norm_layer,
                                                    act_layer=act_layer)})

        # depthwise
        layers.update({"dwconv": ConvBnAct(cnf.exp_channels,
                                           cnf.exp_channels,
                                           kernel_size=cnf.kernel_size,
                                           stride=cnf.stride,
                                           groups=cnf.exp_channels,
                                           norm_layer=norm_layer,
                                           act_layer=act_layer)})

        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.in_channels, cnf.exp_channels)})  # squeeze from in_channnels

        # Point-wise linear projection
        layers.update({"project_conv": ConvBnAct(cnf.exp_channels,
                                                 cnf.out_channels,
                                                 kernel_size=1,
                                                 norm_layer=norm_layer,
                                                 act_layer=nn.Identity)})

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_channels
        self.is_strided = cnf.stride > 1

        # while use shortcut, use dropout layer
        if self.use_res_connect and cnf.drop_rate > 0.0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        res = self.block(x)
        res = self.dropout(res)
        if self.use_res_connect:
            res += x
        return res


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 1000,
                 drop_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(EfficientNet, self).__init__()

        # kernel_size, in_channels, out_channels, exp_ratio, strides, use_se, drop_connect_rate, repeats
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def _round_repeat(repeats):
            """Round number of repeats based on depth multiplier"""
            return int(math.ceil(depth_coefficient * repeats))

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(InvertedResidualConfig.adjust_channels,
                                  width_coefficient=width_coefficient)

        # build inverted residual setting
        bneck_conf = partial(InvertedResidualConfig, width_coefficient=width_coefficient)

        b = 0
        num_blocks = float(sum(_round_repeat(i[-1]) for i in default_cnf))
        inverted_residual_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(_round_repeat(cnf.pop(-1))):
                if i > 0:
                    # strides equal 1 except first cnf
                    cnf[-3] = 1  # strides
                    cnf[1] = cnf[2]  # input_channel equal output_channel
                cnf[-1] = args[-2] * b / num_blocks  # update dropout ratio
                index = str(stage + 1) + chr(i + 97)  # 1a, 2a, 2b, ... (a: 97, b: 98, ...)
                inverted_residual_setting.append(bneck_conf(*cnf, index))
                b += 1

        # create layers
        layers = OrderedDict()

        # first conv
        layers.update({"first_conv": ConvBnAct(in_planes=3,
                                               out_planes=adjust_channels(32),
                                               kernel_size=3,
                                               stride=2,
                                               norm_layer=norm_layer)})

        # build inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # build top
        last_conv_in_channels = inverted_residual_setting[-1].out_channels
        last_conv_out_channels = adjust_channels(1280)
        layers.update({"top": ConvBnAct(in_planes=last_conv_in_channels,
                                        out_planes=last_conv_out_channels,
                                        kernel_size=1,
                                        norm_layer=norm_layer)})

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        if drop_rate > 0:
            classifier.append(nn.Dropout(p=drop_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_out_channels, num_classes))
        self.classifier = nn.Sequential(*classifier)
        self._init_weights()

    # initial weights
    def _init_weights(self):
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
        x = self.features(x)
        x = self.avgpool(x)
        feas = torch.flatten(x, 1)
        x = self.classifier(feas)
        if is_feat:
            return feas, x
        else:
            return x


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        drop_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b1(num_classes=1000):
    # input image size 240x240
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        drop_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    # input image size 260x260
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        drop_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    # input image size 300x300
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        drop_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    # input image size 380x380
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        drop_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    # input image size 456x456
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        drop_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    # input image size 528x528
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        drop_rate=0.5,
                        num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    # input image size 600x600
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        drop_rate=0.5,
                        num_classes=num_classes)


model_dict = {"effnetv1_b0": efficientnet_b0,
              "effnetv1_b1": efficientnet_b1,
              "effnetv1_b2": efficientnet_b2,
              "effnetv1_b3": efficientnet_b3,
              "effnetv1_b4": efficientnet_b4,
              "effnetv1_b5": efficientnet_b5,
              "effnetv1_b6": efficientnet_b6,
              "effnetv1_b7": efficientnet_b7}

input_size_dict = {"effnetv1_b0": 224,
                   "effnetv1_b1": 240,
                   "effnetv1_b2": 260,
                   "effnetv1_b3": 300,
                   "effnetv1_b4": 380,
                   "effnetv1_b5": 456,
                   "effnetv1_b6": 528,
                   "effnetv1_b7": 600}


def create_efficientnetv1(model_name, num_classes: int = 1000):
    return model_dict[model_name](num_classes=num_classes)


if __name__ == '__main__':
    model_name = "effnetv1_b1"
    net = model_dict[model_name](num_classes=100)
    input_size = input_size_dict[model_name]
    print(input_size)
    x = torch.rand((1, 3, input_size, input_size))
    print(net(x).shape)
