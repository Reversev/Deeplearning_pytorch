# ！/usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2022/12/16 14:40
# @Author: 'IReverser'
# @ FileName: regnet.py
# RegNet
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def _make_divisible(channel, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8. (8x)
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(channel + divisor / 2) // divisor * divisor)
    # make sure that round down does not go down by more than 10%
    if new_ch < 0.9 * channel:
        new_ch += divisor
    return new_ch


def _mcfg(**kwargs):
    cfg = dict(se_ratio=0., bottle_ratio=1., stem_width=32)
    cfg.update(**kwargs)
    return cfg


model_cfgs = {
    "regnetx_200mf": _mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13),
    "regnetx_400mf": _mcfg(w0=24, wa=24.48, wm=2.54, group_w=16, depth=22),
    "regnetx_600mf": _mcfg(w0=48, wa=36.97, wm=2.24, group_w=24, depth=16),
    "regnetx_800mf": _mcfg(w0=56, wa=35.73, wm=2.28, group_w=16, depth=16),

    "regnetx_1.6gf": _mcfg(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18),
    "regnetx_3.2gf": _mcfg(w0=88, wa=26.31, wm=2.25, group_w=48, depth=25),
    "regnetx_4.0gf": _mcfg(w0=96, wa=38.65, wm=2.43, group_w=40, depth=23),
    "regnetx_6.4gf": _mcfg(w0=184, wa=60.83, wm=2.07, group_w=56, depth=17),
    "regnetx_8.0gf": _mcfg(w0=80, wa=49.56, wm=2.88, group_w=120, depth=23),

    "regnetx_12gf": _mcfg(w0=80, wa=34.01, wm=2.25, group_w=24, depth=19),
    "regnetx_16gf": _mcfg(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18),
    "regnetx_32gf": _mcfg(w0=80, wa=34.01, wm=2.25, group_w=24, depth=18),

    "regnety_200mf": _mcfg(w0=24, wa=36.44, wm=2.49, group_w=8, depth=13, se_ratio=0.25),
    "regnety_400mf": _mcfg(w0=48, wa=27.89, wm=2.09, group_w=8, depth=16, se_ratio=0.25),
    "regnety_600mf": _mcfg(w0=48, wa=32.54, wm=2.32, group_w=16, depth=15, se_ratio=0.25),
    "regnety_800mf": _mcfg(w0=56, wa=38.84, wm=2.4, group_w=16, depth=14, se_ratio=0.25),

    "regnety_1.6gf": _mcfg(w0=48, wa=20.71, wm=2.65, group_w=24, depth=27, se_ratio=0.25),
    "regnety_3.2gf": _mcfg(w0=80, wa=42.63, wm=2.66, group_w=24, depth=21, se_ratio=0.25),
    "regnety_4.0gf": _mcfg(w0=96, wa=31.41, wm=2.24, group_w=64, depth=22, se_ratio=0.25),
    "regnety_6.4gf": _mcfg(w0=112, wa=33.22, wm=2.27, group_w=72, depth=25, se_ratio=0.25),
    "regnety_8.0gf": _mcfg(w0=192, wa=76.82, wm=2.19, group_w=56, depth=17, se_ratio=0.25),

    "regnety_12gf": _mcfg(w0=168, wa=73.36, wm=2.37, group_w=112, depth=19, se_ratio=0.25),
    "regnety_16gf": _mcfg(w0=200, wa=106.23, wm=2.48, group_w=112, depth=18, se_ratio=0.25),
    "regnety_32gf": _mcfg(w0=232, wa=115.89, wm=2.53, group_w=232, depth=20, se_ratio=0.25),
}


def generate_width_depth(w0, wa, wm, depth, q=8):
    """generates per black widths from RegNet parameters."""
    assert w0 > 0 and wa > 0 and wm > 1 and w0 % q == 0
    width_cout = np.arange(depth) * wa + w0
    width_exps = np.round(np.log(width_cout / w0) / np.log(wm))
    widths_j = w0 * np.power(wm, width_exps)
    widths_j = np.round(np.divide(widths_j, q)) * q
    num_stages, max_stage = len(np.unique(widths_j)), width_exps.max() + 1
    assert num_stages == int(max_stage)
    assert num_stages == 4
    widths = widths_j.astype(int).tolist()
    return widths, num_stages


def adjust_width_group_comp(widths: list, groups: list):
    """adjust the compatibility of widths and groups."""
    groups = [min(g, w_bot) for g, w_bot in zip(groups, widths)]
    # Adjust w to an integral multiple of g
    widths = [int(round(w / g) * g) for w, g in zip(widths, groups)]
    return widths, groups


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_s: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 groups: int = 1,
                 act: Optional[nn.Module] = nn.ReLU(inplace=True)):
        super(ConvBNAct, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channel,
                              out_channels=out_channel,
                              kernel_size=kernel_s,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = nn.BatchNorm2d(out_channel)
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.bn(self.conv(x))
        x = self.act(x)
        return x


class RegHead(nn.Module):
    def __init__(self,
                 in_unit: int = 368,
                 out_unit: int = 1000,
                 output_size: tuple = (1, 1),
                 drop_ratio: float = 0.25):
        super(RegHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

        if drop_ratio > 0:
            self.dropout = nn.Dropout(p=drop_ratio)
        else:
            self.dropout = nn.Identity()

        self.fc = nn.Linear(in_features=in_unit, out_features=out_unit)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, input_ch: int, expand_ch: int, se_ratio: float = 0.25):
        super(SqueezeExcitation, self).__init__()
        squeeze_ch = int(input_ch * se_ratio)
        self.fc1 = nn.Conv2d(expand_ch, squeeze_ch, 1)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_ch, expand_ch, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.act1(self.fc1(scale))
        scale = self.act2(self.fc2(scale))
        return scale * x


class Bottleneck(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 stride: int = 1,
                 group_width: int = 1,
                 se_ratio: float = 0.,
                 drop_ratio: float = 0.):
        super(Bottleneck, self).__init__()

        self.conv1 = ConvBNAct(in_channel=in_channel, out_channel=out_channel, kernel_s=1)
        self.conv2 = ConvBNAct(in_channel=out_channel,
                               out_channel=out_channel,
                               kernel_s=3,
                               stride=stride,
                               padding=1,
                               groups=out_channel // group_width)

        if se_ratio > 0:
            self.se = SqueezeExcitation(in_channel, out_channel, se_ratio)
        else:
            self.se = nn.Identity()

        self.conv3 = ConvBNAct(in_channel=out_channel, out_channel=out_channel, kernel_s=1, act=None)
        self.act3 = nn.ReLU(inplace=True)

        if drop_ratio > 0:
            self.dropout = nn.Dropout(p=drop_ratio)
        else:
            self.dropout = nn.Identity()

        if (in_channel != out_channel) or (stride != 1):
            self.downsample = ConvBNAct(in_channel=in_channel, out_channel=out_channel, kernel_s=1, stride=stride, act=None)
        else:
            self.downsample = nn.Identity()

    def zero_init_last_bn(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.se(x)
        x = self.conv3(x)

        x = self.dropout(x)

        shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)
        return x


class RegStage(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 depth: int,
                 group_width: int,
                 se_ratio: float):
        super(RegStage, self).__init__()
        for i in range(depth):
            block_stride = 2 if i == 0 else 1
            block_in_channel = in_channel if i == 0 else out_channel

            name = "b{}".format(i + 1)
            self.add_module(name,
                            Bottleneck(in_channel=block_in_channel,
                                       out_channel=out_channel,
                                       stride=block_stride,
                                       group_width=group_width,
                                       se_ratio=se_ratio))

    def forward(self, x: Tensor) -> Tensor:
        for block in self.children():
            x = block(x)
        return x


class RegNet(nn.Module):
    """
    RegNet model.
    Paper: https://arxiv.org/abs/2003.13678
    Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
    and refer to: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/regnet.py
    """
    def __init__(self,
                 cfg: dict,
                 in_channel: int = 3,
                 num_classes: int = 1000,
                 zero_init_last_bn: bool = True):
        super(RegNet, self).__init__()

        # RegStem
        stem_channel = cfg["stem_width"]
        self.stem = ConvBNAct(in_channel, stem_channel, kernel_s=3, stride=2, padding=1)

        # build stages
        input_channel = stem_channel
        stage_info = self._build_stage_info(cfg)
        for i, stage_args in enumerate(stage_info):
            stage_name = "s{}".format(i + 1)
            self.add_module(stage_name, RegStage(in_channel=input_channel, **stage_args))
            input_channel = stage_args["out_channel"]

        # RegHead
        self.head = RegHead(in_unit=input_channel, out_unit=num_classes)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

        if zero_init_last_bn:
            for m in self.modules():
                if hasattr(m, "zero_init_last_bn"):
                    m.zero_init_last_bn()

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.children():
            x = layer(x)
            # print(x.shape)
        return x

    @staticmethod
    def _build_stage_info(cfg: dict):
        w0, wa, wm, d = cfg["w0"], cfg["wa"], cfg["wm"], cfg["depth"]
        widths, num_stages = generate_width_depth(w0, wa, wm, d)

        stage_widths, stage_depths = np.unique(widths, return_counts=True)
        stage_groups = [cfg["group_w"] for _ in range(num_stages)]
        stage_widths, stage_groups = adjust_width_group_comp(stage_widths, stage_groups)

        info = []
        for i in range(num_stages):
            info.append(dict(out_channel=stage_widths[i],
                             depth=stage_depths[i],
                             group_width=stage_groups[i],
                             se_ratio=cfg["se_ratio"]))
        return info


def create_regnet(model_name="RegNetx_200mf", num_classes=1000):
    model_name = model_name.lower().replace("-", "_")
    if model_name not in model_cfgs.keys():
        print("support model name: \n{}".format("\n".join(model_cfgs.keys())))
        raise KeyError("not support model name: {}".format(model_name))

    model = RegNet(cfg=model_cfgs[model_name], num_classes=num_classes)
    return model


if __name__ == '__main__':
    from torchsummary import summary
    x = torch.rand((1, 3, 224, 224))
    net = create_regnet(model_name="RegNetx_200mf")
    print(net(x).shape)
    summary(net, (3, 224, 224))
