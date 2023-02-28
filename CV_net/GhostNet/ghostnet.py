# ！/usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2022/12/16 14:40
# @Author: 'IReverser'
# @ FileName: mobilevit.py
# GhostNet
"""
    GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
    https://arxiv.org/abs/1911.11907
    https://github.com/iamhankai/ghostnet.pytorch
"""
import torch
import torch.nn as nn
import math


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


class SElayer(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SElayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction),
                                nn.ReLU(inplace=True),
                                nn.Linear(channels // reduction, channels))

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avgpool(x).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        out = torch.clamp(out, 0, 1)
        return x * out


def depthwise_conv(in_channels, out_channels, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=kernel_size // 2,
                  groups=in_channels, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True) if relu else nn.Identity(),
    )


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(in_channels, hidden_dim, kernel_size=1, relu=True),   # **
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride == 2 else nn.Identity(),
            # Squeeze and Excitation
            SElayer(hidden_dim) if use_se else nn.Identity(),
            # pw linear
            GhostModule(hidden_dim, out_channels, kernel_size=1, relu=False),  # **
        )

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(depthwise_conv(in_channels, in_channels, kernel_size, stride, relu=False),
                                          nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                                          nn.BatchNorm2d(out_channels),)

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.0):
        super(GhostNet, self).__init__()
        # set inverted residual blocks
        self.cfgs = cfgs

        # build first layer
        out_channels = _make_divisible(16 * width_mult, 4)
        layer = [nn.Sequential(nn.Conv2d(3, out_channels, 3, 2, 1, bias=False),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(inplace=True))]
        in_channels = out_channels

        # build inverted residual blocks
        for k, exp_size, c, use_se, s in self.cfgs:
            out_channels = _make_divisible(c * width_mult, 4)
            hidden_channels = _make_divisible(exp_size * width_mult, 4)
            layer.append(GhostBottleneck(in_channels, hidden_channels, out_channels, k, s, use_se))
            in_channels = out_channels
        self.features = nn.Sequential(*layer)

        # build last several layers
        out_channels = _make_divisible(exp_size * width_mult, 4)
        self.squeeze = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True),
                                     nn.AdaptiveAvgPool2d((1, 1)),)

        in_channels = out_channels

        out_channels = 1280
        self.classifier = nn.Sequential(nn.Linear(in_channels, out_channels, bias=False),
                                        nn.BatchNorm1d(out_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.2),
                                        nn.Linear(out_channels, num_classes))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


cfgs = [
    # k, t, c, SE, s
    [3,  16,  16, 0, 1],  # stage1
    [3,  48,  24, 0, 2],  # stage2
    [3,  72,  24, 0, 1],
    [5,  72,  40, 1, 2],  # stage3
    [5, 120,  40, 1, 1],
    [3, 240,  80, 0, 2],  # stage4
    [3, 200,  80, 0, 1],
    [3, 184,  80, 0, 1],
    [3, 184,  80, 0, 1],
    [3, 480, 112, 1, 1],
    [3, 672, 112, 1, 1],
    [5, 672, 160, 1, 2],  # stage5
    [5, 960, 160, 0, 1],
    [5, 960, 160, 1, 1],
    [5, 960, 160, 0, 1],
    [5, 960, 160, 1, 1]
]


def create_ghostnet(num_classes: int = 1000, width_mult=1.):
    return GhostNet(cfgs, num_classes=num_classes, width_mult=width_mult)


if __name__ == '__main__':
    from torchsummary import summary
    x = torch.rand((1, 3, 224, 224))
    net = create_ghostnet()
    net.eval()
    print(net(x).shape)
    summary(net, (3, 224, 224))
