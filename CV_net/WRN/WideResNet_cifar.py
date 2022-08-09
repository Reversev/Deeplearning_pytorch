#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2022/2/5 19:10
# @Author : 'IReverser'
# @FileName: model.py
import torch
import math
import torch.nn.functional as F
import torch.nn as nn

__all__ = ["wrn_cifar"]


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_channels == out_channels)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class WideResNet(nn.Module):
    def __init__(self, BasicBlock, num_classes, depth, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        self.num_blocks = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'Depth of the model should be 6n + 4'
        n = (depth - 4) // 6
        block = BasicBlock

        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, self.num_blocks[0], kernel_size=3, stride=1, padding=1, bias=False)

        # 1st block
        self.block1 = self._make_layer(block, self.num_blocks[0], self.num_blocks[1], n, 1, dropRate)
        # 2nd block
        self.block2 = self._make_layer(block, self.num_blocks[1], self.num_blocks[2], n, 2, dropRate)
        # 3rd block
        self.block3 = self._make_layer(block, self.num_blocks[2], self.num_blocks[3], n, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(self.num_blocks[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(self.num_blocks[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = nn.ModuleList()
        for i in range(nb_layers):
            # i == 0 and in_planes or out_planes: if i==0 in_planes else out_planes
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        x = self.conv1(x)             # 16 x 32 x 32
        x = self.block1(x)            # 160 x 32 x 32
        x = self.block2(x)            # 320 x 16 x 16
        x = self.block3(x)            # 640 x 8 x 8
        x = self.relu(self.bn1(x))
        x = F.avg_pool2d(x, 8)        # [640, 8, 8] -> [640 x 8 x 8, 1]
        fea = x.view(-1, self.num_blocks[3])  # [640 x 8 x 8, 1] -> [1, 64 * widen_factor]
        x = self.fc(fea)                  # [1, 64 * widen_factor] -> [64 * widen_factor, num_classes]
        if is_feat:
            return fea, x
        else:
            return x


def wrn_cifar(**kwargs):
    """
    Constructs a Wide Residual Networks in CIFAR10/100.
    """
    model = WideResNet(BasicBlock, **kwargs)
    return model


def wrn_28_10(num_classes=10):
    return WideResNet(BasicBlock,
                      num_classes=num_classes,
                      depth=28,
                      widen_factor=10, dropRate=0.3)


def wrn_40_2(num_classes=10):
    return WideResNet(BasicBlock,
                      num_classes=num_classes,
                      depth=40,
                      widen_factor=2, dropRate=0.0)


def wrn_50_2(num_classes=10):
    return WideResNet(BasicBlock,
                      num_classes=num_classes,
                      depth=50,
                      widen_factor=2, dropRate=0.0)


def wrn_101_2(num_classes=10):
    return WideResNet(BasicBlock,
                      num_classes=num_classes,
                      depth=101,
                      widen_factor=2, dropRate=0.0)


# x = torch.randn((1, 3, 32, 32))
# net = wrn_40_2(num_classes=10)    # Total params: 2,243,546 (2.24M)
# print(net(x, is_feat=False).shape)
# from torchsummary import summary
# net = net.to('cuda')
# summary(net, (3, 32, 32))
