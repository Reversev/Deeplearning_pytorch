#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2022/2/5 19:10
# @Author : 'IReverser'
# @FileName: model.py
"""ref:  K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016."""
import math
import torch.nn as nn

__all__ = ['preact_resnet']


def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(x))

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))

        out += residual
        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes * 4, kernel_size=1, bias=False)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(x))
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))

        out += residual
        return out


class PreAct_ResNet_cifar(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_cifar, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self.make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt((2. / n)))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            )
        layers = nn.ModuleList()
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, ))
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        if is_feat:
            return x, out
        else:
            return out


def preact_resnet(**kwargs):
    model = PreAct_ResNet_cifar(**kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    model = PreAct_ResNet_cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    model = PreAct_ResNet_cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model


model_list = {
    '110': preact_resnet110_cifar,
    '164': preact_resnet164_cifar,
    '1001': preact_resnet1001_cifar,
}
# if __name__ == '__main__':
#     import torch
#     net = preact_resnet110_cifar()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())
#     from torchsummary import summary
#     net = net.to('cuda')
#     summary(net, (3, 32, 32))

# if __name__ == '__main__':
#     import torch
#     net = model_list.get('164')(num_classes=10)
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())
