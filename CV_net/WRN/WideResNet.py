import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    Ref: https://github.com/fshp971/RPG/blob/d3dd7249a3daaa6a229014adbb07ee1779ad6b4e/models/resnet.py
    Wide ResNet for Imagenet in paper: (WRN-50-2, WRN-34-1, WRN-18-1)
    https://arxiv.org/pdf/1605.07146.pdf
"""


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, wide=1):
        super(BasicBlock, self).__init__()
        planes = int(planes * wide)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, wide=1):
        super(Bottleneck, self).__init__()
        mid_planes = planes * wide
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, wide=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.wide = wide

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.wide))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


class WRN(nn.Module):
    def __init__(self, num_blocks, num_classes=1000, wide=1):
        super(WRN, self).__init__()
        self.in_planes = 16
        self.wide = wide

        block = BasicBlock

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64 * wide, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.wide))
            self.in_planes = planes * self.wide * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def WRN101_2(num_classes=1000):    # params: 126,879,016B
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, wide=2)


def WRN50_2(num_classes=1000):    # params: 68,875,560B
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, wide=2)


def WRN34_1(num_classes=1000):    # params: 21,789,992B
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, wide=1)


def WRN18_1(num_classes=1000):    # params: 11,681,832B
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, wide=1)


def WRN34_10(num_classes=1000):    # params: 2,168,056B
    return WRN([3, 4, 6], num_classes=num_classes, wide=2)


# x = torch.randn((1, 3, 224, 224))
# # net = WRN50_2(num_classes=1000)
# net = WRN34_1(num_classes=1000)
# print(net(x).shape)
# from torchsummary import summary
# net = net.to('cpu')
# summary(net, (3, 224, 224), device='cpu')
