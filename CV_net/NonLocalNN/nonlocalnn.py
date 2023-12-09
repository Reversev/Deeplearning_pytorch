# !/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2023/7/13 12:29
# @Author : ''
# @FileName: nonlocalnn.py
# referred from https://github.com/tea1528/Non-Local-NN-Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode="embedded", dimension=3, bn_layer=True):
        super(NLBlockND, self).__init__()
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        assert dimension in [1, 2, 3]

        if mode not in ["gaussian", "embedded", "dot", "concatenate"]:
            raise ValueError("mode must be one of 'gaussian', 'embedded', 'dot' or 'concatenate'")

        self.mode = mode
        self.dimension = dimension
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size (inter_channels) is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimension
        if dimension == 3:
            conv = nn.Conv3d
            max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv = nn.Conv2d
            max_pool = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv = nn.Conv1d
            max_pool = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through convolution with kernel size 1
        self.g = conv(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add batch norm
        if bn_layer:
            self.W_z = nn.Sequential(
                conv(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels),
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)
            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        batch_size = x.shape[0]

        # (N, C, T*H*W)  C: inter_channels = in_channels // 2 if inter_channels is None else inter_channels
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # (N, C, T*H*W)
        g_x = g_x.permute(0, 2, 1)  # (N, T*H*W, C)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)  # (N, C, T*H*W)
            phi_x = x.view(batch_size, self.in_channels, -1)  # (N, C, T*H*W)
            theta_x = theta_x.permute(0, 2, 1)  # (N, T*H*W, C)
            f = torch.matmul(theta_x, phi_x)
        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # (N, C, T*H*W)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # (N, C, T*H*W)
            theta_x = theta_x.permute(0, 2, 1)  # (N, T*H*W, C)
            f = torch.matmul(theta_x, phi_x)
        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)  # (N, C, T*H*W, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)  # (N, C, 1, T*H*W)

            h = theta_x.size(2)  # T*H*W
            w = phi_x.size(3)  # T*H*W
            theta_x = theta_x.repeat(1, 1, 1, w)  # (N, C, T*H*W, T*H*W)
            phi_x = phi_x.repeat(1, 1, h, 1)  # (N, C, T*H*W, T*H*W)

            concat = torch.cat([theta_x, phi_x], dim=1)  # (N, C+C, T*H*W, T*H*W)
            f = self.W_f(concat)  # (N, C, T*H*W, T*H*W)
            f = f.view(f.size(0), f.size(2), f.size(3))  # (N, T*H*W, T*H*W)

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x   T*H*W
            f_div_C = f / N  # (N, T*H*W, T*H*W)

        y = torch.matmul(f_div_C, g_x)  # (N, T*H*W, C) = (N, T*H*W, T*H*W) * (N, T*H*W, C)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()  # (N, T*H*W, C) -> (N, C, T*H*W)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # (N, C, T*H*W)

        # reduce
        W_y = self.W_z(y)
        z = W_y + x

        return z


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet2D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, non_local=False):
        super(ResNet2D, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)

        # add non-local block after layer 2
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, non_local=non_local)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, non_local=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        last_idx = len(strides)
        if non_local:
            last_idx = len(strides) - 1

        for i in range(last_idx):
            layers.append(block(self.in_planes, planes, strides[i]))
            self.in_planes = planes * block.expansion

        if non_local:
            layers.append(NLBlockND(in_channels=planes, dimension=2))
            layers.append(block(self.in_planes, planes, strides[-1]))

        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        feat = out.view(out.size(0), -1)
        out = self.linear(feat)
        if is_feat:
            return feat, out
        else:
            return out


def resnet14(non_local=False, **kwargs):
    """Constructs a ResNet-8 model.
    """
    # 6 * N + 2
    N = 2
    return ResNet2D(BasicBlock, [N, N, N], non_local=non_local, **kwargs)


def resnet20(non_local=False, **kwargs):
    """Constructs a ResNet-8 model.
    """
    # 6 * N + 2
    N = 3
    return ResNet2D(BasicBlock, [N, N, N], non_local=non_local, **kwargs)


def resnet56(non_local=False, **kwargs):
    """Constructs a ResNet-56 model.
    """
    N = 9
    return ResNet2D(BasicBlock, [N, N, N], non_local=non_local, **kwargs)


model_list = {
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet56': resnet56}


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    net = model_list.get('resnet14')(num_classes=10, non_local=True)
    fea, x = net(x, is_feat=True)
    print(fea.shape, x.shape)
    from torchsummary import summary

    summary(net.cuda(), (3, 32, 32))

    # for bn_layer in [True, False]:
    #     img = torch.zeros(2, 3, 20)
    #     net = NLBlockND(in_channels=3, mode='concatenate', dimension=1, bn_layer=bn_layer)
    #     out = net(img)
    #     print(out.size())
    #
    #     img = torch.zeros(2, 3, 20, 20)
    #     net = NLBlockND(in_channels=3, mode='concatenate', dimension=2, bn_layer=bn_layer)
    #     out = net(img)
    #     print(out.size())
    #
    #     img = torch.randn(2, 3, 8, 20, 20)
    #     net = NLBlockND(in_channels=3, mode='concatenate', dimension=3, bn_layer=bn_layer)
    #     out = net(img)
    #     print(out.size())
