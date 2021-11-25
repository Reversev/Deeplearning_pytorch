#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2021/8/25 16:14
# @Author : 'IReverser'
# @FileName: model.py
import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, n_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, n_classes)
        )
        if init_weights:
            self._initalize_weights()

    def forward(self, x):
        # BATCH_SIZE * 3 * 224 * 224
        x = self.features(x)
        # print(x.shape)
        # BATCH_SIZE * 512 * 7 * 7
        x = torch.flatten(x, start_dim=1)
        # BATCH_SIZE * 512 * 7 * 7
        x = self.classifier(x)
        return x

    def _initalize_weights(self):
        for m in self.modules():
            if isinstance(nn, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_feature_layer(cfg: list, flag=0):
    layers = []
    in_channels = 3
    for ch in cfg:
        if ch == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            if flag == 1 and in_channels == 3:
                layers += [nn.LocalResponseNorm(size=2, alpha=0.0001, beta=0.75, k=1.0)],
            in_channels = ch
    return nn.Sequential(*layers)


cfgs = {
    "vgg11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "vgg11L": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "vgg13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "vgg16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "vgg19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def vgg(model_name='vgg16', n_classes=1000, **kwargs):
    assert model_name in cfgs, "Warning: model {} is not in cfgs dictionary!".format(model_name)
    cfg = cfgs[model_name]

    flag = 0
    if model_name[-1] == 'L':
        flag = 1
    model = VGG(make_feature_layer(cfg, flag), n_classes, **kwargs)

    return model


# x = torch.randn((1, 3, 224, 224))
# net = vgg(model_name='vgg16', n_classes=1000, init_weights=True)
# y = net(x)
# print(y.shape)
# # output parameters
# from torchsummary import summary
# summary(net, input_size=[[3, 224, 224]], batch_size=2, device='cpu')
