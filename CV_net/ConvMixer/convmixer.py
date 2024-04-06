#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2024/3/24 10:27
# @Author : 'IReverser'
# @FileName: convmixer.py
# Reference: https://github.com/locuslab/convmixer
import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# def ConvMixer(dim, depth, kernel_size=9, patch_size=7, num_classes=1000):
#     layers = nn.Sequential()
#
#     # patch embedding
#     patch_embedding = nn.Sequential(
#         nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
#         nn.GELU(),
#         nn.BatchNorm2d(dim),
#     )
#     layers.append(patch_embedding)
#
#     # DW Conv and point-wise Conv as backbone
#     backbone = nn.Sequential(
#         *[nn.Sequential(
#             Residual(
#                 nn.Sequential(
#                     nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
#                     nn.GELU(),
#                     nn.BatchNorm2d(dim), )),
#             nn.Conv2d(dim, dim, kernel_size=1),
#             nn.GELU(),
#             nn.BatchNorm2d(dim)) for _ in range(depth)])
#     layers.append(backbone)
#
#     # fc
#     fc = nn.Sequential(
#         nn.AdaptiveAvgPool2d((1, 1)),
#         nn.Flatten(),
#         nn.Linear(dim, num_classes)
#     )
#     layers.append(fc)
#
#     return layers


class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, num_classes=1000):
        super(ConvMixer, self).__init__()
        # patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        # DW Conv and point-wise Conv as backbone
        self.backbone = nn.Sequential(
            *[nn.Sequential(
                Residual(
                    nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(dim), )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)) for _ in range(depth)])
        # fc
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, num_classes))

    def forward(self, x):
        out = self.patch_embedding(x)
        out = self.backbone(out)
        out = self.fc(out)
        return out


model_dict = {
    'ConvMixer-512_12': ConvMixer(512, 12, kernel_size=8, patch_size=7),  # 4.2M
    'ConvMixer-512_16': ConvMixer(512, 16, kernel_size=8, patch_size=7),  # 5.4M
    'ConvMixer-1024_12': ConvMixer(1024, 12, kernel_size=8, patch_size=7),  # 14.6M
    'ConvMixer-1024_16': ConvMixer(1024, 16, kernel_size=9, patch_size=7),  # 19.4M
    'ConvMixer-768_32': ConvMixer(768, 32, kernel_size=3, patch_size=14),  # 20.2M
    'ConvMixer-768_32_7': ConvMixer(768, 32, kernel_size=7, patch_size=7),  # 21.1M
    'ConvMixer-1024_20': ConvMixer(1024, 20, kernel_size=9, patch_size=14),  # 24.4M
    'ConvMixer-1536_20_3': ConvMixer(1536, 20, kernel_size=3, patch_size=7),  # 49.4M
    'ConvMixer-1536_20_9': ConvMixer(1536, 20, kernel_size=9, patch_size=7),  # 51.6M
    'ConvMixer-1536_20_9_7': ConvMixer(1536, 20, kernel_size=9, patch_size=14),  # 52.3M
}


def create_convmixer(model_name: str, num_classes: int = 1000):
    model = model_dict[model_name]
    if num_classes != 1000:
        model.fc[-1] = nn.Linear(model.fc[-1].in_features, num_classes)
    else:
        pass
    return model


if __name__ == '__main__':
    # x = torch.randn((1, 3, 224, 224))
    # model = ConvMixer(512, 12, kernel_size=8, patch_size=7, num_classes=5)
    # print(model(x).shape)

    from torchsummary import summary

    x = torch.rand((2, 3, 224, 224))
    model = create_convmixer(model_name="ConvMixer-512/12")
    print(model(x).shape)
    summary(create_convmixer(model_name="ConvMixer-512/12", num_classes=5), (3, 224, 224), device='cpu')
