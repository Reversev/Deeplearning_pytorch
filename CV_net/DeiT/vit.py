# ！/usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2023/2/25 16:54
# @Author: 'IReverser'
# @ FileName: vit.py
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model


@register_model
def vit_tiny_patch16_224(pretrained=False, num_classes=1000, **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=192,
                              depth=12,
                              num_heads=3,
                              num_classes=num_classes,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # load offical pretrained model
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


def vit_small_patch16_224(pretrained=False, num_classes=1000, **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=384,
                              depth=12,
                              num_heads=6,
                              num_classes=num_classes,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # load offical pretrained model
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


def vit_base_patch16_224(pretrained=False, num_classes=1000, **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              num_classes=num_classes,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # load offical pretrained model
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


def vit_base_patch16_384(pretrained=False, num_classes=1000, **kwargs):
    model = VisionTransformer(img_size=384,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              num_classes=num_classes,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # load offical pretrained model
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


model_dict = {
    "vit_tiny": vit_tiny_patch16_224,
    "vit_small": vit_small_patch16_224,
    "vit_base": vit_base_patch16_224,
    'deit_base_384': vit_base_patch16_384,
}


def create_vit(model_name: str, num_classes: int = 1000, pretrained=False):
    return model_dict[model_name](num_classes=num_classes, pretrained=pretrained)


if __name__ == '__main__':
    from torchsummary import summary

    x = torch.rand((1, 3, 224, 224))
    net = create_vit(model_name="vit_tiny")
    print(net(x).shape)
    summary(net, (3, 224, 224))
