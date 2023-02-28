# ！/usr/bin/python
# -*- coding: utf-8 -*-
# @Time: 2023/2/23 9:40
# @Author: 'IReverser'
# @ FileName: deit.py
# Reference: https://github.com/facebookresearch/deit
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


class DistilledViT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super(DistilledViT, self).__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))   # pos_embed += dist_pos + class_pos
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]    # logits: x[:, 0], distill: x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)   # classifier head
        x_dist = self.head_dist(x_dist)  # distillation head
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, num_classes=1000, **kwargs):
    model = DistilledViT(patch_size=16,
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
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, num_classes=1000, **kwargs):
    model = DistilledViT(patch_size=16,
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
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, num_classes=1000, **kwargs):
    model = DistilledViT(patch_size=16,
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
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, num_classes=1000, **kwargs):
    model = DistilledViT(img_size=384,
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
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


model_dict = {
    'deit_tiny': deit_tiny_distilled_patch16_224,
    'deit_small': deit_small_distilled_patch16_224,
    'deit_base': deit_base_distilled_patch16_224,
    'deit_base_384': deit_base_distilled_patch16_384,
}


def create_deit(model_name: str, num_classes: int = 1000, pretrained=False):
    return model_dict[model_name](num_classes=num_classes, pretrained=pretrained)


if __name__ == '__main__':
    from torchsummary import summary

    x = torch.rand((1, 3, 224, 224))
    net = create_deit(model_name="deit_tiny")
    print(net(x)[0].shape, net(x)[1].shape)
    summary(net, (3, 224, 224))

