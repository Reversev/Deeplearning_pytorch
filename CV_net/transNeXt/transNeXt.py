#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2024/3/24 10:27
# @Author : 'IReverser'
# @FileName: transNeXt.py
# Reference: https://github.com/DaiShiResearch/TransNeXt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)  # B, C, H, W -> B, C, H, W
        x = x.flatten(2).transpose(1, 2)  # B, C, H * W -> B, H * W, C
        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, out_feats=None, act_layer=nn.GELU, drop=0.):
        super(ConvolutionalGLU, self).__init__()
        out_feats = out_feats or in_feats
        hidden_feats = hidden_feats or in_feats
        hidden_feats = int(2 * hidden_feats / 3)
        self.fc1 = nn.Linear(in_feats, hidden_feats * 2)
        self.dwconv = DWConv(hidden_feats)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_feats, out_feats)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.drop(self.fc2(x))
        return x


@torch.no_grad()
def get_relative_position_cpb(query_size, key_size, pretrain_size=None):
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw)
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw)

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = torch.sign(relative_coords_table) * \
                            torch.log2(torch.abs(relative_coords_table) + 1.0) / \
                            torch.log2(torch.tensor(8, dtype=torch.float32))
    return idx_map, relative_coords_table


@torch.no_grad()
def get_seqlen_and_mask(input_resolution, wd_size):
    attn_map = F.unfold(torch.ones([1, 1, input_resolution[0], input_resolution[1]]), wd_size,
                        dilation=1, padding=(wd_size // 2, wd_size // 2), stride=1)
    attn_local_len = attn_map.sum(-2).squeeze().unsqueeze(-1)
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_local_len, attn_mask


class AggregatedAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, wd_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.sr_ratio = sr_ratio

        assert wd_size % 2 == 1, "window size must be odd"
        self.window_size = wd_size
        self.local_len = wd_size ** 2

        self.pool_H, self.pool_W = input_resolution[0] // self.sr_ratio, input_resolution[1] // self.sr_ratio
        self.pool_len = self.pool_H * self.pool_W

        self.unfold = nn.Unfold(kernel_size=wd_size, padding=wd_size // 2, stride=1)
        self.temperature = nn.Parameter(torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1)) #Initialize softplus(temperature) to 1/0.24.

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Components to generate pooled features.
        self.pool = nn.AdaptiveAvgPool2d((self.pool_H, self.pool_W))
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        # relative bias for local features
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0,
                                  std=0.0004))

        # Generate padding_mask && sequnce length scale
        local_seq_length, padding_mask = get_seqlen_and_mask(input_resolution, wd_size)
        self.register_buffer("seq_length_scale", torch.as_tensor(np.log(local_seq_length.numpy() + self.pool_len)),
                             persistent=False)
        self.register_buffer("padding_mask", padding_mask, persistent=False)

        # dynamic_local_bias:
        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        B, N, C = x.shape

        # Generate queries, normalize them with L2, add query embedding, and then magnify with sequence length scale and temperature.
        # Use softplus function ensuring that the temperature is not lower than 0.
        q_norm=F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3),dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * self.seq_length_scale

        # Generate unfolded keys and values and l2-normalize them
        k_local, v_local = self.kv(x).chunk(2, dim=-1)
        k_local = F.normalize(k_local.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
        kv_local = torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
        k_local, v_local = self.unfold(kv_local).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)

        # Compute local similarity
        attn_local = ((q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2) \
                      + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(self.padding_mask, float('-inf'))

        # Generate pooled features
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = self.pool(self.act(self.sr(x_))).reshape(B, -1, self.pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        # Generate pooled keys and values
        kv_pool = self.kv(x_).reshape(B, self.pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        # Use MLP to generate continuous relative positional bias for pooled features.
        pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                    relative_pos_index.view(-1)].view(-1, N, self.pool_len)
        # Compute pooled similarity
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias

        # Concatenate local & pooled similarity matrices and calculate attention weights through the same Softmax
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Split the attention weights and separately aggregate the values of local & pooled features
        attn_local, attn_pool = torch.split(attn, [self.local_len, self.pool_len], dim=-1)
        x_local = (((q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local).unsqueeze(-2) @ v_local.transpose(-2, -1)).squeeze(-2)
        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        # Linear projection and output
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1)) # initialize softplus to 1/0.24.

        # generate sequence length scale
        self.register_buffer("seq_len_scale", torch.as_tensor(np.log(input_resolution[0] * input_resolution[1])), persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query_embedding = nn.Parameter(nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim),
                                                                  mean=0, std=0.02))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=1)

        # use mlp to generate continuous relative positional bias
        rel_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:, relative_pos_index.view(-1)].view(-1, N, N)

        # calculate attention map using sequence length scaled cosine attention and query embedding
        attn = ((F.normalize(q, dim=-1) + self.query_embedding) * F.softplus(self.temperature) * self.seq_len_scale) \
               @ F.normalize(k, dim=-1).transpose(-2, -1) + rel_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, input_resolution, wd_size=3, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        if sr_ratio == 1:
            self.attn = Attention(dim,
                                  input_resolution,
                                  num_heads=num_heads,
                                  qkv_bias=qkv_bias,
                                  attn_drop=attn_drop,
                                  proj_drop=drop)
        else:
            self.attn = AggregatedAttention(dim,
                                            input_resolution,
                                            wd_size=wd_size,
                                            num_heads=num_heads,
                                            qkv_bias=qkv_bias,
                                            attn_drop=attn_drop,
                                            proj_drop=drop,
                                            sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvolutionalGLU(in_feats=dim, hidden_feats=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos_index, relative_coords_table))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, patch_size=7, stride=4, in_channels=3, embed_dim=768):
        super(OverlapPatchEmbed, self).__init__()
        patch_size = to_2tuple(patch_size)
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class TransNeXt(nn.Module):
    """
    The parameter "img size" is primarily utilized for generating relative spatial coordinates,
    which are used to compute continuous relative positional biases. As this TransNeXt implementation does not support multi-scale inputs,
    it is recommended to set the "img size" parameter to a value that is exactly the same as the resolution of the inference images.
    It is not advisable to set the "img size" parameter to a value exceeding 800x800.
    The "pretrain size" refers to the "img size" used during the initial pre-training phase,
    which is used to scale the relative spatial coordinates for better extrapolation by the MLP.
    For models trained on ImageNet-1K at a resolution of 224x224,
    as well as downstream task models fine-tuned based on these pre-trained weights,
    the "pretrain size" parameter should be set to 224x224.
    """
    def __init__(self, img_size=224, pretrain_size=None, wd_size=[3, 3, 3, None],
                 patch_size=16, in_channels=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4):
        super(TransNeXt, self).__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        pretrain_size = pretrain_size or img_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            # generate relative positional coordinate table and index for each stage to compute continuous relative positional bias.
            relative_pos_index, relative_coords_table = get_relative_position_cpb(query_size=to_2tuple(img_size // (2 ** (i + 2))),
                                                                                  key_size=to_2tuple(img_size // (2 ** (num_stages + 1))),
                                                                                  pretrain_size=to_2tuple(pretrain_size // (2 ** (i + 2))))

            self.register_buffer(f"relative_pos_index{i + 1}", relative_pos_index, persistent=False)
            self.register_buffer(f"relative_coords_table{i + 1}", relative_coords_table, persistent=False)

            patch_embed = OverlapPatchEmbed(patch_size=patch_size * 2 - 1 if i == 0 else 3,
                                            stride=patch_size if i == 0 else 2,
                                            in_channels=in_channels if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])
            block = nn.ModuleList([Block(dim=embed_dims[i], input_resolution=to_2tuple(img_size // (2 ** (i + 2))), wd_size=wd_size[i],
                                         num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                                         sr_ratio=sr_ratios[i])
                                   for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        for n, m in self.named_modules():
            self._init_weights(m, n)

    def _init_weights(self, m: nn.Module, name: str = ''):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"query_embedding", "relative_pos_bias_local", "cpb", "temperature"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            relative_pos_index = getattr(self, f"relative_pos_index{i + 1}")
            relative_coords_table = getattr(self, f"relative_coords_table{i + 1}")
            for blk in block:
                x = blk(x, H, W, relative_pos_index, relative_coords_table)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def transnext_micro(**kwargs):
    model = TransNeXt(wd_size=[3, 3, 3, None],
                      patch_size=4, embed_dims=[48, 96, 192, 384], num_heads=[2, 4, 8, 16],
                      mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 15, 2], sr_ratios=[8, 4, 2, 1],
                      **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def transnext_tiny(**kwargs):
    model = TransNeXt(wd_size=[3, 3, 3, None],
                      patch_size=4, embed_dims=[72, 144, 288, 576], num_heads=[3, 6, 12, 24],
                      mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 15, 2], sr_ratios=[8, 4, 2, 1],
                      **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def transnext_small(**kwargs):
    model = TransNeXt(window_size=[3, 3, 3, None],
                      patch_size=4, embed_dims=[72, 144, 288, 576], num_heads=[3, 6, 12, 24],
                      mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[5, 5, 22, 5], sr_ratios=[8, 4, 2, 1],
                      **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def transnext_base(**kwargs):
    model = TransNeXt(wd_size=[3, 3, 3, None],
                      patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[4, 8, 16, 32],
                      mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[5, 5, 23, 5], sr_ratios=[8, 4, 2, 1],
                      **kwargs)
    model.default_cfg = _cfg()

    return model


model_dict = {
    'transnext_micro': transnext_micro,
    'transnext_tiny': transnext_tiny,
    'transnext_small': transnext_small,
    'transnext_base': transnext_base,
}


def create_transnext(model_name: str, num_classes: int = 1000):
    return model_dict[model_name](num_classes=num_classes)


if __name__ == '__main__':
    # x = torch.randn((1, 3, 224, 224))
    # model = transnext_micro(num_classes=5)
    # print(model(x).shape)

    from torchsummary import summary
    x = torch.rand((2, 3, 224, 224))
    model = create_transnext(model_name="transnext_micro")
    print(model(x).shape)
    summary(create_transnext(model_name="transnext_micro"), (3, 224, 224))

