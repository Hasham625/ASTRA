"""Vision Transformer (ViT) with dual LayerNorm support for ASTrA.

ViT-Small: embed_dim=384, depth=12, num_heads=6, patch_size=4
Supports dual LayerNorm ('normal'/'pgd') analogous to dual BatchNorm in ResNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import NormalizeByChannelMeanStd


class layer_norm_multiple(nn.Module):
    """Dual LayerNorm — analogous to batch_norm_multiple in ResNet."""

    def __init__(self, normalized_shape, ln_names=None):
        super(layer_norm_multiple, self).__init__()
        self.ln_names = ln_names
        if self.ln_names is None:
            self.ln_list = nn.LayerNorm(normalized_shape)
            return
        len_ln_names = len(ln_names)
        self.ln_list = nn.ModuleList([nn.LayerNorm(normalized_shape) for _ in range(len_ln_names)])
        self.ln_names_dict = {ln_name: i for i, ln_name in enumerate(ln_names)}

    def forward(self, x):
        out = x[0]
        name_ln = x[1]
        if name_ln is None:
            out = self.ln_list(out)
        else:
            ln_index = self.ln_names_dict[name_ln]
            out = self.ln_list[ln_index](out)
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)                          # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)          # (B, num_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)            # (B, num_patches+1, embed_dim)
        x = x + self.pos_embed
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm ViT block. Accepts/returns [tensor, bn_name] for nn.Sequential compatibility."""

    def __init__(self, embed_dim=384, num_heads=6, mlp_ratio=4, bn_names=None):
        super().__init__()
        self.ln1 = layer_norm_multiple(embed_dim, ln_names=bn_names)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln2 = layer_norm_multiple(embed_dim, ln_names=bn_names)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio))

    def forward(self, x):
        tensor, bn_name = x[0], x[1]
        tensor = tensor + self.attn(self.ln1([tensor, bn_name]))
        tensor = tensor + self.mlp(self.ln2([tensor, bn_name]))
        return [tensor, bn_name]


class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4, bn_names=None, num_classes=1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.bn_names = bn_names

        self.normalize = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, bn_names) for _ in range(depth)]
        )
        self.norm = layer_norm_multiple(embed_dim, ln_names=bn_names)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.fc.in_features = embed_dim

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x, bn_name=None):
        x = self.normalize(x)
        x = self.patch_embed(x)                    # (B, N+1, embed_dim)
        x = self.blocks([x, bn_name])              # [tensor, bn_name]
        x = self.norm([x[0], bn_name])             # (B, N+1, embed_dim)
        x = x[:, 0]                                # CLS token -> (B, embed_dim)

        if isinstance(self.fc, proj_head_vit):
            x = self.fc(x, bn_name)
        else:
            x = self.fc(x)
        return x

    def forward(self, x, bn_name=None):
        return self._forward_impl(x, bn_name)


class proj_head_vit(nn.Module):
    """Projection head using dual LayerNorm — analogous to proj_head in ResNet."""

    def __init__(self, ch, bn_names=None, twoLayerProj=False):
        super(proj_head_vit, self).__init__()
        self.in_features = ch
        self.twoLayerProj = twoLayerProj

        self.fc1 = nn.Linear(ch, ch * 4)
        self.ln1 = layer_norm_multiple(ch * 4, ln_names=bn_names)
        self.fc2 = nn.Linear(ch * 4, ch, bias=False)
        self.ln2 = layer_norm_multiple(ch, ln_names=bn_names)

        if not twoLayerProj:
            self.fc3 = nn.Linear(ch, ch, bias=False)
            self.ln3 = layer_norm_multiple(ch, ln_names=bn_names)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, bn_name):
        x = self.fc1(x)
        x = self.ln1([x, bn_name])
        x = self.relu(x)
        x = self.fc2(x)
        x = self.ln2([x, bn_name])
        if not self.twoLayerProj:
            x = self.relu(x)
            x = self.fc3(x)
            x = self.ln3([x, bn_name])
        return x


def vit_small(img_size=32, patch_size=4, bn_names=None, num_classes=1000):
    return ViT(img_size=img_size, patch_size=patch_size, embed_dim=384,
               depth=12, num_heads=6, mlp_ratio=4, bn_names=bn_names,
               num_classes=num_classes)
