"""ViT-based Strategy Network for ASTrA.

Smaller ViT (embed_dim=192, depth=6, num_heads=3) with 3 output heads
for adaptive adversarial attack parameter selection via REINFORCE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )

    def forward(self, x):
        normed = self.ln1(x)
        x = x + self.attn(normed, normed, normed, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class ViT_Strategy(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=192, depth=6,
                 num_heads=3, mlp_ratio=4, args=None):
        super(ViT_Strategy, self).__init__()
        self.args = args
        self.saved_log_probs = []
        self.saved_rewards = []
        self.rewards = []
        self.R1s = []
        self.R2s = []
        self.R3s = []

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.Attack_epsilon = nn.Linear(embed_dim, len(args.epsilon_types))
        self.Attack_iters = nn.Linear(embed_dim, len(args.attack_iters_types))
        self.Attack_step_size = nn.Linear(embed_dim, len(args.step_size_types))

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

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]  # CLS token

        Attack_epsilon = self.Attack_epsilon(x)
        Attack_iters = self.Attack_iters(x)
        Attack_step_size = self.Attack_step_size(x)

        return Attack_epsilon, Attack_iters, Attack_step_size


def ViT_Strategy_Small(args, img_size=32, patch_size=4):
    return ViT_Strategy(img_size=img_size, patch_size=patch_size,
                        embed_dim=192, depth=6, num_heads=3, args=args)
