import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.vggt.layers.attention import Attention, MemEffAttention


class LoraLinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: Optional[int] = None,
        dropout: float=0.0
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise ValueError("Base model is not linear")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        for param in self.base.parameters():
            param.requires_grad = False
        if alpha is None:
            alpha = 2*r

        self.lora_dropout = (
            nn.Dropout(p=dropout)
            if dropout > 0.0 
            else nn.Identity()
        )

        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.scale = alpha/r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)

        lora_out = F.linear(
            F.linear(
                self.lora_dropout(x),
                self.lora_A
            ),
            self.lora_B
        )

        return base_out + self.scale * lora_out


class LoraLinearInference(LoraLinear):
    def __init__(
            self,
            base: nn.Linear,
            r: int,
            alpha: Optional[int]=None
        ):
        super().__init__(base, r, alpha)

        self.register_buffer(
            "agg_lora_matrix",
            None,
            persistent=False,
        )

    def merge_weights(self) -> None:
        with torch.no_grad():
            self.agg_lora_matrix = self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(
            x,
            self.agg_lora_matrix
        )

        return base_out + lora_out


class SelfAttentionLora(MemEffAttention):
    def __init__(self,
            dim, num_heads = 8, qkv_bias = True, proj_bias = True, attn_drop = 0, proj_drop = 0, norm_layer = nn.LayerNorm, qk_norm = False, fused_attn = True, rope=None,
            lora_r: int=16,
            lora_alpha: int=32,
            lora_dropout: float=0.0
        ) -> None:
        super().__init__(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop, norm_layer, qk_norm, fused_attn, rope)

        self.qkv = LoraLinear(
            self.qkv,
            lora_r,
            lora_alpha,
            lora_dropout
        )

        self.proj = LoraLinear(
            self.proj,
            lora_r,
            lora_alpha,
            lora_dropout
        )
