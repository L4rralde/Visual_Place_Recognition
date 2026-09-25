from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vggt_omega.models.layers.attention import SelfAttention


def inject_lora(
    linear: nn.Linear,
    r: int,
    alpha: Optional[int] = None,
    dropout: float = 0.1
) -> nn.Linear:
    # Freeze base module
    for p in linear.parameters():
        p.requires_grad = False

    alpha = alpha if alpha is not None else 2 * r
    scale = alpha / r
    out_features, in_features = linear.weight.shape

    # Register parameters directly on the linear layer
    linear.register_parameter("lora_A", nn.Parameter(torch.empty(r, in_features)))
    linear.register_parameter("lora_B", nn.Parameter(torch.zeros(out_features, r)))

    nn.init.kaiming_uniform_(linear.lora_A, a=math.sqrt(5))
    nn.init.zeros_(linear.lora_B)

    dropout_layer = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    # Forward hook adds the LoRA delta directly to the output
    def hook(module, inputs, output):
        x = inputs[0]
        lora_out = F.linear(
            F.linear(dropout_layer(x), module.lora_A),
            module.lora_B
        )
        return output + scale * lora_out

    linear.register_forward_hook(hook)
    return linear


class LoraLinear(nn.Module):
    def __init__(
            self,
            base: nn.Linear,
            r: int,
            alpha: Optional[int]=None,
            dropout: float=0.1
        ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise ValueError("Base model is not linear")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False

        if alpha is None:
            alpha = 2*r
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        out_features, in_features = base.weight.shape
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

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

    def merge_weights(self) -> None:
        raise NotImplementedError("Do not use yet")
        delta_w = self.scale * (self.lora_B @ self.lora_A)
        self.base.weight.data += delta_w


class SelfAttentionLora(SelfAttention):
    def __init__(
        self,
        dim,
        num_heads = 8,
        qkv_bias = False,
        proj_bias = True,
        attn_drop = 0,
        proj_drop = 0,
        mask_k_bias = False,
        use_qk_norm = False,
        device=None,
        lora_r: int=16,
        lora_alpha: int=32,
        lora_dropout: float=0.1
    ):
        super().__init__(
            dim,
            num_heads,
            qkv_bias,
            proj_bias,
            attn_drop,
            proj_drop,
            mask_k_bias,
            use_qk_norm,
            device
        )

        #self.qkv = LoraLinear(
        #    self.qkv,
        #    lora_r,
        #    lora_alpha,
        #    lora_dropout
        #)

        self.qkv = inject_lora(self.qkv, lora_r, lora_alpha, lora_dropout)

        #self.proj = LoraLinear(
        #    self.proj,
        #    lora_r,
        #    lora_alpha,
        #    lora_dropout
        #)
        self.proj = inject_lora(self.proj, lora_r, lora_alpha, lora_dropout)