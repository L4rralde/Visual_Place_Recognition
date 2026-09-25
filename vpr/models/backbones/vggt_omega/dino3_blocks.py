from typing import List, Optional, Any, Literal, Tuple, Type
from functools import partial

import torch
import torch.nn as nn

from .vggt_omega.models.layers import (
    RopePositionEmbedding,
    SelfAttentionBlock,
)
from .vggt_omega.models.layers.vision_transformer import (
    norm_layer_dict,
    ffn_layer_dict,
    DinoVisionTransformer
)
from .self_attention_lora import SelfAttentionLora
from .vggt_omega.models.layers.attention import SelfAttention


class Dinov3BlocksAdapter(nn.Module):
    def __init__(
        self,
        block_list: List[nn.Module],
        block_idcs: List[int],
        embed_dim: int,
        num_heads: int,
        rope_embed: RopePositionEmbedding,
        device: Optional[Any]=None,
        norm_layer: str = "layernorm",
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        ffn_layer: str = "mlp",
        drop_path_rate: float = 0.0,
        layerscale_init: Optional[float] = None,
        mask_k_bias: bool = False,
        attn_class: Type[SelfAttention]=SelfAttention,
        **kwargs
    ) -> None:
        super().__init__()
        assert len(block_list) == len(block_idcs)

        self.rope_embed = rope_embed

        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        norm_layer_cls = norm_layer_dict[norm_layer]

        new_block_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
                attn_class=attn_class
            )
            for i in range(len(block_idcs))
        ]
        for new_blk, blk in zip(new_block_list, block_list):
            missing, unexpected = new_blk.load_state_dict(
                blk.state_dict(),
                strict=False,
            )

            missing = [
                name for name in missing
                if not "lora_" in name
            ]
            assert not missing, missing
            assert not unexpected, unexpected

        self.blocks = nn.ModuleList(new_block_list)
        self._frozen: bool = True

    def forward(self, x: torch.Tensor, rope: Tuple[int, int]):
        if self.rope_embed is not None:
            H, W = rope
            rope_sincos = self.rope_embed(H=H, W=W)
        else:
            rope_sincos = None
        for blk in self.blocks:
            x = blk(x, rope_sincos)
        
        return x

    def unfreeze(self) -> None:
        if not self._frozen:
            return

        print(f"Unfreezing {self.__class__.__name__} adapter")
        for param in self.blocks.parameters():
            param.requires_grad = True
        self.blocks.train()
        self._frozen = False


class Dinov3BlocksAdapterLora(Dinov3BlocksAdapter):
    def __init__(
        self,
        block_list,
        block_idcs,
        embed_dim,
        num_heads,
        rope_embed,
        device = None,
        norm_layer = "layernorm",
        ffn_ratio = 4,
        qkv_bias = True,
        ffn_bias = True,
        proj_bias = True,
        ffn_layer = "mlp",
        drop_path_rate = 0,
        layerscale_init = None,
        mask_k_bias = False,
        lora_rank: int=16,
        lora_alpha: int=32,
        lora_dropout: float=0.1,
        **kwargs
    ):
        attn_class = partial(
            SelfAttentionLora,
            lora_r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        super().__init__(
            block_list,
            block_idcs,
            embed_dim,
            num_heads,
            rope_embed,
            device,
            norm_layer,
            ffn_ratio,
            qkv_bias,
            ffn_bias,
            proj_bias,
            ffn_layer,
            drop_path_rate,
            layerscale_init,
            mask_k_bias,
            attn_class
        )
    
    def unfreeze(self) -> None:
        if not self._frozen:
            return

        print(f"Unfreezing {self.__class__.__name__} adapter")
        for name, param in self.blocks.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        self.blocks.train()
        self._frozen = False


def vit_large_blocks(
    dino_vit: DinoVisionTransformer,
    block_idcs: List[int],
    **kwargs
) -> Dinov3BlocksAdapter:
    block_list = [dino_vit.blocks[i] for i in block_idcs]

    assert dino_vit.embed_dim == 1024
    assert dino_vit.num_heads == 16

    if kwargs.get("lora", False):
        block_type = Dinov3BlocksAdapterLora
    else:
        block_type = Dinov3BlocksAdapter
    
    adapter = block_type(
        block_list,
        block_idcs,
        embed_dim=1024,
        num_heads=16,
        rope_embed=dino_vit.rope_embed,
        ffn_ratio=4.0,
        norm_layer="layernormbf16",
        layerscale_init=1e-05,
        mask_k_bias=True,
        **kwargs
    )

    return adapter
