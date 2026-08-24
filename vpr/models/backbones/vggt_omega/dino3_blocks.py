from typing import List, Optional, Any, Literal, Tuple
from functools import partial

import torch
import torch.nn as nn

from .vggt_omega.models.layers import (
    RopePositionEmbedding,
    SelfAttentionBlock,
    RMSNorm
)
from .vggt_omega.models.layers.vision_transformer import (
    norm_layer_dict,
    ffn_layer_dict,
    dtype_dict,
    DinoVisionTransformer
)

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
            )
            for i in range(len(block_idcs))
        ]
        for new_blk, blk in zip(new_block_list, block_list):
            new_blk.load_state_dict(blk.state_dict())
        self.blocks = nn.ModuleList(new_block_list)

    def forward(self, x: torch.Tensor, rope: Tuple[int]):
        if self.rope_embed is not None:
            H, W = rope
            rope_sincos = self.rope_embed(H=H, W=W)
        else:
            rope_sincos = None
        for blk in self.blocks:
            x = blk(x, rope_sincos)
        
        return x

    def unfreeze(self) -> None:
        for param in self.blocks.parameters():
            param.requires_grad = True
        self.blocks.train()


def vit_large_blocks(
    dino_vit: DinoVisionTransformer,
    block_idcs: List[int],
    **kwargs
) -> Dinov3BlocksAdapter:
    block_list = [dino_vit.blocks[i] for i in block_idcs]

    assert dino_vit.embed_dim == 1024
    assert dino_vit.num_heads == 16

    adapter = Dinov3BlocksAdapter(
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
