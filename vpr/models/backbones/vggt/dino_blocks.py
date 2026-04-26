from functools import partial
from typing import List

import torch
import torch.nn as nn

from vpr.models.backbones.vggt.vggt.layers import(
    Mlp,
    SwiGLUFFNFused,
    MemEffAttention,
    NestedTensorBlock as Block
)


class DinoBlocksAdapter(nn.Module):
    def __init__(
        self,
        block_list: List[nn.Module],
        block_idcs: List[int],
        embed_dim: int, #Depends on ViT Conf
        depth: int, #Depends on ViT Conf
        num_heads: int, #Depends on ViT Conf
        mlp_ratio: int, #Depends on ViT Conf
        qkv_bias: bool=True,
        ffn_bias: bool=True,
        proj_bias: bool=True,
        drop_path_rate: float=0.0,
        drop_path_uniform: bool=False,
        init_values: float=None,
        act_layer: nn.Module=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        qk_norm: bool=False,
        norm_layer_ws: dict={}
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        dpr = [dpr[i] for i in block_idcs]

        if ffn_layer == "mlp":
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            def f(*args, **kwargs):
                return nn.Identity()
            ffn_layer = f
        else:
            raise NotImplementedError
        
        new_block_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                qk_norm=qk_norm,
            )
            for i in range(len(block_list))
        ]
        for new_blk, blk in zip(new_block_list, block_list):
            new_blk.load_state_dict(blk.state_dict())
        self.blocks = nn.ModuleList(new_block_list)
        self.norm = norm_layer(embed_dim)
        if norm_layer_ws:
            self.norm.load_state_dict(norm_layer_ws)

    def forward(self, x: torch.Tensor):
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x
    

def vit_large_blocks(dino_vit, block_idcs, **kwargs):
    block_list = [dino_vit.blocks[i] for i in block_idcs]
    assert dino_vit.embed_dim == 1024
    assert dino_vit.n_blocks == 24
    assert dino_vit.num_heads == 16
    init_values = kwargs.pop('init_values', 1.0)
    model = DinoBlocksAdapter(
        block_list,
        block_idcs,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        init_values=init_values,
        block_fn=partial(Block, attn_class=MemEffAttention),
        #norm_layer_ws=dino_vit.norm.state_dict(),
        **kwargs,
    )
    return model
