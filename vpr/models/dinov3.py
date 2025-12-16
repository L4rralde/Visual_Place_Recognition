from os import PathLike

import torch
import torch.nn as nn


class DINOv3(nn.Module):
    def __init__(
            self,
            model_name: str,
            url: str,
            dinov3_repo_path: PathLike, 
            num_trainable_blocks: int = 2,
            norm_layer: bool = False,
            return_token: bool = False,
            rope_sincos: bool = True,
            probing_from_layer: int|None = None
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.model = torch.hub.load(
            dinov3_repo_path,
            model_name,
            source = 'local',
            weights = url
        )
        self.num_channels = self.model.num_features
        self.num_trainable_blocks = num_trainable_blocks
        self.num_blocks = len(self.model.blocks)
        self.norm_layer = norm_layer
        self.return_token = return_token
        self.rope_sincos = rope_sincos
        self.linear_probing = probing_from_layer is not None

        #Validate inputs
        if self.linear_probing:
            assert num_trainable_blocks <= (probing_from_layer + 1), \
                'num_trainable_blocks exceeds available layers for probing'
            assert 0 <= probing_from_layer < self.num_blocks, \
                f'probing_from_layer must be between 0 and {self.num_blocks - 1}'

        assert 0 <= num_trainable_blocks <= self.num_blocks, \
            f'num_trainable_blocks must be between 0 and total blocks: {self.num_blocks}'

        if not self.linear_probing:
            valid_blocks = self.model.blocks
        else:
            valid_blocks = self.model.blocks[:probing_from_layer + 1]

        if self.num_trainable_blocks > 0:
            self.frozen_blocks = valid_blocks[:-self.num_trainable_blocks]
            self.trainable_blocks = valid_blocks[-self.num_trainable_blocks:]
        else:
            self.frozen_blocks = valid_blocks
            self.trainable_blocks = []
        
        self.freeze_blocks()

    def freeze_blocks(self) -> None:
        for blk in self.frozen_blocks:
            for param in blk.parameters():
                param.requires_grad = False
        
        if self.num_trainable_blocks == 0 and not self.linear_probing:
            for param in self.model.norm.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple:
        B, _, h, w = x.shape
        x, (H, W) = self.model.prepare_tokens_with_masks(x)

        if self.rope_sincos:
            rope_sincos = self.model.rope_embed(H=H, W=W)
        else:
            rope_sincos = None

        # First blocks are frozen
        with torch.no_grad():
            for blk in self.frozen_blocks:
                x = blk(x, rope_sincos)

        # Last blocks are trained
        for blk in self.trainable_blocks:
            x = blk(x, rope_sincos)

        if self.norm_layer:
            if self.num_trainable_blocks == 0 and not self.linear_probing:
                with torch.no_grad():
                    x = self.model.norm(x)
            else:
                x = self.model.norm(x)

        class_token = x[:, 0]
        register_token = x[:, 1: self.model.n_storage_tokens + 1] #Probably it adds nothing for inference

        features = x[:, self.model.n_storage_tokens + 1 :]

        features = features.reshape((
            B,
            h // self.model.patch_size,
            w // self.model.patch_size,
            self.num_channels
        )).permute(0, 3, 1, 2).contiguous()

        if self.return_token:
            return features, class_token
        return features #Future fix: must return same type.
