from typing import Optional, List, Dict, Tuple
import gc
import os, sys

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from .transforms import preprocess_image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vggt_omega.models.vggt_omega import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from vggt_omega.models.aggregator import slice_expand_and_flatten, Aggregator
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer

def load_pretrained_vggt_omega(checkpoint: str) -> VGGTOmega:
    model = VGGTOmega()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model


class VggtOmegaBase(nn.Module):
    def __init__(
        self,
        vggt_omega: VGGTOmega,
        **kwargs
    ) -> None:
        super().__init__()
        if 'num_trainable_blocks' in kwargs:
            print("num_trainable_blocks argument is not supported for VGGT backbone. VGGT is used as is")
        self.norm_layer = kwargs.get('norm_layer', True)
        self.probing_from_layer: int = kwargs.get('probing_from_layer', -1)
        self.num_channels = vggt_omega.aggregator.patch_embed.embed_dim
        self.PATCH_SIZE = self.patch_size = vggt_omega.aggregator.patch_size
        self._resnet_mean = vggt_omega.aggregator._resnet_mean
        self._resnet_std = vggt_omega.aggregator._resnet_std
        assert isinstance(vggt_omega.aggregator.patch_embed.head, nn.Identity)
        self._vggt_omega: Optional[VGGTOmega] = None
        self._dino: Optional[DinoVisionTransformer] = None

    @property
    def vggt_omega(self) -> VGGTOmega:
        if self._vggt_omega is None:
            raise RuntimeError("self.vggt_omega is not set in this class")
        return self._vggt_omega

    @property
    def dino(self) -> DinoVisionTransformer:
        if self._dino is not None:
            return self._dino
        if self._vggt_omega is not None:
            return self._vggt_omega.aggregator.patch_embed
        raise RuntimeError("self.dino is not set in this class")

    def preprocess_image(self, img_list: Image.Image, **kwargs) -> Image.Image:
        return preprocess_image(img_list, **kwargs)

    def pose_encoding_to_extri_intri(
        self,
        pose_encoding: torch.Tensor,
        image_size_hw: Tuple[int]
    ) -> Tuple[torch.Tensor]:
        return encoding_to_camera(
            pose_encoding,
            image_size_hw
        )

    def _clip_probing_from_layer(self) -> int:
        if self.probing_from_layer < 0:
            self.probing_from_layer += self.dino.n_blocks
        assert 0 <= self.probing_from_layer < self.dino.n_blocks, \
            "Index probing_from_layer out of range"

    def prepare_tokens_for_salad(
        self,
        patch_tokens: Dict[str, torch.Tensor],
        images_shape: Tuple[int]
    ) -> Tuple[torch.Tensor]:
        B, S, C_in, H, W = images_shape

        f = patch_tokens['x_salad_patchtokens']
        t = patch_tokens['x_salad_clstoken']
        
        f = f.reshape((B*S, H//14, W//14, self.num_channels)).permute(0, 3, 1, 2)

        return f, t

    def inference(self, img_path_list: List[str], **kwargs) -> Dict[str, np.ndarray]:
        assert torch.cuda.is_available(), "Sadly, only works with cuda"
        DEVICE = "cuda"

        images = load_and_preprocess_images(img_path_list, **kwargs).to(DEVICE)
        with torch.inference_mode():
            predictions = self.forward(images)

        extrinsics, intrinsics = self.pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )

        predictions['extrinsic'] = extrinsics
        predictions['intrinsic'] = intrinsics

        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                predictions[key] = value.cpu().numpy().squeeze(0)

        return predictions

    def dino_forward_features(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        assert not self.dino.untie_cls_and_patch_norms, "I assumed it was False"
        assert not self.dino.untie_global_and_local_cls_norm, "I assumed it was False"
        x, rope = self.dino.prepare_tokens_with_masks(x)
        for i, blk in enumerate(self.dino.blocks):
            if self.dino.rope_embed is not None:
                H, W = rope
                rope_sincos = self.dino.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i == self.probing_from_layer:
                x_for_salad = x.clone()
        
        x_norm = self.dino.norm(x)
        if self.norm_layer:
            x_for_salad = self.dino.norm(x_for_salad)
        x_norm_cls_reg = x_norm[:, : self.dino.n_storage_tokens + 1]
        x_norm_patch = x_norm[:, self.dino.n_storage_tokens + 1 :]
        output = {
            "x_norm_clstoken": x_norm_cls_reg[:, 0],
            "x_storage_tokens": x_norm_cls_reg[:, 1:],
            "x_norm_patchtokens": x_norm_patch,
            "x_salad_clstoken": x_for_salad[:, 0],
            "x_salad_patchtokens": x_for_salad[:, self.dino.n_storage_tokens + 1 :],
            "x_prenorm": x,
            "masks": None,
        }
        return output

    def dino_forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)

        patch_tokens = self.dino_forward_features(images)

        return patch_tokens

    def alternate_attention(
        self,
        img_shape: tuple,
        tokens: torch.Tensor,
        patch_tokens: torch.Tensor
    ) -> Tuple[List[torch.Tensor | None], int]:
        batch_size, num_frames, num_channels, height, width = img_shape
        _, num_tokens, embed_dim = tokens.shape
        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self._vggt_omega.aggregator.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        outputs = []
        for block_idx in range(self._vggt_omega.aggregator.depth):
            tokens, frame_tokens = self._vggt_omega.aggregator._run_frame_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
            )
            tokens = self._vggt_omega.aggregator._run_inter_frame_attention_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self._vggt_omega.aggregator.inter_frame_attention_types[block_idx],
            )
            if block_idx in self._vggt_omega.aggregator.cached_layer_indices:
                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
            else:
                outputs.append(None)

        return outputs, self._vggt_omega.aggregator.patch_token_start

    def heads_forward(
        self,
        images: torch.Tensor,
        aggregated_tokens_list: List[torch.Tensor],
        patch_token_start: int
    ) -> Dict[str, torch.Tensor]:

        predictions = {}
        with torch.autocast(device_type="cuda", enabled=False):
            if self._vggt_omega.camera_head is not None:
                predictions["pose_enc"] = self._vggt_omega.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                )

            if self._vggt_omega.dense_head is not None:
                depth, depth_conf = self._vggt_omega.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self._vggt_omega.text_alignment_head is not None:
                predictions.update(
                    self._vggt_omega.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

        return predictions
    

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        batch_size, num_frames, num_channels, height, width = images.shape

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            patch_tokens = self.dino_forward(images)
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]
            
            camera_token = slice_expand_and_flatten(
                self._vggt_omega.aggregator.camera_token, batch_size, num_frames
            )
            register_token = slice_expand_and_flatten(
                self._vggt_omega.aggregator.register_token, batch_size, num_frames
            )
            
            aggregated_tokens_list, patch_token_start = self.alternate_attention(
                img_shape = (batch_size, num_frames, num_channels, height, width),
                tokens=torch.cat([camera_token, register_token, patch_tokens], dim=1),
                patch_tokens=patch_tokens,
            )

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Alternate attention blocks did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
        }

        predictions["images"] = images

        heads_predictions = self.heads_forward(
            images,
            aggregated_tokens_list,
            patch_token_start
        )

        return {
            **predictions,
            **heads_predictions
        }


class VggtOmegaBackbone(VggtOmegaBase):
    def __init__(self, vggt_omega: VGGTOmega, **kwargs):
        super().__init__(vggt_omega, **kwargs)
        self._vggt_omega = vggt_omega
        self._clip_probing_from_layer()

    @staticmethod
    def from_pretrained(checkpoint: str, **kwargs) -> "VggtOmegaBackbone":
        vggt_omega = load_pretrained_vggt_omega(checkpoint)
        return VggtOmegaBackbone(vggt_omega, **kwargs)


class VggtOmegaDino(VggtOmegaBase):
    def __init__(self, vggt_omega: VGGTOmega, norm_layer: bool=True, **kwargs):
        super().__init__(vggt_omega, norm_layer=norm_layer, **kwargs)
        self._dino = vggt_omega.aggregator.patch_embed
        self._clip_probing_from_layer()

    @staticmethod
    def from_pretrained(checkpoint: str, **kwargs) -> "VggtOmegaDino":
        vggt_omega = VGGTOmega()
        full_state = torch.load(checkpoint, map_location="cpu", weights_only=True)

        # Filter weights: Keep ONLY the dino parts
        # This assumes the prefix in the state_dict matches the module structure
        prefix = "aggregator.patch_embed."
        dino_state = {k: v for k, v in full_state.items() if k.startswith(prefix)}

        # Free the huge full_state dictionary immediately
        del full_state
        gc.collect() 

        # Load strictly the filtered weights
        # We must use strict=False because we are intentionally missing the rest of the model
        keys = vggt_omega.load_state_dict(dino_state, strict=False)

        backbone = VggtOmegaDino(VggtOmegaDino, **kwargs)

        #Final cleanup
        del vggt_omega
        del dino_state
        gc.collect()

        return backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, S, C_in, H, W = images.shape

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            patch_tokens = self.dino_forward(images)
            f, t = self.prepare_tokens_for_salad(patch_tokens, (B, S, C_in, H, W))

        return f, t
