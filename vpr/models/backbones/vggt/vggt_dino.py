from typing import List, Dict, Tuple
import gc

from PIL import Image
import torch
import torch.nn as nn

from .transforms import preprocess_image
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vggt.models.vggt import VGGT
from vggt.models.aggregator import slice_expand_and_flatten
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def load_vggt_state_dict():
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state_dict = torch.hub.load_state_dict_from_url(_URL, map_location='cpu')
    return state_dict


def load_pretrained_vggt() -> VGGT:
    vggt = VGGT()
    state_dict = load_vggt_state_dict()
    vggt.load_state_dict(state_dict)
    return vggt


class VggtBase(nn.Module):
    PATCH_SIZE = 14
    def __init__(
        self,
        vggt: VGGT,
        **kwargs
    ):
        super().__init__()
        if 'num_trainable_blocks' in kwargs:
            print("num_trainable_blocks argument is not supported for VGGT backbone. VGGT is used as is")
        self.norm_layer = kwargs.get('norm_layer', True)
        self.num_channels = vggt.aggregator.patch_embed.embed_dim
        self._resnet_std = vggt.aggregator._resnet_std
        self._resnet_mean = vggt.aggregator._resnet_mean
        self._vggt: VGGT|None = None
        self._dino = None
    
    @property
    def vggt(self) -> VGGT:
        if self._vggt is None:
            raise RuntimeError("self.vggt is not set in this class")
        return self._vggt

    @property
    def dino(self) -> nn.Module:
        if self._dino is not None:
            return self._dino
        if self._vggt is not None:
            return self._vggt.aggregator.patch_embed
        raise RuntimeError("self.dino is not set in this class")

    def inference(self, img_path_list: List[str]) -> dict:
        assert torch.cuda.is_available(), "Sadly, only works with cuda"
        DEVICE = "cuda"
        gc.collect() #Collect garbage
        torch.cuda.empty_cache()

        images = load_and_preprocess_images(img_path_list).to(DEVICE)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(DEVICE, dtype=dtype):
                predictions = self.forward(images)
        extrinsic, intrinsic = self.pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            images.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        torch.cuda.empty_cache()

        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                predictions[key] = value.cpu().numpy().squeeze(0)

        return predictions

    def dino_forward(self, images: torch.Tensor) -> torch.Tensor:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean.to('cuda')) / self._resnet_std.to('cuda')

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.dino(images) #This is all we need.

        return patch_tokens

    def alternate_attention(self, images: torch.Tensor, patch_tokens: torch.Tensor) -> tuple:
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self._vggt.aggregator.camera_token, B, S)
        register_token = slice_expand_and_flatten(self._vggt.aggregator.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self._vggt.aggregator.rope is not None:
            pos = self._vggt.aggregator.position_getter(
                B * S,
                H // self._vggt.aggregator.patch_size,
                W // self._vggt.aggregator.patch_size,
                device=images.device
            )

        if self._vggt.aggregator.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self._vggt.aggregator.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self._vggt.aggregator.aa_block_num):
            for attn_type in self._vggt.aggregator.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._vggt.aggregator._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._vggt.aggregator._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self._vggt.aggregator.patch_start_idx

    def heads_forward(self, images: torch.Tensor, aggregated_tokens_list: List[torch.Tensor], patch_start_idx: int, query_points:torch.Tensor) -> Dict[str, torch.Tensor]:
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self._vggt.camera_head is not None:
                pose_enc_list = self._vggt.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self._vggt.depth_head is not None:
                depth, depth_conf = self._vggt.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self._vggt.point_head is not None:
                pts3d, pts3d_conf = self._vggt.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self._vggt.track_head is not None and query_points is not None:
            track_list, vis, conf = self._vggt.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        
        patch_tokens = self.dino_forward(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        aggregated_tokens_list, patch_start_idx = self.alternate_attention(images, patch_tokens)
        predictions = self.heads_forward(images, aggregated_tokens_list, patch_start_idx, query_points)

        return predictions

    def prepare_tokens_for_salad(self, patch_tokens: Dict[str, torch.Tensor], images_shape: Tuple[int]) -> Tuple[torch.Tensor]:
        B, S, C_in, H, W = images_shape

        if self.norm_layer:
            f = patch_tokens['x_norm_patchtokens']
            t = patch_tokens['x_norm_clstoken']
        else:
            raise RuntimeError("Not implemented yet. Work in progress.")
        
        f = f.reshape((B*S, H//14, W//14, self.num_channels)).permute(0, 3, 1, 2)

        return f, t

    def pose_encoding_to_extri_intri(self, pose_encoding: torch.Tensor, image_size_hw: tuple) -> tuple:
        return pose_encoding_to_extri_intri(pose_encoding, image_size_hw)

    def preprocess_image(self, img_list: Image.Image) -> Image.Image:
        return preprocess_image(img_list)


class VggtBackbone(VggtBase):
    def __init__(self, vggt, **kwargs):
        super().__init__(vggt, **kwargs)
        self._vggt = vggt

    @staticmethod
    def from_pretrained(**kwargs) -> "VggtBackbone":
        vggt = load_pretrained_vggt()
        return VggtBackbone(vggt, **kwargs)


class VggtDino(VggtBase):
    def __init__(self, vggt: VGGT, norm_layer: bool=True, **kwargs):
        super().__init__(vggt, **kwargs)
        self.norm_layer = norm_layer
        self._dino = vggt.aggregator.patch_embed

    @staticmethod
    def from_pretrained(**kwargs) -> "VggtDino":
        vggt = VGGT()        
        full_state = load_vggt_state_dict()

        # Filter weights: Keep ONLY the dino parts
        # This assumes the prefix in the state_dict matches the module structure
        prefix = "aggregator.patch_embed."
        dino_state = {k: v for k, v in full_state.items() if k.startswith(prefix)}
        
        # Free the huge full_state dictionary immediately
        del full_state
        gc.collect() 
        
        # Load strictly the filtered weights
        # We must use strict=False because we are intentionally missing the rest of the model
        keys = vggt.load_state_dict(dino_state, strict=False)
        #print(f"VRAM Optimization: Loaded only {len(dino_state)} keys. Discarded the rest.")

        # Create the backbone
        backbone = VggtDino(vggt, **kwargs)
        
        # Final Cleanup
        del vggt
        del dino_state
        gc.collect()
        
        return backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, S, C_in, H, W = images.shape
        assert C_in == 3, "Wrong torch image format"

        with torch.no_grad():
            patch_tokens = self.dino_forward(images)
        
        #x_norm_patchtokens shape = B*S, Total patches, channles
        #x_norm_clstoken shape: B*S, channles
        f, t = self.prepare_tokens_for_salad(patch_tokens, (B, S, C_in, H, W))

        return f, t
