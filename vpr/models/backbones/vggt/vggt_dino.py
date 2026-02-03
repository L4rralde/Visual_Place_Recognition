from typing import List, Dict, Tuple
import gc

import torch
import torch.nn as nn

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vggt.models.vggt import VGGT
from vggt.models.aggregator import slice_expand_and_flatten
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class VggtBackbone(nn.Module):
    def __init__(
        self,
        vggt: VGGT,
        **kwargs
    ):
        super().__init__()
        # 1. DO NOT do self.vggt = vggt. That keeps the whole model alive.
        
        if 'num_trainable_blocks' in kwargs:
            print("num_trainable_blocks argument is not supported for VGGT backbone. VGGT is used as is")
        if 'norm_layer' in kwargs:
            #FUTURE
            print("norm_layer argument flag is not supported for da3. VGGT is used as is")
            print("FUTURE. But this argument can be implemented in the future")

        # 2. Extract only the specific submodule and attributes you need
        self.num_channels = vggt.aggregator.patch_embed.embed_dim
        self.dino = vggt.aggregator.patch_embed 
        self._resnet_std = vggt.aggregator._resnet_std
        self._resnet_mean = vggt.aggregator._resnet_mean
        
        # 3. The 'vggt' object passed in is now a local variable. 
        # When __init__ finishes, if no one else references it, 
        # Python will garbage collect the unused parts (the other 95% of the model).

    @staticmethod
    def from_pretrained(**kwargs) -> "VggtBackbone":
        vggt = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        vggt.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
        return VggtBackbone(vggt, **kwargs)

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
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
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
        camera_token = slice_expand_and_flatten(self.vggt.aggregator.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.vggt.aggregator.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.vggt.aggregator.rope is not None:
            pos = self.vggt.aggregator.position_getter(
                B * S,
                H // self.vggt.aggregator.patch_size,
                W // self.vggt.aggregator.patch_size,
                device=images.device
            )

        if self.vggt.aggregator.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.vggt.aggregator.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.vggt.aggregator.aa_block_num):
            for attn_type in self.vggt.aggregator.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self.vggt.aggregator._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self.vggt.aggregator._process_global_attention(
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
        return output_list, self.vggt.aggregator.patch_start_idx

    def heads_forward(self, images: torch.Tensor, aggregated_tokens_list: List[torch.Tensor], patch_start_idx: int, query_points:torch.Tensor) -> Dict[str, torch.Tensor]:
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.vggt.camera_head is not None:
                pose_enc_list = self.vggt.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.vggt.depth_head is not None:
                depth, depth_conf = self.vggt.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.vggt.point_head is not None:
                pts3d, pts3d_conf = self.vggt.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.vggt.track_head is not None and query_points is not None:
            track_list, vis, conf = self.vggt.track_head(
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


class VggtDino(VggtBackbone):
    def __init__(self, vggt: VGGT, norm_layer: bool=True, **kwargs):
        super().__init__(vggt, **kwargs)
        self.norm_layer = norm_layer

    @staticmethod
    def from_pretrained(**kwargs) -> "VggtDino":
        vggt = VGGT()        
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        full_state = torch.hub.load_state_dict_from_url(_URL, map_location='cpu')

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

    def prepare_tokens_for_salad(self, patch_tokens: Dict[str, torch.Tensor], images_shape: Tuple[int]) -> Tuple[torch.Tensor]:
        B, S, C_in, H, W = images_shape

        if self.norm_layer:
            f = patch_tokens['x_norm_patchtokens']
            t = patch_tokens['x_norm_clstoken']
        else:
            raise RuntimeError("Not implemented yet. Work in progress.")
        
        f = f.reshape((B*S, H//14, W//14, self.num_channels)).permute(0, 3, 1, 2)

        return f, t
