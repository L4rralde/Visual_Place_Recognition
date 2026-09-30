from typing import Optional, Dict, List
import gc

import torch
import torch.nn as nn
import numpy as np

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vpr.models.backbones.vggt_omega import VggtOmegaBackbone, load_pretrained_vggt_omega
from vpr.models import SALAD
from utils import LightningLog
from vpr.models.backbones.vggt_omega.vggt_omega.utils.load_fn import load_and_preprocess_images
from vpr.models.backbones.vggt_omega.vggt_omega.models.aggregator import slice_expand_and_flatten


class VggtOmegaSalad(nn.Module):
    def __init__(
        self,
        vggt_omega: nn.Module,
        backbone_args: dict={},
        agg_args: dict={}
    ) -> None:
        super().__init__()
        self.backbone: VggtOmegaBackbone = VggtOmegaBackbone(vggt_omega, **backbone_args)
        self.aggregator: SALAD = SALAD(
            num_channels=self.backbone.num_channels,
            **agg_args
        )

        @staticmethod
        def from_lightning_log(
            path: str,
            vggt_omega: Optional[nn.Module]=None,
            vggt_omega_checkpoint: Optional[str]=None
        ) -> "VggtOmegaSalad":
            log = LightningLog(path)
            if not log.agg_arch.upper == "SALAD":
                raise RuntimeError("By the moment SALAD is the only supported aggregator")
            if not log.backbone_arch.upper() == "VGGT_OMEGA":
                raise RuntimeError("This log may not correspond to vggt-omega-salad")

            if vggt_omega is None:
                if vggt_omega_checkpoint is None:
                    raise ValueError("VGGT_Omega checkpoint is required")
                vggt_omega = load_pretrained_vggt_omega(vggt_omega_checkpoint)

            model = VggtOmegaSalad(
                vggt_omega,
                log.backbone_config,
                log.agg_config
            )
            full_state = log.state_dict
            salad_state = {
                k:v for k,v in full_state.items()
                if k.startswith("aggregator")
            }
            adapter_state = {
                k:v for k,v in full_state.items()
                if k.startswith("backbone.adapter")
            }
            learned_state = {**salad_state, **adapter_state}

            del full_state
            gc.collect()

            model.load_state_dict(learned_state, strict=False)

            return model

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        batch_size, num_frames, num_channels, height, width = images.shape

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            patch_tokens = self.backbone.dino_forward(images)

            feats, cls = self.backbone.prepare_tokens_for_salad(
                patch_tokens, 
                (batch_size, num_frames, num_channels, height, width)
            )
            
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]
            
            camera_token = slice_expand_and_flatten(
                self.backbone._vggt_omega.aggregator.camera_token,
                batch_size,
                num_frames
            )

            register_token = slice_expand_and_flatten(
                self.backbone._vggt_omega.aggregator.register_token,
                batch_size,
                num_frames
            )
            aggregated_tokens_list, patch_token_start = self.backbone.alternate_attention(
                img_shape = (batch_size, num_frames, num_channels, height, width),
                tokens=torch.cat([camera_token, register_token, patch_tokens], dim=1),
                patch_tokens=patch_tokens,
            )

        global_descriptor = self.aggregator((feats, cls))
        if len(global_descriptor.shape) == 2:
            global_descriptor = global_descriptor.unsqueeze(0)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Alternate attention blocks did not cache the final layer, which VGGTOmega needs.")

        predictions = {"images": images}

        heads_predictions = self.backbone.heads_forward(
            images,
            aggregated_tokens_list,
            patch_token_start
        )

        predictions['descriptor'] = global_descriptor
        return {
            **predictions,
            **heads_predictions
        }

    def inference(self, img_path_list: List[str], **kwargs) -> Dict[str, np.ndarray]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
    
        DEVICE = "cuda"
        images = load_and_preprocess_images(img_path_list, **kwargs).to(DEVICE)
        with torch.inference_mode():
            predictions = self.forward(images)

        extrinsics, intrinsics = self.backbone.pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            predictions["images"].shape[-2:],
        )

        predictions['extrinsic'] = extrinsics
        predictions['intrinsic'] = intrinsics

        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                predictions[key] = value.cpu().numpy().squeeze(0)

        return predictions
