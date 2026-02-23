from typing import List, Dict
import gc

import torch
import torch.nn as nn
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vpr.models.backbones.vggt import VggtBackbone, load_pretrained_vggt
from vpr.models import SALAD
from utils import LightningLog
from submodules.vggt.vggt.utils.load_fn import load_and_preprocess_images


class VggtSalad(nn.Module):
    def __init__(
        self,
        vggt: object,
        backbone_args: dict={},
        agg_args: dict={}
    ) -> None:
        super().__init__()
        self.backbone = VggtBackbone(vggt, **backbone_args)
        self.aggregator = SALAD(
            num_channels=self.backbone.num_channels,
            **agg_args
        )

    @staticmethod
    def from_lightning_log(path: str, vggt: object|None=None) -> "VggtSalad":
        log = LightningLog(path)
        assert log.agg_arch.upper() == "SALAD", "By the moment only SALAD is supported"
        assert log.backbone_arch.upper() == "VGGT", "This log might not correspond to vggt-salad"
        if vggt is None:
            vggt = load_pretrained_vggt()
        model = VggtSalad(vggt, log.backbone_config, log.agg_config)
        full_state = log.state_dict
        prefix = "aggregator"
        salad_state = {k: v for k, v in full_state.items() if k.startswith(prefix)}
        del full_state
        gc.collect()

        model.load_state_dict(salad_state, strict=False)

        return model
    
    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        
        patch_tokens = self.backbone.dino_forward(images)
        feats, cls = self.backbone.prepare_tokens_for_salad(patch_tokens, images.shape)
        global_descriptor = self.aggregator((feats, cls))
        if len(global_descriptor.shape) == 2:
            global_descriptor = global_descriptor.unsqueeze(0)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        aggregated_tokens_list, patch_start_idx = self.backbone.alternate_attention(images, patch_tokens)
        predictions = self.backbone.heads_forward(images, aggregated_tokens_list, patch_start_idx, query_points)
        predictions['descriptor'] = global_descriptor

        return predictions

    def inference(self, img_path_list: List[str]) -> Dict[str, np.ndarray]:
        assert torch.cuda.is_available(), "Only works with cuda"
        DEVICE = "cuda"
        gc.collect()
        torch.cuda.empty_cache()

        images = load_and_preprocess_images(img_path_list).to(DEVICE)

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(DEVICE, dtype=dtype):
                predictions = self.forward(images)

        extrinsic, intrinsic = self.backbone.pose_encoding_to_extri_intri(
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
