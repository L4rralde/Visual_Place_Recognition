from typing import List, Dict

from PIL import Image
import torch
import torch.nn as nn
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vpr.models.backbones.da3 import DepthAnything3Dino
from vpr.models import SALAD
from utils import LightningLog


class DA3Salad(nn.Module):
    def __init__(
        self,
        backbone_arch: str,
        backbone_args: dict={},
        agg_args: dict={}
    ) -> None:
        super().__init__()
        self.backbone = DepthAnything3Dino(backbone_arch, **backbone_args)
        self.aggregator = SALAD(
            num_channels=self.backbone.num_channels,
            **agg_args
        )

    @staticmethod
    def from_lightning_log(path: str) -> "DA3Salad":
        log = LightningLog(path)
        assert log.agg_arch == "SALAD", "By the moment only SALAD is supported"
        model = DA3Salad(
            log.backbone_arch,
            log.backbone_config,
            log.agg_config
        )
        model.load_state_dict(log.state_dict)

        return model

    def forward(
        self,
        x: torch.Tensor | List[str | Image.Image | np.ndarray],
        feat_layer: int = -1, #FUTURE: must be a backbone config, i.e., add to yaml and pass in __init__
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        infer_gs: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        #Predictions are two fold: da3 prediction output and global features from SALAD.
        # 1. Prepare input for da3
        image, extrinsics, intrinsics = self.backbone._prepare_inputs(x, extrinsics, intrinsics)

        if feat_layer == -1:
            feat_layer = self.backbone.dino_alt_start - 1
        assert feat_layer < self.backbone.dino_alt_start, "Double check what's the last layer before alternate attention"

        output = self.backbone.da3_inference(
            image,
            extrinsics,
            intrinsics,
            self.backbone.process_res,
            export_feat_layers=[feat_layer],
            export_depth=True,
            infer_gs=infer_gs,
            **kwargs
        )

        feats, cls = self.backbone._format_output_for_salad(output, feat_layer)
        global_descriptor = self.aggregator((feats, cls))

        output['descriptor'] = global_descriptor
        output.pop('aux')
        output.pop('aux_cls')

        return output

    @torch.inference_mode()
    def inference(
        self,
        x: torch.Tensor | List[str | Image.Image | np.ndarray],
        feat_layer: int = -1, #FUTURE: must be a backbone config, i.e., add to yaml and pass in __init__
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        infer_gs: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        output = self.forward(
            x, feat_layer, extrinsics, intrinsics, infer_gs, **kwargs
        )
        
        for k, v in output.items():
            output[k] = v.squeeze(0).cpu().numpy()
        output['conf'] = output.pop('depth_conf')
        
        return output
