from typing import List, Dict
import gc

from PIL import Image
import torch
import torch.nn as nn
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vpr.models.backbones.da3 import DepthAnything3Dino, da3_from_pretained
from vpr.models import SALAD
from utils import LightningLog


class DA3Salad(nn.Module):
    def __init__(
        self,
        da3: object,
        backbone_args: dict={},
        agg_args: dict={}
    ) -> None:
        super().__init__()
        self.backbone: DepthAnything3Dino = DepthAnything3Dino(da3, **backbone_args)
        self.aggregator: SALAD = SALAD(
            num_channels=self.backbone.num_channels,
            **agg_args
        )
        assert self.backbone.num_channels == self.aggregator.num_channels, "Uncompatible selection of backbone and aggregator"

    @staticmethod
    def from_lightning_log(path: str) -> "DA3Salad":
        log = LightningLog(path)
        assert log.agg_arch == "SALAD", "By the moment only SALAD is supported"
        da3 = da3_from_pretained(log.backbone_arch)
        model = DA3Salad(
            da3,
            log.backbone_config,
            log.agg_config
        )
        "We shouldn't save all weights, just salad's"
        model.load_state_dict(log.state_dict)

        return model

    def forward(
        self,
        x: torch.Tensor | List[str | Image.Image | np.ndarray],
        feat_layer: int = -1, #FUTURE: must be a backbone config, i.e., add to yaml and pass in __init__
        process_res: int = -1,
        infer_gs: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        #Predictions are two fold: da3 prediction output and global features from SALAD.
        # 1. Prepare input for da3
        image = self.backbone._prepare_inputs(x)
        if feat_layer == -1:
            feat_layer = self.backbone.dino_alt_start - 1
        assert feat_layer < self.backbone.dino_alt_start, "Double check what's the last layer before alternate attention"

        if process_res == -1:
            if isinstance(image[0], np.ndarray):
                H, W, _ = image[0].shape #FIXME. Error. image list may be a image of paths or a list of Image.Image. 
                                        #Not always a np.ndarray.
                                        #By the moment process_res =-1 only when a list of ndarrays are passed
            elif isinstance(image[0], Image.Image):
                W, H = image[0].size
            elif isinstance(image[0], str):
                raise RuntimeError("Need to provide a valid process resolution if a list of paths is passsed")
            process_res = max(H, W)

        output = self.backbone.da3_inference(
            image,
            process_res,
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
        process_res: int = -1,
        infer_gs: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        output = self.forward(
            x,
            feat_layer,
            process_res,
            infer_gs,
            **kwargs
        )
        
        for k, v in output.items():
            if not isinstance(v, torch.Tensor):
                continue
            output[k] = v.squeeze(0).cpu().numpy()
            del v
            gc.collect()
        torch.cuda.empty_cache()


        output['conf'] = output.pop('depth_conf')
        
        return output
