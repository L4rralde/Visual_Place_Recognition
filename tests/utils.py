import os, sys
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import vpr.models.backbones.da3.da3 as da3
from vpr.models.backbones.da3.depth_anything_3.api import DepthAnything3

supported_configs = {'SMALL', 'BASE', 'LARGE', 'GIANT'}


def load_da3_as_is(config_name: str='BASE'):
    config_name = config_name.upper()
    if not config_name in supported_configs:
        raise ValueError(f"Configuration {config_name} is not supported. Try one of the followings: {supported_configs}")
    da3 = DepthAnything3.from_pretrained(f"depth-anything/DA3-{config_name}")
    freeze_model(da3)

    return da3


def load_da3_dino(config_name: str='BASE', return_token: bool=True, process_res: int=252):
    if not config_name in supported_configs:
        raise ValueError(f"Configuration {config_name} is not supported. Try one of the followings: {supported_configs}")
    da3_dino = da3.DepthAnything3Dino(
        model_name=f'da3-{config_name.lower()}',
        return_token=return_token,
        process_res=process_res
    )
    freeze_model(da3_dino)

    return da3_dino


def freeze_model(model) -> None:
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

