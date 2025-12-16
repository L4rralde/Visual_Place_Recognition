
from .salad import SALAD
from .dinov2 import DINOv2
from .dinov3 import DINOv3


def get_backbone(
        backbone_arch='dinov2',
        backbone_config={}
    ):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'resnet50'.
        backbone_config (dict, optional): this must contain all the arguments needed to instantiate the backbone class. Defaults to {}.

    Returns:
        nn.Module: the backbone as a nn.Model object
    """
    if 'dinov2' in backbone_arch.lower():
        return DINOv2(model_name=backbone_arch, **backbone_config)
    elif 'dinov3' in backbone_arch.lower():
        return DINOv3(model_name=backbone_arch, **backbone_config)
    else:
        raise ValueError(f"Backbone {backbone_arch} not supported")
