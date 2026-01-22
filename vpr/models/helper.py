
#Function from dinov2+salad repo with modifications
from typing import Tuple, Callable

from torchvision import transforms as T


def get_backbone(
        backbone_arch: str='dinov2',
        backbone_config: dict={}
    ):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'dinov2'.
        backbone_config (dict, optional): this must contain all the arguments needed to instantiate the backbone class. Defaults to {}.

    Returns:
        nn.Module: the backbone as a nn.Model object
    """
    freeze = backbone_config.pop('frozen', False)

    if 'dinov2' in backbone_arch.lower():
        from .backbones.dinov2 import DINOv2
        backbone = DINOv2(model_name=backbone_arch, **backbone_config)
    elif 'dinov3' in backbone_arch.lower():
        from .backbones.dinov3 import DINOv3
        backbone = DINOv3(model_name=backbone_arch, **backbone_config)
    elif 'da3' in backbone_arch.lower():
        from .backbones.da3 import DepthAnything3Dino
        backbone = DepthAnything3Dino(backbone_arch, **backbone_config)
    else:
        raise ValueError(f"Backbone {backbone_arch} not supported")

    if freeze:
        freeze_model(backbone)

    return backbone


def get_transforms(backbone_arch: str, input_config: dict) -> Tuple[Callable]:
    is_dino = backbone_arch.lower().startswith('dino')
    if is_dino:
        from .backbones.dino_transforms import get_transforms as dino_get_transforms
        train_transform, valid_transform = dino_get_transforms(input_config)
    elif 'da3' in backbone_arch.lower():
        from .backbones.da3 import get_transforms as da3_get_transforms
        train_transform, valid_transform = da3_get_transforms(input_config)
    else:
        raise ValueError(f"Backbone {backbone_arch} not supported")

    return train_transform, valid_transform


DINO_EMBEDDING_DIMS = {
    'small': 384,
    'base': 768,
    'large': 1024,
    'giant': 1536
}


def freeze_model(model) -> None:
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
