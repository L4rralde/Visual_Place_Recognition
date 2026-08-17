from typing import Tuple, Callable, Optional

from PIL import Image
import torch
from torchvision.transforms import v2 as T

import sys, os
sys.path.append(os.path.dirname(__file__))
from vggt_omega.utils.load_fn import(
    _crop_to_supported_aspect_ratio,
    _balanced_target_shape,
    _max_size_target_shape
)


def _make_rgb_image(image: Image) -> Image:
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("RGB")


def preprocess_image(
    image: Image.Image,
    mode: str='balanced',
    image_resolution: int=512,
    patch_size: int=16
) -> Image.Image:
    image = _crop_to_supported_aspect_ratio(_make_rgb_image(image))
    width, height = image.size
    aspect_ratio = height / max(width, 1)
    if mode == "balanced":
        target_h, target_w = _balanced_target_shape(aspect_ratio, image_resolution, patch_size)
    else:
        target_h, target_w = _max_size_target_shape(aspect_ratio, image_resolution, patch_size)

    image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)

    return image


class VggtOmegaTransform:
    def __init__(self, img_resolution: int=512, mode: str="balanced") -> None:
        self.img_resolution = img_resolution
        self.mode = mode

    def __call__(self, img: Image.Image, mode: Optional[str] = None) -> Image.Image:
        if mode is None:
            mode = self.mode
        return preprocess_image(img, mode, self.img_resolution)


def get_transforms(input_config: dict) -> Tuple[Callable]:
    img_size = input_config.get('img_size', 512)
    mode = input_config.get('mode', 'balanced')
    if isinstance(img_size, int):
        resize = VggtOmegaTransform(img_size, mode)
    elif isinstance(img_size, list) and len(img_size) == 2:
        #same in dino
        resize = T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR)
    else:
        raise ValueError(f"Unexpected format for img_size: {img_size}")

    train_transform = T.Compose([
        resize,
        T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])
    valid_transform = T.Compose([
        resize,
        T.ToTensor(),
    ])

    return train_transform, valid_transform