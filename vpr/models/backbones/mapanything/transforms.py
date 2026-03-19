import os
from typing import List, Tuple, Callable

import PIL
from PIL.ImageOps import exif_transpose
from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
import torchvision.transforms as tvf
import numpy as np

import sys
sys.path.append(os.path.dirname(__file__))
from mapanything.utils.image import find_closest_aspect_ratio
from mapanything.utils.cropping import crop_resize_if_necessary


def preprocess_images(
    img_list: List[PIL.Image.Image]|PIL.Image.Image,
    size: int=518,
    resize_mode: str="fixed_mapping"
    ):
    patch_size = 14
    valid_resize_modes = {"fixed_mapping", "fixed_size"}
    if resize_mode not in valid_resize_modes:
        raise ValueError(
            f"Resize_mode must be one of {valid_resize_modes}, got '{resize_mode}'"
        )

    elif resize_mode == "fixed_size":
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError(
                f"Size must be a tuple/list of (width, height) for resize_mode='fixed_size', got {size}"
            )
        if not all(isinstance(x, int) for x in size):
            raise ValueError(
                f"Size values must be integers for resize_mode='fixed_size', got {size}"
            )

    is_single_img = not isinstance(img_list, list)
    if is_single_img:
        img_list = [img_list]
    norm_type = 'dinov2'
    
    aspect_ratios = []
    loaded_images = []
    for raw_img in img_list:
        img = exif_transpose(raw_img.convert("RGB"))
        W1, H1 = img.size
        aspect_ratios.append(W1/H1)
        loaded_images.append((img, H1, W1))

    average_aspect_ratio = sum(aspect_ratios)/len(aspect_ratios)

    if resize_mode == "fixed_mapping":
        target_width, target_height = find_closest_aspect_ratio(
            average_aspect_ratio, size
        )
        target_size = (target_width, target_height)
    elif resize_mode == "fixed_size":
        # Use exact size provided, aligned to patch_size
        target_size = (
            (size[0] // patch_size) * patch_size,
            (size[1] // patch_size) * patch_size,
        )
    else:
        RuntimeError("Something went wrong. Unsopported resize_mode")
    
    img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
    ImgNorm = tvf.Compose(
        [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
    )

    imgs = []
    for img, W1, H1 in loaded_images:
        # Resize and crop the image to the target size
        img = crop_resize_if_necessary(img, resolution=target_size)[0]

        imgs.append(ImgNorm(img)[None])
    if is_single_img:
        return imgs[0]
    return imgs


class MapAnythingTransform:
    def __init__(self, img_size: int=518, mode: str="fixed_mapping"):
        self.img_size = img_size
        self.mode = mode
    
    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return preprocess_images(img, self.img_size, self.mode)


def get_transforms(input_config: dict) -> Tuple[Callable, Callable]:
    img_size = input_config.get('img_size', 518)
    mode = input_config.get('mode', 'fixed_mapping')
    if mode == 'fixed_mapping' and not isinstance(img_size, int):
        raise ValueError("When using mode 'fixed_mapping', a single integer is expected")
    if mode == 'fixed_size':
        if not isinstance(img_size, (tuple, list)) or not len(img_size) == 2:
            raise ValueError("When using mode 'fixed_size', a list of two integers is expected")
    
    resize = MapAnythingTransform(img_size, mode) #Already transforms to tensor.
    train_transform = tvf.Compose([
        resize, #Bug: https://github.com/L4rralde/Visual_Place_Recognition/issues/1?reload=1
        tvf.RandAugment(num_ops=3, interpolation=tvf.InterpolationMode.BILINEAR),
    ])
    valid_transform = resize #Bug: https://github.com/L4rralde/Visual_Place_Recognition/issues/1?reload=1


    return train_transform, valid_transform
