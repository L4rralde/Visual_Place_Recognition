from typing import Tuple, Callable

from torchvision import transforms as T

import numpy as np
import cv2
from PIL import Image


class OptimizedDA3Resize:
    def __init__(self, target_size: int, patch_size: int = 14):
        self.target_size = target_size
        self.patch_size = patch_size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.target_size / float(max(w, h))
        
        # 1. Calcular dimensiones escaladas iniciales
        temp_w = w * scale
        temp_h = h * scale
        
        # 2. Redondear al múltiplo de patch_size más cercano
        # La fórmula: round(valor / patch) * patch
        new_w = int(round(temp_w / self.patch_size) * self.patch_size)
        new_h = int(round(temp_h / self.patch_size) * self.patch_size)
        
        # Asegurar que no queden en 0 (especialmente en imágenes muy estrechas)
        new_w = max(self.patch_size, new_w)
        new_h = max(self.patch_size, new_h)

        # 3. Selección de interpolación
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        
        # 4. Resize vía OpenCV (eficiente en RAM para 4K)

        img_np = np.array(img)
        resized_arr = cv2.resize(img_np, (new_w, new_h), interpolation=interp)

        return Image.fromarray(resized_arr)


def get_transforms(input_config: dict) -> Tuple[Callable]:
    img_size = input_config['img_size']
    if isinstance(img_size, int):
        resize = OptimizedDA3Resize(input_config['img_size'])
    elif isinstance(img_size, list) and len(img_size) == 2:
        #Same in dino
        resize = T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR)
    else:
        raise ValueError(f"Unexpected format for img_size: {img_size}")
    train_transform = T.Compose([
        resize, #Bug: https://github.com/L4rralde/Visual_Place_Recognition/issues/1?reload=1
        T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])
    valid_transform = T.Compose([
        resize, #Bug: https://github.com/L4rralde/Visual_Place_Recognition/issues/1?reload=1
        T.ToTensor(),
    ])

    return train_transform, valid_transform
