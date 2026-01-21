
from typing import Tuple, Callable

from torchvision import transforms as T


IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}


def get_transforms(input_config: dict) -> Tuple[Callable]:
    image_size = input_config['img_size']
    mean_std = input_config.get('mean_std', IMAGENET_MEAN_STD)
    train_transform = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=mean_std['mean'], std=mean_std['std']),
    ])
    valid_transform = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=mean_std['mean'], std=mean_std['std']),
    ])

    return train_transform, valid_transform
