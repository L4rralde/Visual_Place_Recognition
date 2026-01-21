from typing import Tuple, Callable

from torchvision import transforms as T


def get_transforms() -> Tuple[Callable]:
    train_transform = T.Compose([
        T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])
    valid_transform = T.Compose([
        T.ToTensor(),
    ])

    return train_transform, valid_transform
