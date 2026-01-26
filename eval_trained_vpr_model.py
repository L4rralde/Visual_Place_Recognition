import argparse
from typing import Callable

from vpr_model import VPRModel
from eval import model_eval
from vpr.models.helper import get_transforms


class Transforms:
    @staticmethod
    def get_transform(backbone: str, size: tuple) -> Callable:
        config = {
            'img_size': size
        }
        _, val_transform = get_transforms(backbone, config)
        return val_transform

    @staticmethod
    def dino_v2(size: tuple) -> Callable:
        return Transforms.get_transform('dino', size)

    @staticmethod
    def dino_v3(size: tuple) -> Callable:
        return Transforms.get_transform('dino', size)

    @staticmethod
    def da3(size: int|tuple) -> Callable:
        return Transforms.get_transform('da3', size)


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path', type=str)
    parser.add_argument('--img-size', type=int, nargs='+')

    #parser.add_argument('--yaml', type=str, default='') #FUTURE
    args = parser.parse_args()
    assert len(args.img_size) < 3, "Expected one or two numbers for image size"

    return args


def main() -> None:
    args = parse_args()
    model = VPRModel.from_lightning_log(args.log_path)
    img_size = args.img_size

    for size in img_size:
        assert size % model.backbone.PATCH_SIZE == 0, "Img size not divisible by patch size"

    if len(img_size) == 1:
        img_size = img_size[0]

    input_transform = Transforms.get_transform(
        model.encoder_arch,
        img_size
    )

    model_eval(model, input_transform, verbose=True)


if __name__ == '__main__':
    main()
