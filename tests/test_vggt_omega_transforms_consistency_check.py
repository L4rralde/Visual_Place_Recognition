from argparse import ArgumentParser
import random

import numpy as np
import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vpr.models.backbones.vggt_omega.transforms import(
    get_transforms
)
from vpr.models.backbones.vggt_omega.vggt_omega.utils.load_fn import(
    load_and_preprocess_images
)
from test_utils import ImgDirDataset


#Everythin is all right as long as all images are of the same aspect ration

def parse_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument('img_dir', type=str)
    parser.add_argument('--num-seeds', type=int, default=10)
    parser.add_argument('--mode', type=str, choices=['balanced', 'max_size'], default='balanced')
    args = parser.parse_args()

    return args


def compare_transforms(
    dataset: object,
    num_seeds: int=1,
    max_batch_size: int=10,
    **kwargs
) -> None:
    for i in range(num_seeds):
        seed = i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        curr_max = min(max_batch_size, len(dataset))
        batch_size = random.randint(1, curr_max)
        selected_imgs = [
            dataset[j]
            for j in random.sample(range(len(dataset)), batch_size)
        ]

        selected_paths = [path for _, path in selected_imgs]
        t_selected_imgs = torch.cat([img.unsqueeze(0) for img, _ in selected_imgs])

        mode = kwargs.get('mode', 'balanced')
        ref_rocessed_imgs = load_and_preprocess_images(selected_paths, mode)
        diff = (ref_rocessed_imgs - t_selected_imgs).abs().sum()

        status = 'PASS' if diff < 1e-6 else 'FAIL'
        print(f"seed: {i}. diff={diff :.2f}. {status}")
        assert status == 'PASS'


def main() -> None:
    args = parse_args()
    input_config = {
        'img_size': 512,
        'mode': args.mode
    }
    _, val_t = get_transforms(input_config)
    dataset = ImgDirDataset(args.img_dir, val_t)

    compare_transforms(
        dataset,
        args.num_seeds,
        mode=args.mode
    )


if __name__ == '__main__':
    main()
