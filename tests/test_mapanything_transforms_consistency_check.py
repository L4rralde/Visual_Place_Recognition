from argparse import ArgumentParser
import random

import numpy as np
import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vpr.models.backbones.mapanything.transforms import get_transforms
from test_utils import ImgDirDataset
from vpr.models.backbones.mapanything.mapanything.utils.image import load_images
from vpr.models.backbones.mapanything.mapanything.utils.inference import(
    validate_input_views_for_inference,
    preprocess_input_views_for_inference,
)


def parse_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument('img_dir', type=str)
    parser.add_argument('--num-seeds', type=int, default=10)
    args = parser.parse_args()

    return args


def compare_transforms(dataset: ImgDirDataset, num_seeds: int=1, max_batch_size: int=10) -> None:
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
        
        if dataset.transform.mode == 'fixed_size':
            ref_views = load_images(selected_paths, resize_mode='fixed_size', size=dataset.transform.img_size)
        else:
            ref_views = load_images(selected_paths)
        
        validated_views = validate_input_views_for_inference(ref_views) #When only images are passed, this does nothing.
        # Transfer the views to the same device as the model
        ignore_keys = set(
            [
                "instance",
                "idx",
                "true_shape",
                "data_norm_type",
            ]
        )

        #When obly images are passed, this does view['image'] = view['image'].to(self.device)
        for view in validated_views: #Send some inputs to device
            for name in view.keys():
                if name in ignore_keys:
                    continue
                val = view[name]
                if name == "camera_poses" and isinstance(val, tuple): #Won't happen
                    view[name] = tuple(
                        x.to('cpu', non_blocking=True) for x in val
                    )
                elif hasattr(val, "to"): #Meh
                    view[name] = val.to('cpu', non_blocking=True)

        # Pre-process the input views
        processed_views = preprocess_input_views_for_inference(validated_views) #This one does not modify the images

        ref_rocessed_imgs = torch.cat([v['img'] for v in processed_views])

        diff = (ref_rocessed_imgs - t_selected_imgs).abs().sum()

        status = 'PASS' if diff < 1e-6 else 'FAIL'
        print(f"seed: {i}. diff={diff :.2f}. {status}")
        assert status == 'PASS'


def main() -> None:
    args = parse_args()
    input_config = {
        'img_size': 518,
        'mode': 'fixed_mapping'
    }
    _, val_t = get_transforms(input_config)
    dataset = ImgDirDataset(args.img_dir, val_t)

    print("Comparinfg with default mode: 'fixed_mapping'")
    compare_transforms(dataset, args.num_seeds)

    input_config = {
        'img_size': (322, 322),
        'mode': 'fixed_size'
    }
    _, val_t = get_transforms(input_config)
    dataset = ImgDirDataset(args.img_dir, val_t)

    print(f"Comparing with mode: 'fixed_size'")
    compare_transforms(dataset, args.num_seeds)


if __name__ == '__main__':
    main()