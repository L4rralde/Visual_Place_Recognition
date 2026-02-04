from typing import List, Dict, Iterable
from argparse import ArgumentParser
import random

import torch
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vpr.models.backbones.vggt.vggt_dino import load_pretrained_vggt
from model_flavors.vggt_salad import VggtSalad
from submodules.vggt.vggt.utils.load_fn import load_and_preprocess_images
from submodules.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from test_utils import ImgDirDataset


def parse_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument('img_dir', type=str)
    parser.add_argument('--num-seeds', type=int, default=10)
    args = parser.parse_args()

    return args


def vggt_inference(model: object, image_names: List[str], device: str='cpu') -> Dict[str, np.ndarray]:
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Load and preprocess example images (replace with your own image paths)
    images = load_and_preprocess_images(image_names).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)
    
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"],
        images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    torch.cuda.empty_cache()

    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            predictions[key] = value.cpu().numpy().squeeze(0)

    return predictions


def compare_pipelines(
    vggt: torch.nn.Module,
    vggt_salad: VggtSalad,
    dataset: Iterable,
    device: str,
    max_batch_size: int=10,
    num_seeds: int=1
) -> None:
    for i in range(num_seeds):
        seed = i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
        curr_max = min(max_batch_size, len(dataset))
        batch_size = random.randint(1, curr_max)
        selected_paths = [
            dataset[j][1]
            for j in random.sample(range(len(dataset)), batch_size)
        ]

        vggt_preds = vggt_inference(vggt, selected_paths, device)
        vggt_salad_preds = vggt_salad.inference(selected_paths)

        shared_keys = set([
            k
            for k in vggt_preds.keys()
            if k in vggt_salad_preds
        ])
        shared_keys -= {'pose_enc', 'pose_enc_list'}

        acc_diff = 0.0
        for key in shared_keys:
            diff = np.abs(vggt_preds[key] - vggt_salad_preds[key]).sum()
            if diff > 1e-6:
                raise RuntimeError(f"{key} mismatch: {diff}")
            acc_diff += diff

        status = "PASS" if acc_diff < 1e-6 else "FAIL"
        assert status == "PASS"

        print(f"seed: {seed}. diff: {acc_diff: .2f}. {status}")


def main() -> None:
    args = parse_args()
    dataset = ImgDirDataset(args.img_dir)

    print("Loading vggt...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vggt = load_pretrained_vggt().to(device)
    print("Finished loading vggt.")
    backbone_args = {'return_token': True}
    agg_args = {
        'num_clusters': 64,
        'cluster_dim': 128,
        'token_dim': 256
    }
    vggt_salad = VggtSalad(vggt, backbone_args, agg_args).to(device)

    compare_pipelines(vggt, vggt_salad, dataset, device, num_seeds=args.num_seeds)


if __name__ == '__main__':
    main()
