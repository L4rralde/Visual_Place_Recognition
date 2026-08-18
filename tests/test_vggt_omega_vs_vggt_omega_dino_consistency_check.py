import os
import argparse
import random
from typing import Iterable, List, Dict

import torch
import numpy as np
from torchvision.transforms import v2 as T

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import vpr.models.backbones.vggt_omega as vggto
from test_utils import freeze_model, ImgDirDataset
from vpr.models.backbones.vggt_omega.vggt_omega.utils.load_fn import load_and_preprocess_images
from vpr.models.backbones.vggt_omega.vggt_omega.utils.pose_enc import encoding_to_camera


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir')
    parser.add_argument('--ckpt', type=str, help="Path to VGGTOmega Checkpoint", required=True)
    parser.add_argument('--mode', choices=['balanced', 'max_size'], default='balanced')
    parser.add_argument('--num-seeds', type=int, default=10)

    args = parser.parse_args()
    return args


def load_pretrained_vggt_omega(checkpoint):
    vggt_omega = vggto.load_pretrained_vggt_omega(checkpoint)
    freeze_model(vggt_omega)

    return vggt_omega


def vggt_omega_inference(
    model: torch.nn.Module,
    image_list: List[os.PathLike],
    mode: str,
    device: str='cuda'
) -> Dict[str, np.ndarray]:
    images = load_and_preprocess_images(image_list, mode=mode).to("cuda")

    with torch.inference_mode():
        predictions = model(images)

    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )

    predictions['extrinsic'] = extrinsics
    predictions['intrinsic'] = intrinsics

    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            predictions[key] = value.cpu().numpy().squeeze(0)

    return predictions


def compare_pipelines(
    vggt_omega: torch.nn.Module,
    vggt_omega_backbone: vggto.VggtOmegaBackbone,
    dataset: Iterable,
    device: str='cuda',
    max_batch_size: int=16,
    mode: str='balanced',
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

        ref_preds = vggt_omega_inference(
            vggt_omega,
            selected_paths,
            mode,
            device
        )

        other_preds = vggt_omega_backbone.inference(
            selected_paths,
            mode=mode
        )

        shared_keys = set([
            k
            for k in ref_preds.keys()
            if k in other_preds
        ])

        acc_diff = 0.0
        for key in shared_keys:
            diff = np.abs(ref_preds[key] - other_preds[key]).sum()
            if diff > 1e-6:
                raise RuntimeError(f"{key} mismatch: {diff}")
            acc_diff += diff

        status = "PASS" if acc_diff < 1e-6 else "FAIL"
        assert status == "PASS"

        print(f"seed: {seed}. diff: {acc_diff: .2f}. {status}")


def main():
    args = parse_args()
    dataset = ImgDirDataset(args.img_dir)

    print("Loading VGGTOmega...")
    vggt_omega = load_pretrained_vggt_omega(args.ckpt).to("cuda").eval()

    print("Finished loading VGGTOmega")

    backbone_args = {
        'probing_from_layer': random.randint(0, 23),
        'norm_layer': random.choice([True, False])
    }
    vggt_omega_backbone = vggto.VggtOmegaBackbone(
        vggt_omega,
        **backbone_args
    )

    compare_pipelines(
        vggt_omega,
        vggt_omega_backbone,
        dataset,
        mode=args.mode,
        num_seeds=args.num_seeds
    )


if __name__ == '__main__':
    main()
