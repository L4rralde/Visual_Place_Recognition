from typing import List, Dict, Iterable
from argparse import ArgumentParser
import random

import torch
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vpr.models.backbones.mapanything import load_pretrained_mapanything, mapanything_inference
from model_flavors.mapanything_salad import MapAnythingSalad, MapAnythingBackbone
from test_utils import ImgDirDataset


def parse_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument('img_dir', type=str)
    parser.add_argument('--num-seeds', type=int, default=10)
    args = parser.parse_args()

    return args


def compare_preds(baseline_preds, preds) -> float:
    acc_diff = 0.0
    for i, (ref_pred, pred) in enumerate(zip(baseline_preds, preds)):
        for k in ref_pred.keys():
            try:
                diff = (pred[k] - ref_pred[k]).abs().sum()
            except:
                diff = (pred[k] ^ ref_pred[k]).to(torch.float32).sum()
            if diff >= 1e-6:
                raise RuntimeError(f"{k} mismatch: {diff} in view {i}")
            acc_diff += diff
    return acc_diff


def compare_pipelines(
    baseline: torch.nn.Module,
    with_salad: MapAnythingSalad,
    as_backbone: MapAnythingBackbone,
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

        baseline_preds = mapanything_inference(baseline, selected_paths)
        with_salad_preds = with_salad.inference(selected_paths)

        try:
            acc_diff = compare_preds(baseline_preds, with_salad_preds)
        except RuntimeError as error:
            raise RuntimeError(f"Salad preds mismatch: {error}")

        asbackbone_preds = as_backbone.inference(selected_paths)
        try:
            acc_diff += compare_preds(baseline_preds, asbackbone_preds)
        except RuntimeError as error:
            raise RuntimeError(f"As backbone preds mismatch: {error}")

        status = "PASS" if acc_diff < 1e-6 else "FAIL"
        assert status == "PASS"

        print(f"seed: {seed}. diff: {acc_diff: .2f}. {status}") 


def main() -> None:
    args = parse_args()
    dataset = ImgDirDataset(args.img_dir)

    print("Loading mapanything...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mapanything = load_pretrained_mapanything().eval().to(device)
    print("Finished loading mapanything.")
    as_backbone = MapAnythingBackbone(mapanything)
    agg_args = {
        'num_clusters': 64,
        'cluster_dim': 128,
        'token_dim': 256
    }
    with_salad = MapAnythingSalad(mapanything, agg_args=agg_args).to(device)

    compare_pipelines(
        mapanything,
        with_salad,
        as_backbone,
        dataset,
        device,
        num_seeds=args.num_seeds
    )


if __name__ == '__main__':
    main()
