import sys, os
import argparse
import random

import torch
import numpy as np
import torchvision.transforms as T

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import vpr.models.backbones.da3.da3 as da3
from utils import load_da3_as_is, supported_configs, ImgDirDataset, freeze_model
from model_flavors.da3_salad import DA3Salad


def load_da3_salad(config: str):
    da3 = da3.da3_from_pretained(config)
    backbone_args = {
        'frozen': True,
        'return_token': True,
    }
    agg_args = {
        'num_clusters': 64,
        'cluster_dim': 128,
        'token_dim': 256
    }
    da3_salad = DA3Salad(da3, backbone_args, agg_args)
    freeze_model(da3_salad)
    return da3_salad


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir')
    parser.add_argument('img_size', type=int, default=504)
    args = parser.parse_args()
    return args


def compare_pipelines_consistency(
    da3_salad: DA3Salad,         # The function that accepts tensors: da3_dino
    da3_as_is,      # The model passed to intermediate_features (da3_as_is)
    dataset,
    img_size=252,
    num_seeds=5, 
    max_batch_size=20
):
    """
    Compares two pipelines (Tensor Input vs Path Input) to ensure feature equality.
    
    Pipeline A: Manual PIL Load -> ToTensor -> da3_dino()
    Pipeline B: Paths -> da3.intermediate_features()
    """

    for i in range(num_seeds):
        seed = i
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # --- Random Batch Generation ---
        # Size between 1 and 20 (or max available if less than 20)
        curr_max = min(max_batch_size, len(dataset))
        batch_size = random.randint(1, curr_max)
        
        selected_imgs = [
            dataset[i]
            for i in random.sample(range(len(dataset)), batch_size)
        ]
        selected_paths = [path for _, path in selected_imgs]
        
        print(f"--- Seed {seed} | Batch Size: {batch_size} ---")
        print([os.path.basename(path) for path in selected_paths])

        # 2. Run Inference
        # Assumption: da3_dino returns a dictionary matching the aux structure 
        # or we need to access the specific key.
        with torch.no_grad():
            out = da3_salad.inference(selected_imgs, process_res=img_size)
            out_golden = da3_as_is.inference(selected_imgs, process_res=img_size)

        shared_keys = {}
        for k in out.keys():
            if out_golden.__getattribute__(k, None) is None:
                continue
            shared_keys.append(k)

        assert len(shared_keys) > 0, "No shared predictions"
        total_diff = 0
        for k in shared_keys:
            src = out[k]
            ref = out[k]
            abs_diff = np.absolute(src, ref).sum()
            assert abs_diff < 1e-6, "Mismatch"
            total_diff += abs_diff

        # 3. Report
        # We use a slightly higher tolerance (1e-4) here because 
        # different loading pipelines (PIL internal vs C++ loaders) 
        # can sometimes cause tiny pixel intensity shifts.
        status = "PASS" if total_diff < 1e-6 else "FAIL"
        
        print(f"  Total L1 Diff: {abs_diff:.6f} -> {status}")
        
        # Optional: Print per-image breakdown if it fails
        assert status == 'PASS'

        print("-" * 30)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    args = parse_args()
    img_dir = os.path.abspath(args.img_dir)
    dataset = ImgDirDataset(img_dir)
    
    for config in supported_configs:
        da3_as_is = load_da3_as_is(config).to(device)
        da3_salad = load_da3_salad(config).to(device)
        try:
            compare_pipelines_consistency(
                da3_salad,
                da3_as_is, 
                dataset,
                args.img_size,
                num_seeds=100
            )
        except AssertionError as e:
            print(f"Model {config} FAIL due to {e}")
            sys.exit(0)


if __name__ == '__main__':
    main()
