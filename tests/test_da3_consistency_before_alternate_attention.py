import os, sys
import argparse
import random

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import vpr.models.backbones.da3.da3 as da3
from test_utils import load_da3_as_is, supported_configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir')
    args = parser.parse_args()
    return args


def run_randomized_consistency_test(model_instance, img_dir, layer_idx=-1, num_seeds=10):
    """
    Runs consistency tests comparing ALL shared images between two randomized batches.
    """
    if layer_idx == -1:
        layer_idx = model_instance.model.backbone.pretrained.alt_start - 1
    # Pool of available image indices (0 to 9)
    image_pool = list(range(10))
    
    print(f"### Starting Consistency Test over {num_seeds} Seeds ###")
    print(f"    Comparing ALL shared images in intersection.\n")

    for i in range(num_seeds):
        seed = i 
        random.seed(seed)
        np.random.seed(seed)
        
        # --- 1. Generate Valid Batches ---
        while True:
            # Randomize sizes (1 to 8)
            size_a = random.randint(1, 8) 
            size_b = random.randint(1, 8)
            
            # Sample indices
            indices_a = random.sample(image_pool, size_a)
            indices_b = random.sample(image_pool, size_b)
            
            set_a = set(indices_a)
            set_b = set(indices_b)
            intersection = list(set_a & set_b)

            # Constraint: Sets differ, but overlap exists
            if set_a != set_b and len(intersection) > 0:
                break
        
        # --- 2. Prepare Paths ---
        paths_a = [os.path.join(img_dir, f'frame_00000{x}.jpg') for x in indices_a]
        paths_b = [os.path.join(img_dir, f'frame_00000{x}.jpg') for x in indices_b]

        print(f"--- Seed {seed} ---")
        print(f"Batch A (Size {len(indices_a)}): {indices_a}")
        print(f"Batch B (Size {len(indices_b)}): {indices_b}")
        print(f"Shared Images (Intersection): {intersection}")

        # --- 3. Run Inference ---
        # Note: I added runner_module as an arg to avoid hardcoding 'da3'
        pred_a = da3.intermediate_features(
            model_instance, paths_a, 
            process_res=224, 
            export_feat_layers=[layer_idx]
        )
        
        pred_b = da3.intermediate_features(
            model_instance, paths_b, 
            process_res=224, 
            export_feat_layers=[layer_idx]
        )

        # --- 4. Compare All Shared Features ---
        key = f'feat_layer_{layer_idx}'
        assert key in pred_a.aux, "Key not found"

        total_seed_diff = 0.0
        
        print(f"  Results for {len(intersection)} shared images:")
        
        for shared_id in intersection:
            # Find position of this specific image in both batches
            idx_in_a = indices_a.index(shared_id)
            idx_in_b = indices_b.index(shared_id)

            feat_a = pred_a.aux[key][idx_in_a]
            feat_b = pred_b.aux[key][idx_in_b]

            # Calculate L1 Difference (Absolute Sum)
            # CRITICAL: We use .abs() so negative and positive errors don't cancel out
            diff = np.absolute((feat_a - feat_b)).sum()
            total_seed_diff += diff
            
            status = "OK" if diff < 1e-5 else "DIFF"
            assert status == "OK"
            #print(f"    Image {shared_id}: Diff {diff:.6f} [{status}]")

        # Summary for this seed
        overall_status = "PASS" if total_seed_diff < 1e-5 else "FAIL"
        assert overall_status == "PASS"
        print(f"  >> Seed Total Diff: {total_seed_diff:.6f} -> {overall_status}")
        print("-" * 30)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    args = parse_args()
    img_dir = os.path.abspath(args.img_dir)

    for config in supported_configs:
        da3_as_is = load_da3_as_is(config).to(device)
        try:
            run_randomized_consistency_test(da3_as_is, img_dir, num_seeds=20)
        except AssertionError as e:
            print(f"Fail at model config: {config} with signature: {e}")
            sys.exit(0)
    


if __name__ == '__main__':
    main()
