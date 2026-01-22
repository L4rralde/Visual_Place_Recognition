import os
from typing import List, Callable
import glob
import argparse
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vpr.models.backbones.da3 import get_transforms as da3_get_transforms
import vpr.models.backbones.da3.da3 as da3
from utils import load_da3_as_is, load_da3_dino, supported_configs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir')
    parser.add_argument('--img-size', default=252)
    args = parser.parse_args()
    return args


class ImgDirDataset(Dataset):
    def __init__(self, img_dir: str, transform: Callable|None = None):        
        self.img_pahts = ImgDirDataset.scan_dir(img_dir)
        assert len(self.img_pahts) > 0, "Found no valid image"
        self.transform = transform

    @staticmethod
    def scan_dir(img_dir: str) -> List[str]:
        valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        all_img_paths = []
        for ext in valid_exts:
            all_img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        
        return all_img_paths

    def __len__(self) -> int:
        return len(self.img_pahts)

    def __getitem__(self, index) -> Image.Image:
        img_path = self.img_pahts[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_path


def compare_pipelines_consistency(
    da3_dino_fn,         # The function that accepts tensors: da3_dino
    model_instance,      # The model passed to intermediate_features (da3_as_is)
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
        selected_tensors = [tensor for tensor, _ in selected_imgs]
        
        print(f"--- Seed {seed} | Batch Size: {batch_size} ---")
        print([os.path.basename(path) for path in selected_paths])

        # 2. Stack Tensors
        img_tensor = torch.stack(selected_tensors).to(da3_dino_fn.da3.device)
        
        # 3. Run Inference
        # Assumption: da3_dino returns a dictionary matching the aux structure 
        # or we need to access the specific key.
        with torch.no_grad():
            out_a = da3_dino_fn(img_tensor)
        
        feat_a, _ = out_a
        feat_a = feat_a.permute(0, 2, 3, 1).cpu().numpy()
            
        # --- Pipeline B: Path Input (intermediate_features) ---
        # We need to parse the layer index from the string 'feat_layer_3' -> 3
        # or pass it explicitly. Assuming we want layer 3 here.
        out_b = da3.intermediate_features(
            model_instance, 
            selected_paths,
            process_res=img_size, 
        )
        
        feat_b = out_b.aux[f'feat_layer_{model_instance.model.backbone.pretrained.alt_start - 1}']

        # --- Comparison ---
        # Both output tensors should be [Batch, Channels, H, W]
        
        # 1. Check Shape
        assert feat_a.shape == feat_b.shape, "Shape mismatch"
        
        # 2. Check Values (L1 Difference)
        abs_diff = np.absolute(feat_a - feat_b).sum()
        
        # 3. Report
        # We use a slightly higher tolerance (1e-4) here because 
        # different loading pipelines (PIL internal vs C++ loaders) 
        # can sometimes cause tiny pixel intensity shifts.
        status = "PASS" if abs_diff < 1e-4 else "FAIL"
        
        print(f"  Total L1 Diff: {abs_diff:.6f} -> {status}")
        
        # Optional: Print per-image breakdown if it fails
        assert status == 'PASS'

        print("-" * 30)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    args = parse_args()

    img_size = args.img_size

    input_config = {'img_size': img_size}
    #train_transform, valid_transform = da3_get_transforms(input_config)
    valid_transform = T.ToTensor()

    img_dir = os.path.abspath(args.img_dir)
    dataset = ImgDirDataset(img_dir, valid_transform)
    
    for config in supported_configs:
        da3_as_is = load_da3_as_is(config).to(device)
        da3_dino = load_da3_dino(config, process_res=img_size).to(device)
        try:
            compare_pipelines_consistency(da3_dino, da3_as_is, dataset, img_size, num_seeds=100)
        except AssertionError as e:
            print(f"Model {config} FAIL due to {e}")
            sys.exit(0)


if __name__ == '__main__':
    main()
