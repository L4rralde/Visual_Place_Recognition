from typing import Tuple
import sys, os
import argparse

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

VPR_REPO_PATH = os.path.dirname(os.path.dirname(__file__))
sys.path.append(VPR_REPO_PATH)
from vpr.models.helper import get_transforms
import hubconf
from benchmarking.benchmarking_utils import ImageDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('--batch-size', type=int, default=16)
    return parser.parse_args()


def get_descriptors_benchmark(
    model: torch.nn.Module,
    dataloader: DataLoader,
    warmup_batches: int,
    total_batches: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    starters = []
    enders = []

    descriptors = []
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.inference_mode():
        with torch.amp.autocast('cuda', dtype=dtype):
            for i, imgs in tqdm(enumerate(dataloader), total=total_batches, desc="Calculating descriptors"):
                imgs = imgs.to('cuda', non_blocking=True)
                if len(imgs.shape) == 4:
                    imgs = imgs.unsqueeze(0)

                if i < warmup_batches:  
                    patch_tokens = model.backbone.dino_forward(imgs)
                    feats, cls_token = model.backbone.prepare_tokens_for_salad(patch_tokens, imgs.shape)
                    output = model.aggregator((feats, cls_token))
                    descriptors.append(output.cpu())
                    continue

                if i == warmup_batches:
                    torch.cuda.synchronize()
                
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                
                patch_tokens = model.backbone.dino_forward(imgs)

                starter.record()
                feats, cls_token = model.backbone.prepare_tokens_for_salad(patch_tokens, imgs.shape)
                output = model.aggregator((feats, cls_token))
                ender.record()
                descriptors.append(output.cpu())
                
                starters.append(starter)
                enders.append(ender)

            # Synchronize ONCE at the end so the CPU/DataLoader isn't blocked during the loop
            torch.cuda.synchronize()

    descriptors = torch.cat(descriptors)
    
    if not starters:
        print("Not enough batches to calculate timing after warmup.")
        return

    times = np.array([s.elapsed_time(e) for s, e in zip(starters, enders)])/1000.0

    return descriptors, times


def main():
    args = parse_args()
    input_dir = os.path.realpath(args.input_dir)

    config = hubconf._vggt_l19_config
    _, transform = get_transforms(
            backbone_arch=config.backbone_arch,
            input_config={} #Use defaults
        )
    dataset = ImageDataset(input_dir, transform)
    
    # Cap workers to avoid memory thrashing on high-core machines
    num_workers = min(os.cpu_count()//2, 8)
    
    print(f"Dataset size: {len(dataset)} imgs")
    print(f"Batch size: {args.batch_size}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True # Ensures consistent batch sizes for accurate timing
    )

    model = hubconf.vggt_l19_salad(VPR_REPO_PATH)
    model = model.eval().to('cuda')

    # Enable benchmark since input size is static (322x322)
    torch.backends.cudnn.benchmark = True
    
    total_batches = len(dataloader)
    warmup_batches = total_batches // 10
    print("warmup_batches", warmup_batches)
    print("total batches", total_batches)
    print("batch size", args.batch_size)

    descriptors, times = get_descriptors_benchmark(
        model,
        dataloader,
        warmup_batches,
        total_batches
    )

    print(f"Average elapsed time: {times.mean()}")


if __name__ == '__main__':
    main()
