from typing import Tuple, Callable
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
from eval import(
    get_val_dataset,
    get_validation_recalls,
    MSLSTest
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--compute-metrics', action='store_true')
    return parser.parse_args()


def get_descriptors_n_profile(
    model: torch.nn.Module,
    dataloader: DataLoader,
    warmup_batches: int,
    profile_only: bool=False
) -> Tuple[torch.Tensor, float]:
    starters = []
    enders = []

    descriptors = []
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.inference_mode():
        with torch.amp.autocast('cuda', dtype=dtype):
            torch.cuda.synchronize()
            for i, (imgs, _) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating descriptors"):
                imgs = imgs.to('cuda', non_blocking=True)
                if len(imgs.shape) == 4:
                    imgs = imgs.unsqueeze(0)
                    
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                
                patch_tokens = model.backbone.dino_forward(imgs)

                starter.record()
                feats, cls_token = model.backbone.prepare_tokens_for_salad(patch_tokens, imgs.shape)
                output = model.aggregator((feats, cls_token))
                ender.record()

                starters.append(starter)
                enders.append(ender)
                if not profile_only:
                    descriptors.append(output.cpu())

            # Synchronize ONCE at the end so the CPU/DataLoader isn't blocked during the loop
            torch.cuda.synchronize()

    if not profile_only:
        descriptors = torch.cat(descriptors)
    
    if not starters:
        print("Not enough batches to calculate timing after warmup.")
        return

    times = np.array([s.elapsed_time(e) for s, e in zip(starters, enders)])/1000.0

    return descriptors, times[warmup_batches:-1].mean()


def model_eval_profile(
    model: torch.nn.Module,
    input_transform: Callable, 
    val_datasets: list,
    verbose: bool = False,
    batch_size: int = 32,
    profile_only: bool=False
):
    model = model.eval()
    model = model.to('cuda')
    torch.backends.cudnn.benchmark = True

    recalls = {}

    for val_name in val_datasets:
        val_dataset, num_references, num_queries, ground_truth = get_val_dataset(val_name, input_transform)
        val_loader = DataLoader(
            val_dataset,
            num_workers=8,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        total_batches = len(val_loader)
        warmup_batches = max(10, len(val_loader)//10)

        if verbose:
            print(f'Evaluating on {val_name}')
            print(f"Total number of batches: {total_batches}")
            print(f"Number of warmup batches: {warmup_batches}")

        descriptors, time = get_descriptors_n_profile(
            model=model,
            dataloader=val_loader,
            warmup_batches=warmup_batches,
            profile_only=profile_only
        )

        print(f"Average inference time of profiled module: {time}")
        if profile_only:
            continue
        
        if verbose:
            print(f'Descriptor dimension {descriptors.shape[1]}')
        r_list = descriptors[ : num_references]
        q_list = descriptors[num_references : ]

        if verbose:
            print('total_size', descriptors.shape[0], num_queries + num_references)

        testing = isinstance(val_dataset, MSLSTest) #Not enough information

        preds = get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=[1, 5, 10, 15, 20, 25],
            gt=ground_truth,
            print_results=True,
            dataset_name=val_name,
            faiss_gpu=False,
            testing=testing
        )

        del descriptors
        if verbose:
            print('========> DONE!\n\n')

        recalls[val_name] = preds

    return recalls


def main():
    args = parse_args()

    model = hubconf.vggt_l19_salad(VPR_REPO_PATH)
    model = model.eval().to('cuda')

    config = hubconf._vggt_l19_config
    _, transform = get_transforms(
        backbone_arch=config.backbone_arch,
        input_config={} #Use defaults
    )

    print(f"Batch size: {args.batch_size}")
    if args.compute_metrics:
        print("Computing vpr metrics alongside profiling time. Don't trust profiling time because of this.")
    else:
        print("Computing only profiling time without vpr metrics. This profiling time is more accurate.")
    model_eval_profile(
        model,
        transform,
        val_datasets=['pitts30k_test'],
        verbose=True,
        batch_size=args.batch_size,
        profile_only=not args.compute_metrics
    )


if __name__ == '__main__':
    main()
