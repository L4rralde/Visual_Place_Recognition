from argparse import ArgumentParser

import torch
import numpy as np
from tqdm import tqdm

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vpr.models.backbones.vggt.vggt.utils.load_fn import load_and_preprocess_images
from vpr.models.backbones.vggt import load_pretrained_vggt
from tests.test_utils import ImgDirDataset


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_dirs', nargs='+')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--warmup-batches', type=int, default=50)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_list = []
    for img_dir in args.img_dirs:
        img_list += ImgDirDataset.scan_dir(img_dir)

    batch_size = args.batch_size
    warmup_batches = args.warmup_batches
    if not len(img_list) > 2*warmup_batches*batch_size:
        raise RuntimeError("Not enough images")

    img_list = img_list[:2*warmup_batches*batch_size]


    if not torch.cuda.is_available():
        raise RuntimeError("Cuda is required")
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = load_pretrained_vggt().eval().to(device)

    starters = []
    enders = []
    with torch.no_grad():
        torch.cuda.synchronize()
        for i in tqdm(range(0, len(img_list), batch_size)):
            image_names = img_list[i : i+batch_size]  
            images = load_and_preprocess_images(image_names).to(device)

            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)

            with torch.cuda.amp.autocast(dtype=dtype):
                starter.record()
                model(images)
                ender.record()
            starters.append(starter)
            enders.append(ender)


        torch.cuda.synchronize()
    times = np.array([s.elapsed_time(e) for s, e in zip(starters, enders)])/1000.0

    avg_inf_time = times[warmup_batches:-1].mean()
    fps = (batch_size) / avg_inf_time
    print(f"Average inference time for batch size {batch_size}: {avg_inf_time}")
    print(f"Throughput: {fps:.2f} img/s")

    image_names = img_list[:batch_size]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    images = load_and_preprocess_images(image_names).to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            model(images)

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    print(f"Peak allocated: {peak_allocated/1024**3:.2f} GB")
    print(f"Peak reserved: {peak_reserved / 1024**3:.2f} GB")


if __name__ == '__main__':
    main()
