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
    parser.add_argument('img_dir')
    parser.add_argument('--batch-size', type=int, default=16)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_list = ImgDirDataset.scan_dir(args.img_dir)
    batch_size = args.batch_size
    assert len(img_list) > 200*batch_size
    
    device = "cuda"
    assert torch.cuda.is_available()
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
                predictions = model(images)
                ender.record()
                starters.append(starter)
                enders.append(ender)

        torch.cuda.synchronize()
    times = np.array([s.elapsed_time(e) for s, e in zip(starters, enders)])/1000.0
    print(times[100:-1].mean())


if __name__ == '__main__':
    main()
