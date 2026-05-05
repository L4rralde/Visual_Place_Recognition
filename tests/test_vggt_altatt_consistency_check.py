import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from random import shuffle
from argparse import ArgumentParser


import torch

from model_flavors.vggt_salad import VggtSalad
from hubconf import vggt_salad
from submodules.vggt.vggt.utils.load_fn import load_and_preprocess_images

def parse_args() -> dict:
    parser = ArgumentParser()
    parser.add_argument('--num-seeds', type=int, default=10)
    args = parser.parse_args()

    return args

cimat_video_frames_path = os.path.join(
    os.path.dirname(__file__),
    "samples",
    "cimat_video"
)

def main():
    args = parse_args()
    device = (
        'cuda' if torch.cuda.is_available()
        else 'cpu'
    )
    model: VggtSalad = vggt_salad('..').eval().to(device)

    for i in range(args.num_seeds):
        img_path_list = [
            os.path.join(cimat_video_frames_path, f"frame_00000{i}.jpg")
            for i in range(10)
        ]
        shuffle(img_path_list)

        img_list_a = list(img_path_list[2:])
        shuffle(img_list_a)
        img_list_a = img_path_list[0:2] + img_list_a
        img_list_b = list(img_path_list[2:])
        shuffle(img_list_b)
        img_list_b = img_path_list[0:2] + img_list_b
        img_list_b = img_list_b[:5]

        shared_imgs = [
            img_path for img_path in img_list_a
            if img_path in img_list_b
        ]
        idcs_a = [
            img_list_a.index(img)
            for img in shared_imgs
        ]
        idcs_b = [
            img_list_b.index(img)
            for img in shared_imgs
        ]

        imgs_a = load_and_preprocess_images(img_list_a).to(device)
        if len(imgs_a.shape) == 4:
            imgs_a = imgs_a.unsqueeze(0)
        
        imgs_b = load_and_preprocess_images(img_list_b).to(device)
        if len(imgs_b.shape) == 4:
            imgs_b = imgs_b.unsqueeze(0)

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(device, dtype=dtype):
                patch_tokens_a = model.backbone.dino_forward(imgs_a)
                if isinstance(patch_tokens_a, dict):
                    patch_tokens_a = patch_tokens_a["x_norm_patchtokens"]

                patch_tokens_b = model.backbone.dino_forward(imgs_b)
                if isinstance(patch_tokens_b, dict):
                    patch_tokens_b = patch_tokens_b["x_norm_patchtokens"]

        
                assert (patch_tokens_a[idcs_a] - patch_tokens_b[idcs_b]).abs().sum() < 1e-9
                
                del imgs_b, patch_tokens_b

                imgs_b = imgs_a[:, [0, 1, 4, 6]].clone()
                imgs_a = imgs_a[:, [0, 1, 8, 6, 9, 1]]
                patch_tokens_b = patch_tokens_a[[0, 1, 4, 6]].clone()
                patch_tokens_a = patch_tokens_a[[0, 1, 8, 6, 9, 1]]
                
                img_pairs_a, paired_tokens_a = model.backbone.pair_patch_tokens_with_ref(
                    imgs_a, patch_tokens_a
                )
                tokens_list_a, start_idx_a = model.backbone.alternate_attention(
                    img_pairs_a, paired_tokens_a
                )
                img_pairs_b, paired_tokens_b = model.backbone.pair_patch_tokens_with_ref(
                    imgs_b, patch_tokens_b
                )
                tokens_list_b, start_idx_b = model.backbone.alternate_attention(
                    img_pairs_b, paired_tokens_b
                )

                assert start_idx_a == start_idx_b
                for token_a, token_b in zip(tokens_list_a, tokens_list_b):
                    diff = (token_a[[0, 2]] - token_b[[0, 2]]).abs().sum()
                    assert diff < 1e-9
                    diff = (token_a[0] - token_a[-1]).abs().sum()
                    assert diff < 1e-9

        print(f"{i+1}. PASS")


if __name__ == '__main__':
    main()
