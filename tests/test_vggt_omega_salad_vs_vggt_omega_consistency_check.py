from argparse import ArgumentParser
import random

from test_vggt_omega_vs_vggt_omega_dino_consistency_check import(
    compare_pipelines
)
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vpr.models.backbones.vggt_omega import load_pretrained_vggt_omega
from model_flavors.vggt_omega_salad import VggtOmegaSalad
from test_utils import ImgDirDataset


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_dir', type=str)
    parser.add_argument('--ckpt', type=str, help="Path to VGGTOmega Checkpoint", required=True)
    parser.add_argument('--mode', choices=['balanced', 'max_size'], default='balanced')
    parser.add_argument('--num-seeds', type=int, default=10)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    dataset = ImgDirDataset(args.img_dir)

    print("Loading VGGTOmega...")
    vggt_omega = load_pretrained_vggt_omega(args.ckpt).eval().to("cuda")

    print("Finished loading VGGTOmega")

    backbone_args = {
        'probing_from_layer': random.randint(0, 23),
        'norm_layer': random.choice([True, False])
    }
    agg_args = {
        'num_clusters': 64,
        'cluster_dim': 128,
        'token_dim': 256
    }
    vggt_omega_salad = VggtOmegaSalad(
        vggt_omega,
        backbone_args,
        agg_args
    ).eval().to("cuda")

    compare_pipelines(
        vggt_omega,
        vggt_omega_salad,
        dataset,
        mode=args.mode,
        num_seeds=args.num_seeds
    )


if __name__ == '__main__':
    main()


