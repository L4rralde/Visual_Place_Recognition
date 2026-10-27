from argparse import ArgumentParser

import torch

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('ckpt')
    parser.add_argument('output')
    parser.add_argument('--lora', action='store_true')
    return parser.parse_args()


def filter_state_dict(state_dict, lora=False):

    adapter_ckpt = {
        k:v for k,v in state_dict.items()
        if k.startswith('backbone.adapter')
    }

    if lora:
        adapter_ckpt = {
            k:v for k,v in adapter_ckpt.items()
            if 'lora' in k
        }

    agg_ckpt = {
        k:v for k,v in state_dict.items()
        if k.startswith('aggregator')
    }

    filtered_ckpt = {**adapter_ckpt, **agg_ckpt}

    return filtered_ckpt


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, weights_only=False)
    state_dict = ckpt['state_dict']
    filtered_ckpt = filter_state_dict(state_dict, args.lora)

    print("Weights to store:")
    for k in filtered_ckpt.keys():
        print(k)

    num_state_dict = sum(p.numel() for p in filtered_ckpt.values())
    print(f"Total number of parameres (M): {num_state_dict/10**6}")

    torch.save(filtered_ckpt, args.output)


if __name__ == '__main__':
    main()

