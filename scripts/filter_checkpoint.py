import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter a PyTorch checkpoint"
    )
    parser.add_argument(
        "input_checkpoint",
        help="Path to input checkpoint"
    )
    parser.add_argument(
        "output_checkpoint",
        help="Path to output checkpoint"
    )
    parser.add_argument(
        '--lora',
        action='store_true'
    )
    args = parser.parse_args()
    return args


def filter_checkpoint(input_path, output_path, filter_adapter_kw: str=''):
    # Load checkpoint
    checkpoint = torch.load(input_path, map_location="cpu")['state_dict']
    salad_state = {
        k: v for k, v in checkpoint.items()
        if k.startswith("aggregator")
    }
    adapter_state = {
        k: v for k,v in checkpoint.items()
        if k.startswith('backbone.adapter')
    }
    if filter_adapter_kw:
        filtered_adapter_state = {
            k: v for k,v in adapter_state.items()
            if filter_adapter_kw in k
        }
    else:
        filtered_adapter_state = adapter_state

    learned_state = {**salad_state, **filtered_adapter_state}
    print("Saving the following weights:")
    for k in learned_state.keys():
        print(k)
    
    # Save filtered checkpoint
    torch.save(learned_state, output_path)

    print(f"Filtered checkpoint saved to: {output_path}")


def main():
    args = parse_args()
    filter_adapter_kw = 'lora_' if args.lora else ''
    filter_checkpoint(args.input_checkpoint, args.output_checkpoint, filter_adapter_kw)


if __name__ == "__main__":
    main()