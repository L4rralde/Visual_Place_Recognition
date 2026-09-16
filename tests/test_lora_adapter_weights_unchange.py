from typing import List, Dict, Iterable
from argparse import ArgumentParser
import random

import torch
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vpr.models.backbones.vggt.vggt_dino import load_pretrained_vggt
from vpr_model import VPRModel


def parse_args() -> Dict:
    parser = ArgumentParser()
    parser.add_argument('log_path', type=str)
    args = parser.parse_args()
    return args


def load_new_model(log_path: str) -> VPRModel:
    model = VPRModel.from_lightning_log(log_path)

    return model


def get_nested_attr(obj, path):
    src_obj = obj
    for attr in path.split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            raise RuntimeError(f"{src_obj.__class__.__name__} has no attribute: {path}")
    return obj


def are_state_dicts_equal(dict1, dict2):
    # Compara la cantidad de llaves
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    
    # Compara cada tensor valor por valor
    for key in dict1:
        if not torch.equal(dict1[key], dict2[key]):
            print(f"key {key} mismatch")
            return False
    return True


def check_unchange_frozen_params(ref_model: torch.nn.Module, new_model: VPRModel) -> None:
    def check_backbone(
        ref_model: torch.nn.Module,
        new_model: VPRModel,
        ref_model_encoder_namespace: str='aggregator.patch_embed',
        new_model_encoder_namespace: str='backbone.dino'
    ) -> None:
        print("Comparing the backbones of both models")
        ref_model_encoder: torch.nn.Module = get_nested_attr(ref_model, ref_model_encoder_namespace)
        new_model_encoder: torch.nn.Module = get_nested_attr(new_model, new_model_encoder_namespace)

        match = are_state_dicts_equal(
            ref_model_encoder.state_dict(),
            new_model_encoder.state_dict()
        )

        if not match:
            raise RuntimeError("Not equals")

        print("Backbones match")

    def check_lora_adapter(
        ref_model: torch.nn.Module,
        new_model: VPRModel,
        ref_model_encoder_blocks_namespace: str = 'aggregator.patch_embed.blocks',
    ) -> None:
        ref_model_encoder_blocks = get_nested_attr(ref_model, ref_model_encoder_blocks_namespace)
        new_model_adapter_blocks = new_model.backbone.adapter.blocks

        adapter_depth = new_model.backbone.adapter_depth
        probing_layer = new_model.backbone.probing_from_layer

        for i in range(adapter_depth):
            ref_blk: torch.nn.Module = getattr(ref_model_encoder_blocks, str(probing_layer+i+1))
            ref_blk_state_dict = ref_blk.state_dict()
            new_blk: torch.nn.Module = get_nested_attr(new_model_adapter_blocks, f"{i}.base_model.model")
            new_blk_state_dict_no_lora = {}
            for k, v in new_blk.state_dict().items():
                if "lora_" in k:
                    continue
                if 'base_layer.' in k:
                    k = k.replace('base_layer.','')
                new_blk_state_dict_no_lora[k] = v

            print(f"Comparing adapter's blk {i} against ref model encoder blk {probing_layer+i+1}")
            match = are_state_dicts_equal(
                ref_blk_state_dict,
                new_blk_state_dict_no_lora
            )
            print("Blocks match")

            if not match:
                raise RuntimeError("Adapters not equal.")

    check_backbone(ref_model, new_model)
    check_lora_adapter(ref_model, new_model)


def main():
    args = parse_args()

    print("Checking weights (that should not) were kept frozen")
    print(f"Model path: {args.log_path}")
    new_model = load_new_model(args.log_path)

    encoder = new_model.encoder_arch.upper()
    assert encoder  == 'VGGT'

    print(f"Loading base model: {encoder}")
    ref_model = load_pretrained_vggt()

    check_unchange_frozen_params(
        ref_model,
        new_model
    )

    print("PASS")


if __name__ == '__main__':
    main()