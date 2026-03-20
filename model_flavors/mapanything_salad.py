from typing import List, Dict
import gc

import torch
import torch.nn as nn
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from vpr.models.backbones.mapanything import(
    MapAnythingBackbone,
    load_pretrained_mapanything,
    load_images
)
from vpr.models import SALAD
from utils import LightningLog
from vpr.models.backbones.mapanything.mapanything.utils.inference import(
    validate_input_views_for_inference,
    preprocess_input_views_for_inference,
    postprocess_model_outputs_for_inference
)


class MapAnythingSalad(nn.Module):
    def __init__(self, mapanything: object, backbone_args: dict={}, agg_args: dict={}):
        super().__init__()
        self.backbone = MapAnythingBackbone(mapanything, **backbone_args)
        self.aggregator = SALAD(
            num_channels=self.backbone.num_channels,
            **agg_args
        )
    
    @classmethod
    def from_lightning_log(cls, path: str, mapanything: object|None):
        log = LightningLog(path)
        assert log.agg_arch.upper() == "SALAD", "By the moment only SALAD is supported"
        assert log.backbone_arch.upper() == "MAPANYTHING", "This log might not correspond to mapanything-salad"
        if mapanything is None:
            mapanything = load_pretrained_mapanything()
        model = MapAnythingSalad(mapanything, log.backbone_config, log.agg_config)
        full_state = log.state_dict
        prefix = "aggregator"
        salad_state = {k: v for k, v in full_state.items() if k.startswith(prefix)}
        del full_state
        gc.collect()

        model.load_state_dict(salad_state, strict=False)
        return model
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        all_encoder_features_across_views, all_encoder_registers_across_views = (
            self.backbone.dino_forward(images)
        )
        feats, cls = self.backbone.prepare_tokens_for_salad(
            all_encoder_features_across_views,
            all_encoder_registers_across_views
        )
        global_descriptor = self.aggregator((feats, cls))
        if len(global_descriptor.shape) == 2:
            global_descriptor = global_descriptor.unsqueeze(0)
        
        # Get input shape of the images, number of views, and batch size per view
        num_views, c, height, width = images.shape
        img_shape = (int(height), int(width))
        views = self.backbone.imgs_tensor_as_views(images)

        # Encode the optional geometric inputs and fuse with the encoded features from the N input views.
        # When optionalinput is not includded, trained tokens are used. Operation for token fusion is addition.
        # Use high precision to prevent NaN values after layer norm in dense representation encoder (due to high variance in last dim of features)
        with torch.autocast("cuda", enabled=False):
            all_encoder_features_across_views = (
                self.backbone._map_anything._encode_and_fuse_optional_geometric_inputs(
                    views, all_encoder_features_across_views
                )
            )

        final_info_sharing_multi_view_feat, intermediate_info_sharing_multi_view_feat = self.backbone.alternate_attention(
            all_encoder_features_across_views,
            all_encoder_registers_across_views,
            batch_size_per_view = 1
        )

        res = self.backbone.heads_forward(
            all_encoder_features_across_views,
            final_info_sharing_multi_view_feat,
            intermediate_info_sharing_multi_view_feat,
            num_views,
            img_shape
        )

        for item, descriptor in zip(res, global_descriptor):
            item['decriptor'] = descriptor

        return res

    def inference(self, img_path_list: List[str]) -> Dict:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        views = load_images(img_path_list) #Here the preprocessing takes place

        validated_views = validate_input_views_for_inference(views) #When only images are passed, this does nothing.
        # Transfer the views to the same device as the model
        ignore_keys = set(
            [
                "instance",
                "idx",
                "true_shape",
                "data_norm_type",
            ]
        )

        #When obly images are passed, this does view['image'] = view['image'].to(self.device)
        for view in validated_views: #Send some inputs to device
            for name in view.keys():
                if name in ignore_keys:
                    continue
                val = view[name]
                if name == "camera_poses" and isinstance(val, tuple): #Won't happen
                    view[name] = tuple(
                        x.to(self.backbone.device, non_blocking=True) for x in val
                    )
                elif hasattr(val, "to"): #Meh
                    view[name] = val.to(self.backbone.device, non_blocking=True)

        # Pre-process the input views
        processed_views = preprocess_input_views_for_inference(validated_views) #This one does not modify the images

        # Set the model input probabilities based on input args for ignoring inputs
        self.backbone._map_anything._configure_geometric_input_config(
            use_calibration=True,
            use_depth=True,
            use_pose=True,
            use_depth_scale=True,
            use_pose_scale=True,
        )

        imgs = self.backbone.imgs_tensor_from_views(processed_views)

        # Run the model
        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=amp_dtype):
                preds = self.forward(imgs)

        # Post-process the model outputs (including multi-view confidence if requested)
        preds = postprocess_model_outputs_for_inference( #Check if this could drop patch tokens/ descriptor
            raw_outputs=preds,
            input_views=processed_views,
            edge_normal_threshold=5.0
        )

        # Restore the original configuration
        self.backbone._map_anything._restore_original_geometric_input_config()

        return preds
