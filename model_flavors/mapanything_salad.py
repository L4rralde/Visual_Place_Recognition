from typing import List, Dict
import gc

import torch
import torch.nn as nn

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
    def from_lightning_log(cls, path: str, mapanything: object|None=None):
        log = LightningLog(path)

        assert log.agg_arch.upper() == "SALAD", "By the moment only SALAD is supported"
        assert log.backbone_arch.upper() == "MAP_ANYTHING", "This log might not correspond to mapanything-salad"
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
        # Get input shape of the images, number of views, and batch size per view
        num_views, c, height, width = images.shape
        img_shape = (int(height), int(width))

        patch_tokens = self.backbone.dino_forward(images)
        global_descriptor = self.aggregator(
            self.backbone.prepare_tokens_for_salad(
                patch_tokens,
                height//self.backbone.PATCH_SIZE,
                width//self.backbone.PATCH_SIZE
            )
        )

        all_encoder_features_across_views, all_encoder_registers_across_views = (
            self.backbone.unpack_dino_outputs(
                patch_tokens,
                height//self.backbone.PATCH_SIZE,
                width//self.backbone.PATCH_SIZE
            )
        )
        
        if len(global_descriptor.shape) == 2:
            global_descriptor = global_descriptor.unsqueeze(0)
        
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
        views = self.backbone.prepare_views(views)

        imgs = self.backbone.imgs_tensor_from_views(views)

        # Run the model
        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=amp_dtype):
                preds = self.forward(imgs)

        # Post-process the model outputs (including multi-view confidence if requested)
        preds = postprocess_model_outputs_for_inference( #Check if this could drop patch tokens/ descriptor
            raw_outputs=preds,
            input_views=views,
            edge_normal_threshold=5.0
        )

        return preds
