from typing import List, Tuple, Dict
from dataclasses import asdict
import gc

import torch
import torch.nn as nn
from uniception.models.encoders import ViTEncoderInput, ViTEncoderOutput, DINOv2Encoder

import os, sys
sys.path.append(os.path.dirname(__file__))
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.inference import(
    validate_input_views_for_inference,
    preprocess_input_views_for_inference,
    postprocess_model_outputs_for_inference
)

#This is quite more complicated than vggt.
#What's next?


def load_pretrained_mapanything() -> MapAnything:
    return MapAnything.from_pretrained("facebook/map-anything")


class MapAnythingBase(nn.Module):
    PATCH_SIZE = 14
    def __init__(self, map_anything: MapAnything, **kwargs):
        super().__init__()
        if 'num_trainable_blocks' in kwargs:
            print("num_trainable_blocks argument is not supported for VGGT backbone. VGGT is used as is")
        self.norm_layer = kwargs.get('norm_layer', True)
        assert self.norm_layer, "Feature not implemented yet. Norm layer must be always true"
        self.num_channels = map_anything.encoder.model.embed_dim
        self._map_anything: MapAnything|None = None
        self._dino: DINOv2Encoder|None = None
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA must be available")
        self.device = 'cuda'
    
    @property
    def map_anything(self) -> MapAnything:
        if self._map_anything is None:
            raise RuntimeError("self.map_anything is not available in this class")
        return self._map_anything

    @property
    def dino(self) -> DINOv2Encoder:
        if self._dino is not None:
            return self._dino
        if self._map_anything is not None:
            return self._map_anything.encoder
        raise RuntimeError("self.dino is not set in this class")

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
                        x.to(self.device, non_blocking=True) for x in val
                    )
                elif hasattr(val, "to"): #Meh
                    view[name] = val.to(self.device, non_blocking=True)

        # Pre-process the input views
        processed_views = preprocess_input_views_for_inference(validated_views) #This one does not modify the images

        # Set the model input probabilities based on input args for ignoring inputs
        self.map_anything._configure_geometric_input_config(
            use_calibration=True,
            use_depth=True,
            use_pose=True,
            use_depth_scale=True,
            use_pose_scale=True,
        )

        # Run the model
        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=amp_dtype):
                preds = self.forward(
                    processed_views,
                    memory_efficient_inference=True,
                    minibatch_size=None,
                )

        # Post-process the model outputs (including multi-view confidence if requested)
        preds = postprocess_model_outputs_for_inference( #Check if this could drop patch tokens/ descriptor
            raw_outputs=preds,
            input_views=processed_views,
        )

        # Restore the original configuration
        self.map_anything._restore_original_geometric_input_config()

        return preds

    def forward(self) -> None:
        pass


class MapAnythingBackbone(nn.Module):
    def __init__(self, model: MapAnything, **kwargs) -> None:
        self.mapanything: MapAnything = model
        self.mapanything.use_register_tokens_from_encoder = True #THis is the easiest way to access to dino cls token
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA must be available")
        self.device = 'cuda'

    @classmethod
    def from_pretrained(cls, **kwargs):
        model = MapAnything.from_pretrained("facebook/map-anything")
        backbone = cls(model, **kwargs)
        return backbone

    def inference(self, images: list) -> dict:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        views = load_images(images) #Here the preprocessing takes place

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
                        x.to(self.device, non_blocking=True) for x in val
                    )
                elif hasattr(val, "to"): #Meh
                    view[name] = val.to(self.device, non_blocking=True)

        # Pre-process the input views
        processed_views = preprocess_input_views_for_inference(validated_views) #This one does not modify the images

        # Set the model input probabilities based on input args for ignoring inputs
        self.mapanything._configure_geometric_input_config(
            use_calibration=True,
            use_depth=True,
            use_pose=True,
            use_depth_scale=True,
            use_pose_scale=True,
        )

        # Run the model
        with torch.autocast("cuda", enabled=True, dtype=amp_dtype):
            preds = self.forward(
                processed_views,
                memory_efficient_inference=True,
                minibatch_size=None,
            )

        # Post-process the model outputs (including multi-view confidence if requested)
        preds = postprocess_model_outputs_for_inference(
            raw_outputs=preds,
            input_views=processed_views,
        )

        # Restore the original configuration
        self.mapanything._restore_original_geometric_input_config()

        return preds

    def forward(self, views):
        batch_size_per_view, _, height, width = views[0]["img"].shape
        img_shape = (int(height), int(width))
        num_views = len(views)

        # Run the image encoder on all the input views
        use_register_tokens_from_encoder = self.mapanything.use_register_tokens_from_encoder

        all_encoder_features_across_views, all_encoder_registers_across_views = (
            self._encode_n_views(views)
        )
        assert all_encoder_registers_across_views is not None, "We need that data"
        #all_encoder_features: Includes all feature tokens. all_encoder_registers_acrros_views[0] includes the cls token.
        print(all_encoder_registers_across_views)
        


    def _encode_n_views(self, views):
        """
        Encode all the input views (batch of images) in a single forward pass.
        Assumes all the input views have the same image shape, batch size, and data normalization type.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.

        Returns:
            A tuple containing:
                List[torch.Tensor]: A list containing the encoded features for all N views.
                List[torch.Tensor]: A list containing the encoded per-view registers for all N views.
        """
        num_views = len(views)
        data_norm_type = views[0]["data_norm_type"][0]
        imgs_list = [view["img"] for view in views]
        all_imgs_across_views = torch.cat(imgs_list, dim=0)
        encoder_input = ViTEncoderInput(
            image=all_imgs_across_views, data_norm_type=data_norm_type
        )
        encoder_output = self.mapanything.encoder(encoder_input)
        print(asdict(encoder_output).keys())
        all_encoder_features_across_views = encoder_output.features.chunk(
            num_views, dim=0
        )
        all_encoder_registers_across_views = None
        #We need "all_encoder_registers_across_views"
        if (
            self.mapanything.use_register_tokens_from_encoder
            and encoder_output.registers is not None
        ):
            all_encoder_registers_across_views = encoder_output.registers.chunk(
                num_views, dim=0
            )

        return all_encoder_features_across_views, all_encoder_registers_across_views


class MapAnythingDino(nn.Module):
    def __init__(self, map_anything: MapAnything, **kwargs) -> None:
        super().__init__()
        if 'num_trainable_blocks' in kwargs:
            print("num_trainable_blocks argument is not supported for VGGT backbone. VGGT is used as is")
        self.norm_layer: nn.Module = kwargs.get('norm_layer', True)
        assert self.norm_layer, "By the moment this feature has not been implemented yet"
        self.num_channels: int = map_anything.encoder.model.embed_dim
        self._dino: DINOv2Encoder = map_anything.encoder #dinov2 from uniception. Which actually instantiates dinov2 from meta
        #Actual dino: map_anything.encoder.model

    @classmethod
    def from_pretrained(cls, **kwargs):
        map_anything = MapAnything.from_pretrained("facebook/map-anything")
        full_state = map_anything.state_dict()
        
        prefix = "encoder."
        dino_state = {
            k: v for k, v in full_state.items()
            if k.startswith(prefix)
        }
        keys = map_anything.load_state_dict(dino_state, strict=False)
        backbone = cls(map_anything, **kwargs)

        del full_state
        del map_anything
        del dino_state
        torch.cuda.empty_cache()
        gc.collect()
    
        return backbone
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        n, c, h, w = images.shape
        assert c == 3, "Wrong input shape"

        data_norm_type = "dinov2"
        encoder_input = ViTEncoderInput(
            image=images, data_norm_type=data_norm_type
        )
        with torch.no_grad():
            #If want access to no normalized tokens, must modify/implement self._dino: DINOv2Encoder forward method
            encoder_output = self._dino(encoder_input) 
        f, t = self.prepare_tokens_for_salad(encoder_output)
        return f, t

    def prepare_tokens_for_salad(self, encoder_output: ViTEncoderOutput) -> Tuple[torch.Tensor, torch.Tensor]:
        f, t = encoder_output.features, encoder_output.registers
        return f, t.squeeze(-1)
