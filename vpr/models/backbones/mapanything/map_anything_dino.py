from typing import List, Tuple, Dict, Any, Optional
import gc
from dataclasses import dataclass

import torch
import torch.nn as nn
from uniception.models.encoders import ViTEncoderInput, DINOv2Encoder, ViTEncoderOutput
from uniception.models.info_sharing.base import MultiViewTransformerInput
from PIL import Image

from .transforms import preprocess_images
import os, sys
sys.path.append(os.path.dirname(__file__))
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from mapanything.utils.inference import postprocess_model_outputs_for_inference
from mapanything.utils.geometry import convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap

#This is quite more complicated than vggt.
#What's next?


def load_pretrained_mapanything() -> MapAnything:
    return MapAnything.from_pretrained("facebook/map-anything")


@dataclass
class ViTEncoderOutputForSalad(ViTEncoderOutput):
    features_for_salad: Optional[Dict[str, torch.Tensor]] = None

class MapAnythingBase(nn.Module):
    PATCH_SIZE = 14
    def __init__(self, map_anything: MapAnything, **kwargs):
        super().__init__()
        if 'num_trainable_blocks' in kwargs:
            print("num_trainable_blocks argument is not supported for VGGT backbone. VGGT is used as is")
        self.norm_layer = kwargs.get('norm_layer', True)
        self.probing_from_layer: int = kwargs.get('probing_from_layer', -1)
        self.num_channels = map_anything.encoder.model.embed_dim
        self._map_anything: MapAnything|None = None
        self._dino: DINOv2Encoder|None = None
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA must be available")
        self.device = 'cuda'
        self.adapter = nn.Identity()
    
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

    def prepare_tokens_for_salad(self, patch_tokens: Dict[str, torch.Tensor], images_shape: Tuple[int]) -> Tuple[torch.Tensor]:
        B, S, C_in, H, W = images_shape

        f = patch_tokens['x_salad_patchtokens']
        t = patch_tokens['x_salad_clstoken']
        
        f = f.reshape((B*S, H//14, W//14, self.num_channels)).permute(0, 3, 1, 2)

        return f, t

    def unpack_dino_outputs(self, patch_tokens: torch.Tensor, h: int, w:int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        num_views, num_tokens, dim = patch_tokens.shape
        assert dim == self.num_channels
        assert num_tokens >= h*w
        features = patch_tokens[:, :h*w].permute(0, 2, 1).view(num_views, dim, h, w)
        features = features.chunk(num_views, dim=0)

        if num_tokens == h*w:
            registers = None
        else:
            registers = patch_tokens[:, h*w:].permute(0, 2, 1).view(num_views, dim, -1)
            registers = registers.chunk(num_views, dim=0)
        return features, registers

    def dino_forward(self, all_imgs_across_views: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Encode all the input views (batch of images) in a single forward pass.
        Assumes all the input views have the same image shape, batch size, and data normalization type.

        Args:
            views (List[dict]): List of dictionaries containing the input views' images and instance information.
        """
        num_views, c, h, w = all_imgs_across_views.shape
        data_norm_type = 'dinov2'

        encoder_input = ViTEncoderInput(
            image=all_imgs_across_views, data_norm_type=data_norm_type
        )
        encoder_output: ViTEncoderOutputForSalad = self.uniception_dino_forward(encoder_input)

        features = encoder_output.features
        features = features.view(num_views, self.num_channels, -1).permute(0, 2, 1)
        regs = encoder_output.registers.permute(0, 2, 1)
        patch_tokens = torch.cat((features, regs), dim=1)
        return patch_tokens, encoder_output.features_for_salad

    def uniception_dino_forward(self, encoder_input: ViTEncoderInput) -> ViTEncoderOutput:
        """
        DINOv2 Encoder Forward Pass

        Args:
            encoder_input (ViTEncoderInput): Input data for the encoder. Input data must contain image normalization type and normalized image tensor.

        Returns:
            ViTEncoderOutput: Output data from the encoder.
        """
        # Check image normalization type
        self.dino._check_data_normalization_type(encoder_input.data_norm_type)

        # Check the dtype and shape of the input image
        assert isinstance(encoder_input.image, torch.Tensor), "Input must be a torch.Tensor"
        assert encoder_input.image.ndim == 4, "Input must be of shape (B, C, H, W)"
        batch_size, channels, height, width = encoder_input.image.shape
        assert channels == 3, "Input must have 3 channels"
        assert (
            height % self.dino.patch_size == 0 and width % self.dino.patch_size == 0
        ), f"Input shape must be divisible by patch size: {self.dino.patch_size}"

        # Extract the features from the DINOv2 model
        result_dict = self.dino_forward_features(encoder_input.image)

        # Patch tokens
        features = result_dict["x_norm_patchtokens"]
        features_for_salad = {
            k: result_dict[k]
            for k in ['x_salad_clstoken', 'x_salad_patchtokens']
        }

        # Resize the features to the expected shape
        # (B x Num_patches x Embed_dim) -> (B x Embed_dim x H / Patch_Size x W / Patch_Size)
        features = features.permute(0, 2, 1)
        features = features.reshape(
            -1, self.dino.enc_embed_dim, height // self.dino.patch_size, width // self.dino.patch_size
        ).contiguous()

        # Additional registers (including cls token) if present
        additional_registers = []

        # Add the cls token
        cls_token = result_dict["x_norm_clstoken"].unsqueeze(1)  # (B x 1 x Embed_dim)
        additional_registers.append(cls_token)

        # Add the registers
        registers = result_dict["x_norm_regtokens"]
        if registers is not None:
            additional_registers.append(registers)

        all_registers = torch.cat(additional_registers, dim=1) if len(additional_registers) > 0 else None
        if all_registers is not None:
            all_registers = all_registers.permute(0, 2, 1).contiguous()  # (B x Embed_dim x Num_registers)

        return ViTEncoderOutputForSalad(
            features=features,
            registers=all_registers,
            features_for_salad=features_for_salad
        )

    def dino_forward_features(self, x, masks=None):
        with torch.no_grad():
            x = self.dino.model.prepare_tokens_with_masks(x, masks)
            for i, blk in enumerate(self.dino.model.blocks):
                x = blk(x)
                if i == self.probing_from_layer:
                    x_for_salad = x.clone()
                    break #To speed up training, let's stop earlier

        x_for_salad = self.adapter(x_for_salad)

        with torch.no_grad():
            if self.norm_layer:
                x_for_salad = self.dino.model.norm(x_for_salad)
            x_norm = self.dino.model.norm(x)
            
        
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.dino.model.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.dino.model.num_register_tokens + 1 :],
            "x_salad_clstoken": x_for_salad[:, 0],
            "x_salad_patchtokens": x_for_salad[:, self.dino.model.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }
    

    def imgs_tensor_as_views(self, imgs: torch.Tensor) -> List[Dict[str, Any]]:
        num_views, c, height, width = imgs.shape
        assert c == 3, "Not an image"

        return [
            {
                'img': imgs[i].unsqueeze(0),
                'true_shape': [[height, width]],
                'idx': i,
                'instance': i,
                'data_norm_type': ['dinov2'],
                'is_metric_scale_tensor': torch.ones(
                    (1, ),
                    dtype=torch.bool,
                    device=self.device
                )
            }
            for i in range(num_views)
        ]

    def imgs_tensor_from_views(self, views: List[Dict[str, Any]]) -> torch.Tensor:
        ims_sample: torch.Tensor = views[0]["img"]
        imgs = torch.zeros(
            (len(views), *ims_sample.shape[-3:]),
            dtype=ims_sample.dtype,
            device=ims_sample.device
        )
        for i in range(len(views)):
            imgs[i] = views[i]["img"]
        return imgs

    def preprocess_images(self, pil_img_list: List[Image.Image]) -> torch.Tensor:
        if not isinstance(pil_img_list, list) :
            raise TypeError(f"input must be a list of tensors.")
        tensor_img_list = preprocess_images(pil_img_list)
        return torch.cat(
            [img.unsqueeze(0) for img in tensor_img_list],
            dim=0
        )

    @staticmethod
    def postprocess_model_outputs_for_inference(
        raw_outputs: List[Dict[str, torch.Tensor]],
        input_views: List[Dict[str, Any]],
        edge_normal_threshold: float=5.0,
        **kwargs
    ) -> List[Dict[str, torch.Tensor]]:
        with torch.autocast('cuda', enabled=False):
            return postprocess_model_outputs_for_inference(
                raw_outputs=raw_outputs,
                input_views=input_views,
                edge_normal_threshold=edge_normal_threshold,
                **kwargs
            )
    
    def _clip_probing_from_layer(self) -> int:
        dino_depth = len(self.dino.model.blocks)
        if self.probing_from_layer < 0:
            self.probing_from_layer = dino_depth + self.probing_from_layer
        assert 0 <= self.probing_from_layer < dino_depth, \
            "Index probing_from_layer out of range"


class MapAnythingBackbone(MapAnythingBase):
    def __init__(self, map_anything, **kwargs):
        super().__init__(map_anything, **kwargs)
        self._map_anything = map_anything
        self._map_anything.use_register_tokens_from_encoder = True
        self._clip_probing_from_layer()

    @classmethod
    def from_pretrained(cls, **kwargs):
        map_anything = load_pretrained_mapanything()
        return cls(map_anything, **kwargs)

    def prepare_views(self, views: Dict[str, Any]) -> Dict[str, Any]:
        for view in views:
            view['img'] = view['img'].to(self.device, non_blocking=True)

        for view in views:
            view["is_metric_scale"] = torch.ones(
                view['img'].shape[0],
                dtype=torch.bool,
                device=self.device
            )
        
        return views

    def inference(self, img_path_list: List[str]) -> Dict:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        views = load_images(img_path_list) #Here the preprocessing takes place
        views = self.prepare_views(views)

        imgs = self.imgs_tensor_from_views(views)

        # Run the model
        with torch.no_grad():
            with torch.autocast("cuda", enabled=True, dtype=amp_dtype):
                preds = self.forward(imgs)

        # Post-process the model outputs (including multi-view confidence if requested)
        preds = self.postprocess_model_outputs_for_inference(preds, views)

        return preds

    def alternate_attention(
        self,
        all_encoder_features_across_views,
        all_encoder_registers_across_views,
        batch_size_per_view
    ):
        #Alternate attention starts. Where's cam token?
        # Expand the scale token to match the batch size
        input_scale_token = (
            self._map_anything.scale_token.unsqueeze(0)
            .unsqueeze(-1)
            .repeat(batch_size_per_view, 1, 1)
        )  # (B, C, 1)

        #Alternate attention
        # Combine all images into view-centric representation
        # Output is a list containing the encoded features for all N views after information sharing.
        info_sharing_input = MultiViewTransformerInput(
            features=all_encoder_features_across_views,
            additional_input_tokens_per_view=all_encoder_registers_across_views,
            additional_input_tokens=input_scale_token,
        )
        final_info_sharing_multi_view_feat = None
        intermediate_info_sharing_multi_view_feat = None
        if self._map_anything.info_sharing_return_type == "no_intermediate_features":
            final_info_sharing_multi_view_feat = self._map_anything.info_sharing(info_sharing_input)
        elif self._map_anything.info_sharing_return_type == "intermediate_features":
            (
                final_info_sharing_multi_view_feat,
                intermediate_info_sharing_multi_view_feat,
            ) = self._map_anything.info_sharing(info_sharing_input)
        
        return final_info_sharing_multi_view_feat, intermediate_info_sharing_multi_view_feat

    def heads_forward(
        self,
        all_encoder_features_across_views,
        final_info_sharing_multi_view_feat,
        intermediate_info_sharing_multi_view_feat,
        num_views,
        img_shape,
    ):
        #Prediction heads
        if self._map_anything.pred_head_type == "linear":
            # Stack the features for all views
            dense_head_inputs = torch.cat(
                final_info_sharing_multi_view_feat.features, dim=0
            )
        elif self._map_anything.pred_head_type in ["dpt", "dpt+pose"]:
            # Get the list of features for all views
            dense_head_inputs_list = []
            if self._map_anything.use_encoder_features_for_dpt:
                # Stack all the image encoder features for all views
                stacked_encoder_features = torch.cat(
                    all_encoder_features_across_views, dim=0
                )
                dense_head_inputs_list.append(stacked_encoder_features)
                # Stack the first intermediate features for all views
                stacked_intermediate_features_1 = torch.cat(
                    intermediate_info_sharing_multi_view_feat[0].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_1)
                # Stack the second intermediate features for all views
                stacked_intermediate_features_2 = torch.cat(
                    intermediate_info_sharing_multi_view_feat[1].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_2)
                # Stack the last layer features for all views
                stacked_final_features = torch.cat(
                    final_info_sharing_multi_view_feat.features, dim=0
                )
                dense_head_inputs_list.append(stacked_final_features)
            else:
                # Stack the first intermediate features for all views
                stacked_intermediate_features_1 = torch.cat(
                    intermediate_info_sharing_multi_view_feat[0].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_1)
                # Stack the second intermediate features for all views
                stacked_intermediate_features_2 = torch.cat(
                    intermediate_info_sharing_multi_view_feat[1].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_2)
                # Stack the third intermediate features for all views
                stacked_intermediate_features_3 = torch.cat(
                    intermediate_info_sharing_multi_view_feat[2].features, dim=0
                )
                dense_head_inputs_list.append(stacked_intermediate_features_3)
                # Stack the last layer
                stacked_final_features = torch.cat(
                    final_info_sharing_multi_view_feat.features, dim=0
                )
                dense_head_inputs_list.append(stacked_final_features)
        else:
            raise ValueError(
                f"Invalid pred_head_type: {self._map_anything.pred_head_type}. Valid options: ['linear', 'dpt', 'dpt+pose']"
            )

        with torch.autocast("cuda", enabled=False):
            # Prepare inputs for the downstream heads
            if self._map_anything.pred_head_type == "linear":
                dense_head_inputs = dense_head_inputs
            elif self._map_anything.pred_head_type in ["dpt", "dpt+pose"]:
                dense_head_inputs = dense_head_inputs_list
            scale_head_inputs = (
                final_info_sharing_multi_view_feat.additional_token_features
            )

            # Run the downstream heads
            dense_final_outputs, pose_final_outputs, scale_final_output = (
                self._map_anything.downstream_head(
                    dense_head_inputs=dense_head_inputs,
                    scale_head_inputs=scale_head_inputs,
                    img_shape=img_shape,
                    memory_efficient_inference=True,
                )
            )

            # Prepare the final scene representation for all views
            if self._map_anything.scene_rep_type in [
                "pointmap",
                "pointmap+confidence",
                "pointmap+mask",
                "pointmap+confidence+mask",
            ]:
                output_pts3d = dense_final_outputs.value
                # Reshape final scene representation to (B * V, H, W, C)
                output_pts3d = output_pts3d.permute(0, 2, 3, 1).contiguous()
                # Split the predicted pointmaps back to their respective views
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                # Pack the output as a list of dictionaries
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            elif self._map_anything.scene_rep_type in [
                "raymap+depth",
                "raymap+depth+confidence",
                "raymap+depth+mask",
                "raymap+depth+confidence+mask",
            ]:
                # Reshape final scene representation to (B * V, H, W, C)
                output_scene_rep = dense_final_outputs.value.permute(
                    0, 2, 3, 1
                ).contiguous()
                # Get the predicted ray origins, directions, and depths along rays
                output_ray_origins, output_ray_directions, output_depth_along_ray = (
                    output_scene_rep.split([3, 3, 1], dim=-1)
                )
                # Get the predicted pointmaps
                output_pts3d = (
                    output_ray_origins + output_ray_directions * output_depth_along_ray
                )
                # Split the predicted quantities back to their respective views
                output_ray_origins_per_view = output_ray_origins.chunk(num_views, dim=0)
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                # Pack the output as a list of dictionaries
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "ray_origins": output_ray_origins_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "ray_directions": output_ray_directions_per_view[i],
                            "depth_along_ray": output_depth_along_ray_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            elif self._map_anything.scene_rep_type in [
                "raydirs+depth+pose",
                "raydirs+depth+pose+confidence",
                "raydirs+depth+pose+mask",
                "raydirs+depth+pose+confidence+mask",
            ]:
                # Reshape output dense rep to (B * V, H, W, C)
                output_dense_rep = dense_final_outputs.value.permute(
                    0, 2, 3, 1
                ).contiguous()
                # Get the predicted ray directions and depths along rays
                output_ray_directions, output_depth_along_ray = output_dense_rep.split(
                    [3, 1], dim=-1
                )
                # Get the predicted camera translations and quaternions
                output_cam_translations, output_cam_quats = (
                    pose_final_outputs.value.split([3, 4], dim=-1)
                )
                # Get the predicted pointmaps in world frame and camera frame
                output_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        output_ray_directions,
                        output_depth_along_ray,
                        output_cam_translations,
                        output_cam_quats,
                    )
                )
                output_pts3d_cam = output_ray_directions * output_depth_along_ray
                # Split the predicted quantities back to their respective views
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_cam_translations_per_view = output_cam_translations.chunk(
                    num_views, dim=0
                )
                output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)
                # Pack the output as a list of dictionaries
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "pts3d_cam": output_pts3d_cam_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "ray_directions": output_ray_directions_per_view[i],
                            "depth_along_ray": output_depth_along_ray_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "cam_trans": output_cam_translations_per_view[i]
                            * scale_final_output,
                            "cam_quats": output_cam_quats_per_view[i],
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            elif self._map_anything.scene_rep_type in [
                "campointmap+pose",
                "campointmap+pose+confidence",
                "campointmap+pose+mask",
                "campointmap+pose+confidence+mask",
            ]:
                # Get the predicted camera frame pointmaps
                output_pts3d_cam = dense_final_outputs.value
                # Reshape final scene representation to (B * V, H, W, C)
                output_pts3d_cam = output_pts3d_cam.permute(0, 2, 3, 1).contiguous()
                # Get the predicted camera translations and quaternions
                output_cam_translations, output_cam_quats = (
                    pose_final_outputs.value.split([3, 4], dim=-1)
                )
                # Get the ray directions and depths along rays
                output_depth_along_ray = torch.norm(
                    output_pts3d_cam, dim=-1, keepdim=True
                )
                output_ray_directions = output_pts3d_cam / output_depth_along_ray
                # Get the predicted pointmaps in world frame
                output_pts3d = (
                    convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                        output_ray_directions,
                        output_depth_along_ray,
                        output_cam_translations,
                        output_cam_quats,
                    )
                )
                # Split the predicted quantities back to their respective views
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_cam_translations_per_view = output_cam_translations.chunk(
                    num_views, dim=0
                )
                output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)
                # Pack the output as a list of dictionaries
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "pts3d_cam": output_pts3d_cam_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "ray_directions": output_ray_directions_per_view[i],
                            "depth_along_ray": output_depth_along_ray_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "cam_trans": output_cam_translations_per_view[i]
                            * scale_final_output,
                            "cam_quats": output_cam_quats_per_view[i],
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            elif self._map_anything.scene_rep_type in [
                "pointmap+raydirs+depth+pose",
                "pointmap+raydirs+depth+pose+confidence",
                "pointmap+raydirs+depth+pose+mask",
                "pointmap+raydirs+depth+pose+confidence+mask",
            ]:
                # Reshape final scene representation to (B * V, H, W, C)
                output_dense_rep = dense_final_outputs.value.permute(
                    0, 2, 3, 1
                ).contiguous()
                # Get the predicted pointmaps, ray directions and depths along rays
                output_pts3d, output_ray_directions, output_depth_along_ray = (
                    output_dense_rep.split([3, 3, 1], dim=-1)
                )
                # Get the predicted camera translations and quaternions
                output_cam_translations, output_cam_quats = (
                    pose_final_outputs.value.split([3, 4], dim=-1)
                )
                # Get the predicted pointmaps in camera frame
                output_pts3d_cam = output_ray_directions * output_depth_along_ray
                # Replace the predicted world-frame pointmaps if required
                if self._map_anything.pred_head_config["adaptor_config"][
                    "use_factored_predictions_for_global_pointmaps"
                ]:
                    output_pts3d = (
                        convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
                            output_ray_directions,
                            output_depth_along_ray,
                            output_cam_translations,
                            output_cam_quats,
                        )
                    )
                # Split the predicted quantities back to their respective views
                output_ray_directions_per_view = output_ray_directions.chunk(
                    num_views, dim=0
                )
                output_depth_along_ray_per_view = output_depth_along_ray.chunk(
                    num_views, dim=0
                )
                output_cam_translations_per_view = output_cam_translations.chunk(
                    num_views, dim=0
                )
                output_cam_quats_per_view = output_cam_quats.chunk(num_views, dim=0)
                output_pts3d_per_view = output_pts3d.chunk(num_views, dim=0)
                output_pts3d_cam_per_view = output_pts3d_cam.chunk(num_views, dim=0)
                # Pack the output as a list of dictionaries
                res = []
                for i in range(num_views):
                    res.append(
                        {
                            "pts3d": output_pts3d_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "pts3d_cam": output_pts3d_cam_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "ray_directions": output_ray_directions_per_view[i],
                            "depth_along_ray": output_depth_along_ray_per_view[i]
                            * scale_final_output.unsqueeze(-1).unsqueeze(-1),
                            "cam_trans": output_cam_translations_per_view[i]
                            * scale_final_output,
                            "cam_quats": output_cam_quats_per_view[i],
                            "metric_scaling_factor": scale_final_output,
                        }
                    )
            else:
                raise ValueError(
                    f"Invalid scene_rep_type: {self._map_anything.scene_rep_type}. \
                    Valid options: ['pointmap', 'raymap+depth', 'raydirs+depth+pose', 'campointmap+pose', 'pointmap+raydirs+depth+pose' \
                                    'pointmap+confidence', 'raymap+depth+confidence', 'raydirs+depth+pose+confidence', 'campointmap+pose+confidence', 'pointmap+raydirs+depth+pose+confidence' \
                                    'pointmap+mask', 'raymap+depth+mask', 'raydirs+depth+pose+mask', 'campointmap+pose+mask', 'pointmap+raydirs+depth+pose+mask' \
                                    'pointmap+confidence+mask', 'raymap+depth+confidence+mask', 'raydirs+depth+pose+confidence+mask', 'campointmap+pose+confidence+mask', 'pointmap+raydirs+depth+pose+confidence+mask']"
                )

            # Get the output confidences for all views (if available) and add them to the result
            if "confidence" in self._map_anything.scene_rep_type:
                output_confidences = dense_final_outputs.confidence
                # Reshape confidences to (B * V, H, W)
                output_confidences = (
                    output_confidences.permute(0, 2, 3, 1).squeeze(-1).contiguous()
                )
                # Split the predicted confidences back to their respective views
                output_confidences_per_view = output_confidences.chunk(num_views, dim=0)
                # Add the confidences to the result
                for i in range(num_views):
                    res[i]["conf"] = output_confidences_per_view[i]

            # Get the output masks (and logits) for all views (if available) and add them to the result
            if "mask" in self._map_anything.scene_rep_type:
                # Get the output masks
                output_masks = dense_final_outputs.mask
                # Reshape masks to (B * V, H, W)
                output_masks = output_masks.permute(0, 2, 3, 1).squeeze(-1).contiguous()
                # Threshold the masks at 0.5 to get binary masks (0: ambiguous, 1: non-ambiguous)
                output_masks = output_masks > 0.5
                # Split the predicted masks back to their respective views
                output_masks_per_view = output_masks.chunk(num_views, dim=0)
                # Get the output mask logits (for loss)
                output_mask_logits = dense_final_outputs.logits
                # Reshape mask logits to (B * V, H, W)
                output_mask_logits = (
                    output_mask_logits.permute(0, 2, 3, 1).squeeze(-1).contiguous()
                )
                # Split the predicted mask logits back to their respective views
                output_mask_logits_per_view = output_mask_logits.chunk(num_views, dim=0)
                # Add the masks and logits to the result
                for i in range(num_views):
                    res[i]["non_ambiguous_mask"] = output_masks_per_view[i]
                    res[i]["non_ambiguous_mask_logits"] = output_mask_logits_per_view[i]

        return res

    def forward(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get input shape of the images, number of views, and batch size per view
        num_views, c, height, width = imgs.shape
        img_shape = (int(height), int(width))

        # Run the image encoder on all the input views
        #DIno forward
        patch_tokens, _ = self.dino_forward(imgs)
        all_encoder_features_across_views, all_encoder_registers_across_views = (
            self.unpack_dino_outputs(patch_tokens, height//self.PATCH_SIZE, width//self.PATCH_SIZE)
        )

        views = self.imgs_tensor_as_views(imgs)

        # Encode the optional geometric inputs and fuse with the encoded features from the N input views.
        # When optionalinput is not includded, trained tokens are used. Operation for token fusion is addition.
        # Use high precision to prevent NaN values after layer norm in dense representation encoder (due to high variance in last dim of features)
        with torch.autocast("cuda", enabled=False):
            all_encoder_features_across_views = (
                self._map_anything._encode_and_fuse_optional_geometric_inputs(
                    views, all_encoder_features_across_views
                )
            )

        final_info_sharing_multi_view_feat, intermediate_info_sharing_multi_view_feat = self.alternate_attention(
            all_encoder_features_across_views,
            all_encoder_registers_across_views,
            batch_size_per_view = 1
        )

        res = self.heads_forward(
            all_encoder_features_across_views,
            final_info_sharing_multi_view_feat,
            intermediate_info_sharing_multi_view_feat,
            num_views,
            img_shape
        )

        return res


class MapAnythingDino(MapAnythingBase):
    def __init__(self, map_anything: MapAnything, **kwargs) -> None:
        super().__init__(map_anything, **kwargs)
        self._dino: DINOv2Encoder = map_anything.encoder #dinov2 from uniception. Which actually instantiates dinov2 from meta
        #Actual dino: map_anything.encoder.model
        self._clip_probing_from_layer

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

        _, features_for_salad = self.dino_forward(images)
        f, t = self.prepare_tokens_for_salad(features_for_salad, (1, n, c, h, w))
        
        return f, t


def mapanything_inference(
    mapanything: MapAnything,
    img_list: List[str]
) -> Dict[str, torch.Tensor]:
    views = load_images(img_list)
    return mapanything.infer(
        views,                            # Input views
        memory_efficient_inference=True,  # Trades off speed for more views (up to 2000 views on 140 GB). Trade off is negligible - see profiling section
        minibatch_size=None,              # Minibatch size for memory-efficient inference (use 1 for smallest GPU memory consumption). Default is dynamic computation based on available GPU memory.
        use_amp=True,                     # Use mixed precision inference (recommended)
        amp_dtype="bf16",                 # bf16 inference (recommended; falls back to fp16 if bf16 not supported)
        apply_mask=True,                  # Apply masking to dense geometry outputs
        mask_edges=True,                  # Remove edge artifacts by using normals and depth
        apply_confidence_mask=False,      # Filter low-confidence regions
        confidence_percentile=10,         # Remove bottom 10 percentile confidence pixels
        use_multiview_confidence=False,   # Enable multi-view depth consistency based confidence in place of learning-based one
    )
