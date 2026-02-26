
from typing import Sequence, List, Dict, Tuple

from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from addict import Dict


import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Prediction
from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer
from depth_anything_3.model.da3 import DepthAnything3Net


def da3_from_pretained(model_name: str, **kwargs) -> DepthAnything3:
    return DepthAnything3.from_pretrained(f"depth-anything/{model_name}")


class DepthAnything3Backbone(nn.Module):
    PATCH_SIZE: int = 14
    def __init__(self, da3: DepthAnything3,**kwargs):
        super().__init__()
        self.num_channels = da3.model.backbone.pretrained.num_features
        self.dino_alt_start = da3.model.backbone.pretrained.alt_start
        self.input_processor = da3.input_processor
        self.da3: DepthAnything3|None = None
        if kwargs.get('keep_da3', True):
            self.da3 = da3

    @property
    def dino(self) -> DinoVisionTransformer:
        return self.da3.model.backbone.pretrained

    @staticmethod
    def from_pretrained(model_name: str = "da3-base", **kwargs) -> "DepthAnything3Backbone":
        da3 = da3_from_pretained(model_name, **kwargs)
        return DepthAnything3Backbone(da3, **kwargs)

    def forward(
        self,
        imgs: torch.Tensor,
        process_res: int = 504,
        export_feat_layers: Sequence[int] | None = None,
        add_imgs: bool=False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        #1. Input preprocessing. Trick to use da3 api preprocessing: convert tensor to ndarray
        image_list = self._prepare_inputs(imgs) #Tensor to np.ndarray
        imgs_cpu = self._preprocess_inputs( #Img reshaping.
            image_list,
            process_res=process_res
        )
        #To device, and float()
        device = self._get_model_device() 
        imgs = imgs_cpu.to(device, non_blocking=True)[None].float()

        if export_feat_layers is None:
            export_feat_layers = [self.dino.alt_start - 1]
        feat_layers = list(export_feat_layers)

        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=imgs.device.type, dtype=autocast_dtype):
            batch_shape = imgs.shape

            #2. per frame processing
            x, aux_output_bf_alt = self._dino_attend(imgs, batch_shape, feat_layers, **kwargs)

            #3. Per sequence processing.
            outputs, aux_output_af_alt = self._alt_attend(
                x,
                batch_shape,
                n=self.da3.model.backbone.out_layers,
                export_feat_layers=feat_layers,
                **kwargs
            )
            aux_outputs = aux_output_bf_alt + aux_output_af_alt
            assert len(aux_output_af_alt) == 0, "For da3-salad, this must be empty"
            #For DINO, we expect aux_output_af_alt to be empty

            #Process alt_attention outputs to get only the normalized features.
            feats = self._alt_attend_feats(outputs)
            #Process aux_features to get only the normalized tokens and normalized cls token.
            aux_feats, cls_token = self._aux_layers_feats(aux_outputs)

            #4. Prediction heads
            output = self._heads_forward(imgs, feats, **kwargs)
    
            #Reshapes aux features of the given list of layers.
            #Each layer's aux features is reshped to Batch_size, sequence, num_vertical_patches, num horizontal patches, embed dim
            #Each layer's acx features is resotred in a dict, e.g. f"feat_layer_{feat_layer}"
            #output.aux is a dictionary 
            H, W = imgs.shape[-2], imgs.shape[-1]
            output.aux = self._extract_auxiliary_features(aux_feats, feat_layers, H, W)
            output.aux_cls = self._extract_cls_token(cls_token, feat_layers)

        #Adding pre-processed images:
        if add_imgs:
            output = self.da3._add_processed_images(output, imgs_cpu)
        return output

    def _dino_attend(self, x, batch_shape, export_feat_layers=[], **kwargs) -> Tuple[torch.Tensor]:
        assert self.dino.alt_start != -1, "Alternate start needed"
        assert isinstance(x, torch.Tensor), "Expected input of type tensor"
        B, S, _, H, W = batch_shape
        x = self.dino.prepare_tokens_with_masks(x)
        aux_output = []
        pos, _ = self.dino._prepare_rope(B, S, H, W, x.device)

        for i, blk in enumerate(self.dino.blocks):
            if i == self.dino.alt_start:
                break
            if i < self.dino.rope_start or self.dino.rope is None:
                _, l_pos = None, None
            else:
                l_pos = pos
            x = self.dino.process_attention(x, blk, 'local', pos=l_pos)

            if i in export_feat_layers:
                aux_output.append(x)
        return x, aux_output

    def _alt_attend(self, x, batch_shape: Tuple[int], n=1, export_feat_layers=[], **kwargs):
        #l_pos, g_pos, are they constants one it is calculated first?
        B, S, _, H, W = batch_shape
        output, total_block_len, aux_output = [], len(self.dino.blocks), []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self.dino._prepare_rope(B, S, H, W, x.device)

        for i, blk in enumerate(self.dino.blocks):
            if i < self.dino.alt_start:
                if i in blocks_to_take:
                    raise RuntimeError(f"Feature from layer {i} is missing. List to take: {n}")
                continue
            if i < self.dino.rope_start or self.dino.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos
            if self.dino.alt_start != -1 and i ==self.dino.alt_start:
                if kwargs.get("cam_token", None) is not None:
                    cam_token = kwargs.get("cam_token")
                else:
                    ref_token = self.dino.camera_token[:, :1].expand(B, -1, -1)
                    src_token = self.dino.camera_token[:, 1:].expand(B, S - 1, -1)
                    cam_token = torch.cat([ref_token, src_token], dim = 1)
                x[:, :, 0] = cam_token
            
            if self.dino.alt_start != -1 and i >= self.dino.alt_start and i % 2 == 1:
                x = self.dino.process_attention(
                    x, blk, "global", pos=g_pos, attn_mask=kwargs.get("attn_mask", None)
                )
            else:
                x = self.dino.process_attention(x, blk, "local", pos=l_pos)
                local_x = x
            
            if i in blocks_to_take:
                out_x = torch.cat([local_x, x], dim=-1) if self.dino.cat_token else x
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(x)
        return output, aux_output

    def _alt_attend_feats(self, outputs: torch.Tensor) -> Tuple[torch.Tensor]:
        camera_tokens = [out[0] for out in outputs]
        if outputs[0][1].shape[-1] == self.dino.embed_dim:
            outputs = [self.dino.norm(out[1]) for out in outputs]
        elif outputs[0][1].shape[-1] == (self.dino.embed_dim * 2): #Idk why feature dim can be doubled, but one is normalized. I think its because features and last local features are passed
            outputs = [
                torch.cat(
                    [out[1][..., : self.dino.embed_dim], self.dino.norm(out[1][..., self.dino.embed_dim :])],
                    dim=-1,
                )
                for out in outputs
            ]
        else:
            raise ValueError(f"Invalid output shape: {outputs[0][1].shape}")
        outputs = [out[..., 1 + self.dino.num_register_tokens :, :] for out in outputs] #Taking feat tokens only
        feats = tuple(zip(outputs, camera_tokens))

        return feats

    def _aux_layers_feats(self, aux_outputs: torch.Tensor) -> None:
        aux_outputs = [self.dino.norm(out) for out in aux_outputs] #Applying final ViT norm layer
        cls_outputs = [out[..., 0, :] for out in aux_outputs]
        aux_feats = [out[..., 1 + self.dino.num_register_tokens :, :] for out in aux_outputs]

        return aux_feats, cls_outputs

    def _get_intermediate_layers_not_chunked(self, x, n=1, export_feat_layers=[], **kwargs):
        batch_shape = x.shape
        x, aux_output_bf_alt = self._dino_attend(x, batch_shape, export_feat_layers, **kwargs)
        output, aux_output_af_alt = self._alt_attend(x, batch_shape, n, export_feat_layers, **kwargs)
        
        aux_output = aux_output_bf_alt + aux_output_af_alt
        return output, aux_output

    def _heads_forward(self, imgs: torch.Tensor, feats: Tuple[torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        H, W = imgs.shape[-2], imgs.shape[-1]
        
        use_ray_pose = kwargs.get('use_ray_pose', False)
        infer_gs = kwargs.get('infer_gs', False)
        # Process features through depth head
        da3_model: DepthAnything3Net = self.da3.model
        with torch.autocast(device_type=imgs.device.type, enabled=False):
            output = da3_model._process_depth_head(feats, H, W)
            if use_ray_pose:
                output = da3_model._process_ray_pose_estimation(output, H, W)
            else:
                output = da3_model._process_camera_estimation(feats, H, W, output)
            if infer_gs:
                output = da3_model._process_gs_head(feats, H, W, output, imgs)
        
        #output = da3_model._process_mono_sky_estimation(output)
        return output

    def _extract_cls_token(self, cls_token: list[torch.Tensor], feat_layers: list[int]) -> Dict[str, torch.Tensor]:
        aux_cls = Dict()
        assert len(cls_token) == len(feat_layers), "Expected a set of cls tokens per layer to extract"
        for cls, feat_layer in zip(cls_token, feat_layers):
            cls_reshaped = cls.reshape(cls.shape[0], cls.shape[1], -1) #B, S, dim
            aux_cls[f"feat_layer_{feat_layer}"] = cls_reshaped
    
        return aux_cls

    def _extract_auxiliary_features(
        self,
        feats: list[torch.Tensor],
        feat_layers: list[int],
        H: int, W: int
    ) -> Dict[str, torch.Tensor]:
        """Extract auxiliary features from specified layers."""
        aux_features = Dict()
        assert len(feats) == len(feat_layers)
        for feat, feat_layer in zip(feats, feat_layers):
            # Reshape features to spatial dimensions
            feat_reshaped = feat.reshape(
                [
                    feat.shape[0],
                    feat.shape[1],
                    H // self.PATCH_SIZE,
                    W // self.PATCH_SIZE,
                    feat.shape[-1],
                ]
            )
            aux_features[f"feat_layer_{feat_layer}"] = feat_reshaped

        return aux_features

    def _format_output_for_salad(self, output: Dict[str, torch.Tensor], feat_layer: int) -> Tuple[torch.Tensor]:
        f = output.aux[f"feat_layer_{feat_layer}"] #Shape = B, S, h_tokens, w_tokens, dim
        B, S, h, w, dim = f.shape
        #We expect B=1 or S = 1
        assert B == 1 or S == 1, "Something wrong happened with features' shape"

        t = output.aux_cls[f"feat_layer_{feat_layer}"]
        B, S, t_dim = t.shape
        assert dim == t_dim == self.num_channels, "Something wrong happened with tokens dim"
        assert B == 1 or S == 1, "Something wrong happened with features' shape"

        f_reshaped = f.view(B*S, h, w, dim).permute(0, 3, 1, 2)
        t_reshaped = t.view(B*S, dim)

        return f_reshaped, t_reshaped

    def _prepare_inputs(
        self,
        x: torch.Tensor | List[str | Image.Image | np.ndarray],
    ) -> tuple:
        if isinstance(x, torch.Tensor):
            S, C, H, W = x.shape #Here, the sequence len will play as Batch size.
                                    #However, images may come from different scenes.
                                    #Rendering alternate attention non-sense at all.
                                    #Since only SALAD will be trained, it might mather nothing after all.
                                    #But we want a model wich returns both features for SALAD and 3D predictions.
                                    #I must remove alternate attention after I check same features are dropped.
                                    #But for applications, I need a model which predicts all.

            assert C == 3, "Number of channels mismatch"
            #process_res = W

            #da3 api expects a list of: np.ndarray, paths or PIL Images.
            #Here, conversion to np.ndarray is conducted
            image = list(x.mul(255).permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy())
            assert image[0].shape[2] == 3, "image (np.ndarray) shape mismatch"
        elif isinstance(x, list):
            image = x
        else:
            raise ValueError("Expected tensor or datatype compatible with da3 api.")

        return image

    def _preprocess_inputs(
        self,
        image: list[np.ndarray | Image.Image | str],
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Preprocess input images using input processor."""
        imgs_cpu, _, _ = self.input_processor(
            image,
            None, #extrinsics
            None, #intrinsics
            process_res,
            process_res_method,
        )
        return imgs_cpu

    @property
    def device(self) -> torch.device:
        return self._get_model_device()

    def _get_model_device(self) -> torch.device:
        """
        Get the device where the model is located.

        Returns:
            Device where the model parameters are located

        Raises:
            ValueError: If no tensors are found in the model
        """
        # Find device from parameters
        for param in self.parameters():
            return param.device

        # Find device from buffers
        for buffer in self.buffers():
            return buffer.device

        raise ValueError("No tensor found in model")


class DepthAnything3Dino(DepthAnything3Backbone):
    def __init__(
        self,
        da3: DepthAnything3,
        **kwargs
    ):
        super().__init__(da3, keep_da3=False)
        if 'num_trainable_blocks' in kwargs:
            print("num_trainable_blocks argument is not supported for da3 backbone. DA3 is used as is")
        if 'norm_layer' in kwargs:
            print("norm_layer argument flag is not supported for da3. DA3 is used as is")
        self._dino: DinoVisionTransformer = da3.model.backbone.pretrained

    @property
    def dino(self) -> DinoVisionTransformer:
        return self._dino

    @staticmethod
    def from_pretrained(model_name: str = "da3-base", **kwargs) -> "DepthAnything3Dino":
        da3 = da3_from_pretained(model_name, **kwargs)
        return DepthAnything3Dino(da3, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        feat_layer: int = -1, #FUTURE: must be a backbone config, i.e., add to yaml and pass in __init__
        process_res: int = -1,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        # 1. IMage preprocessing.
        image = self._prepare_inputs(x) #Convert torch tensor to np ndarray
        if process_res == -1:
            H, W, _ = image[0].shape
            process_res = max(H, W)
        imgs_cpu = self._preprocess_inputs(
            image,
            process_res = process_res
        )
        device = self._get_model_device()
        imgs = imgs_cpu.to(device, non_blocking=True)[None].float()

        if feat_layer == -1:
            feat_layer = self.dino.alt_start - 1
        assert feat_layer < self.dino.alt_start, "Double check what's the last layer before alternate attention"
        feat_layers = [feat_layer]

        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type=imgs.device.type, dtype=autocast_dtype):
            batch_shape = imgs.shape

            #2. Per view processing/patch embedding.
            _, aux_outputs = self._dino_attend(
                imgs,
                batch_shape,
                feat_layers,
                cam_token=None,
                **kwargs
            )
            
            aux_feats, cls_token = self._aux_layers_feats(aux_outputs)

            output = Dict()
            H, W = imgs.shape[-2], imgs.shape[-1]

            #Reshapes aux features of the given list of layers.
            #Each layer's aux features is reshped to Batch_size, sequence, num_vertical_patches, num horizontal patches, embed dim
            #Each layer's acx features is resotred in a dict, e.g. f"feat_layer_{feat_layer}"
            #output.aux is a dictionary 
            output.aux = self._extract_auxiliary_features(aux_feats, feat_layers, H, W)
            output.aux_cls = self._extract_cls_token(cls_token, feat_layers)

        #PATCH_SIZE = 14 (same in DINOv2)
        #Image resizing. Upper resize. Resize to 504.
        #Image is resized such as the largest dimension is 504.
        # Then, we make both dimensions divisible by the PATCH SIZE
        # by converting dimensions to the nearest multiple of the batch size.
        # This means that the processed dimensions depend on the input image.
        # So, also, the number of patches is not fixed.
        # However, SALAD works with a variable number of tokens, but a fixed number is required for comparison.
        # A trade-off is required to compare backbones' performance.
        # Note also that dino+salad is trained with square images, while da3 is not.

        #f is already detached.
        f_reshaped, t_reshaped = self._format_output_for_salad(output, feat_layer)

        return f_reshaped, t_reshaped
        

def intermediate_features(
    model: DepthAnything3,
    image: list[np.ndarray | Image.Image | str],
    process_res: int = 504,
    export_feat_layers: list = []
) -> Prediction:
    if not export_feat_layers:
        dino: DinoVisionTransformer = model.model.backbone.pretrained
        export_feat_layers = [dino.alt_start - 1]
    prediction = model.inference(
        image,
        process_res=process_res,
        export_feat_layers=export_feat_layers,
    )
    
    return prediction


#TODO:
# [x] Modify code to not drop cls token from auxiliar outputs.
# [x] Write code (with out modifying depth anything source code) which takes depth anything outputs before alternate attention blocks (including cls tokens)
# [x] Compare if results are the same. (one to one). Only if no resized is applied to input tensor
# [x] Train SALAD with ViT-Base.
# [x] Adapt code to use other ViT confs.
# [x] Double Check if patch tokens order remain constant. They do in the first release

