
import os
from typing import Sequence
from typing import List, Dict, Tuple

from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from addict import Dict

from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Prediction
from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer
from depth_anything_3.model.da3 import DepthAnything3Net


#FOR COMPARISON
#save_dir = '/media/emmanuel/nvme_storage/da3_salad_data'

def da3_from_pretained(model_name: str, **kwargs) -> DepthAnything3:
    return DepthAnything3.from_pretrained(f"depth-anything/{model_name}")


class DepthAnything3Backbone(nn.Module):
    PATCH_SIZE: int = 14
    def __init__(self, da3: DepthAnything3):
        super().__init__()
        self.da3: DepthAnything3 = da3
    
    @property
    def dino(self) -> DinoVisionTransformer:
        return self.da3.model.backbone.pretrained

    @staticmethod
    def from_pretrained(model_name: str = "da3-base", **kwargs) -> "DepthAnything3Backbone":
        da3 = da3_from_pretained(model_name, **kwargs)
        return DepthAnything3Backbone(da3)

    def forward(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        export_feat_layers: Sequence[int] | None = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        return self.da3_inference(
            image,
            extrinsics,
            intrinsics,
            process_res,
            export_feat_layers,
            **kwargs
        )

    def da3_inference(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        export_feat_layers: Sequence[int] | None = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        #FUTURE. What role do the extrinsics and intrinsics play before alternate attention blocks?
        # Answer: Preprocess images, later, only if BOTH are passed, cls tokens are replaced by a 
        #           cam_token built upon intrinsics and extrinsics. Otherwise, class tokens are replaced
        #           by a(one?) trainble cam_token(s?).
        #           So, it's likely that those only affect preprocessing. And can't affect images at all.
        #Images are to be reshaped. Intrinsics need to be modified. Intrinsics shouldn't modify images...
        #   Extrinsics don't change at all, right?
        assert process_res != -1 , "A valid value must be passed"
        imgs_cpu, extrinsics, intrinsics = self.da3._preprocess_inputs(
            image, extrinsics, intrinsics, process_res
        )

        # Prepare tensors for model
        #This basically does: .to(device, non_blocking=True)[None].float() for each input
        imgs, ex_t, in_t = self.da3._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)

        # Normalize extrinsics
        # If ext_t is None, returns None.
        ex_t_norm = self.da3._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)
        
        feat_layers = list(export_feat_layers) if export_feat_layers is not None else []

        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad():
            with torch.autocast(device_type=imgs.device.type, dtype=autocast_dtype):            
                if ex_t_norm is not None:
                    with torch.autocast(device_type=imgs.device.type, enabled=False):
                        cam_token = self.da3.model.cam_enc(ex_t_norm, in_t, imgs.shape[-2:])
                else:
                    cam_token = None

                outputs, aux_outputs = self.dino._get_intermediate_layers_not_chunked( #With DINOv2 and DINOv3 (+ SALAD) we only compute what we need. This does alternate attention. May be overhead.
                    imgs,
                    n=self.da3.model.backbone.out_layers,
                    export_feat_layers=feat_layers,
                    cam_token=cam_token,
                )
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
                aux_outputs = [self.dino.norm(out) for out in aux_outputs] #Applying final ViT norm layer
                outputs = [out[..., 1 + self.dino.num_register_tokens :, :] for out in outputs] #Taking feat tokens only
                
                #Extra line of code
                cls_outputs = [out[..., 0, :] for out in aux_outputs]
                aux_outputs = [out[..., 1 + self.dino.num_register_tokens :, :] for out in aux_outputs]

                feats = tuple(zip(outputs, camera_tokens))
                aux_feats = aux_outputs
                cls_token = cls_outputs

                H, W = imgs.shape[-2], imgs.shape[-1]

                da3_model: DepthAnything3Net = self.da3.model
                if kwargs.get("export_depth", False):
                    use_ray_pose = kwargs.get('use_ray_pose', False)
                    infer_gs = kwargs.get('infer_gs', False)
                    # Process features through depth head
                    with torch.autocast(device_type=imgs.device.type, enabled=False):
                        output = da3_model._process_depth_head(feats, H, W)
                        if use_ray_pose:
                            output = da3_model._process_ray_pose_estimation(output, H, W)
                        else:
                            output = da3_model._process_camera_estimation(feats, H, W, output)
                        if infer_gs:
                            output = da3_model._process_gs_head(feats, H, W, output, imgs, ex_t_norm, in_t)
                    
                    #output = da3_model._process_mono_sky_estimation(output)    
                else:
                    output = Dict()

                #Reshapes aux features of the given list of layers.
                #Each layer's aux features is reshped to Batch_size, sequence, num_vertical_patches, num horizontal patches, embed dim
                #Each layer's acx features is resotred in a dict, e.g. f"feat_layer_{feat_layer}"
                #output.aux is a dictionary 
                output.aux = da3_model._extract_auxiliary_features(aux_feats, feat_layers, H, W)
                output.aux_cls = self._extract_cls_token(cls_token, feat_layers)

        #Adding pre-processed images:
        output = self.da3._add_processed_images(output, imgs_cpu)
        return output

    def _dino_attend(self, x, export_feat_layers=[], **kwargs):
        assert self.dino.alt_start != -1, "Alternate start needed"
        assert isinstance(x, torch.Tensor), "Expected input of type tensor"
        B, S, _, H, W = x.shape
        x = self.dino.prepare_tokens_with_masks(x)
        aux_output = []
        pos, pos_nodiff = self.dino._prepare_rope(B, S, H, W, x.device)

        for i, blk in enumerate(self.dino.blocks):
            if i == self.dino.alt_start:
                break
            if i < self.dino.rope_start or self.dino.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos
            x = self.dino.process_attention(x, blk, 'local', pos=l_pos)

            if i in export_feat_layers:
                aux_output.append(x)
        return x, aux_output, l_pos, g_pos

    def _alt_attend(self, x, n=1, export_feat_layers=[], **kwargs):
        #l_pos, g_pos, are they constants one it is calculated first?
        B, S, _, H, W = x.shape
        output, total_block_len, aux_output = [], len(self.blocks), []
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        pos, pos_nodiff = self._prepare_rope(B, S, H, W, x.device)

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
                out_x = torch.cat([local_x, x], dim=-1) if self.cat_token else x
                output.append((out_x[:, :, 0], out_x))
            if i in export_feat_layers:
                aux_output.append(x)
        return output, aux_output

    def _get_intermediate_layers_not_chunked(self, x, n=1, export_feat_layers=[], **kwargs):
        x, aux_output_bf_alt, l_pos, g_pos = self._dino_attend(x, export_feat_layers, **kwargs)
        output, aux_output_af_alt = self._alt_attend(x, n, export_feat_layers, **kwargs)
        
        aux_output = aux_output_bf_alt + aux_output_af_alt
        return output, aux_output

    def dino_only_inference(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        export_feat_layers: Sequence[int] | None = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        #FUTURE. What role do the extrinsics and intrinsics play before alternate attention blocks?
        # Answer: Preprocess images, later, only if BOTH are passed, cls tokens are replaced by a 
        #           cam_token built upon intrinsics and extrinsics. Otherwise, class tokens are replaced
        #           by a(one?) trainble cam_token(s?).
        #           So, it's likely that those only affect preprocessing. And can't affect images at all.
        #Images are to be reshaped. Intrinsics need to be modified. Intrinsics shouldn't modify images...
        #   Extrinsics don't change at all, right?
        assert process_res != -1 , "A valid value must be passed"
        imgs_cpu, extrinsics, intrinsics = self.da3._preprocess_inputs(
            image, extrinsics, intrinsics, process_res
        )

        # Prepare tensors for model
        #This basically does: .to(device, non_blocking=True)[None].float() for each input
        imgs, ex_t, in_t = self.da3._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)

        # Normalize extrinsics
        # If ext_t is None, returns None.
        ex_t_norm = self.da3._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)

        feat_layers = list(export_feat_layers) if export_feat_layers is not None else []

        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad():
            with torch.autocast(device_type=imgs.device.type, dtype=autocast_dtype):
                if ex_t_norm is not None:
                    with torch.autocast(device_type=imgs.device.type, enabled=False):
                        cam_token = self.da3.model.cam_enc(ex_t_norm, in_t, imgs.shape[-2:])
                else:
                    cam_token = None

                _, aux_outputs, _, _ = self._dino_attend(
                    imgs,
                    feat_layers,
                    cam_token=cam_token
                )
                aux_outputs = [self.dino.norm(out) for out in aux_outputs] #Applying final ViT norm layer
                
                #Extra line of code
                cls_outputs = [out[..., 0, :] for out in aux_outputs]
                aux_outputs = [out[..., 1 + self.dino.num_register_tokens :, :] for out in aux_outputs]

                aux_feats = aux_outputs
                cls_token = cls_outputs

                H, W = imgs.shape[-2], imgs.shape[-1]

                da3_model: DepthAnything3Net = self.da3.model
                output = Dict()

                #Reshapes aux features of the given list of layers.
                #Each layer's aux features is reshped to Batch_size, sequence, num_vertical_patches, num horizontal patches, embed dim
                #Each layer's acx features is resotred in a dict, e.g. f"feat_layer_{feat_layer}"
                #output.aux is a dictionary 
                output.aux = da3_model._extract_auxiliary_features(aux_feats, feat_layers, H, W)
                output.aux_cls = self._extract_cls_token(cls_token, feat_layers)

        return output

    def _extract_cls_token(self, cls_token: list[torch.Tensor], feat_layers: list[int]) -> Dict[str, torch.Tensor]:
        aux_cls = Dict()
        assert len(cls_token) == len(feat_layers), "Expected a set of cls tokens per layer to extract"
        for cls, feat_layer in zip(cls_token, feat_layers):
            cls_reshaped = cls.reshape(cls.shape[0], cls.shape[1], -1) #B, S, dim
            aux_cls[f"feat_layer_{feat_layer}"] = cls_reshaped
    
        return aux_cls


class DepthAnything3Dino(DepthAnything3Backbone):
    def __init__(
        self,
        da3: DepthAnything3,
        return_token: bool=False,
        training_salad: bool=False,
        **kwargs
    ):
        super().__init__(da3)
        self.return_token = return_token
        if 'num_trainable_blocks' in kwargs:
            print("num_trainable_blocks argument is not supported for da3 backbone. DA3 is used as is")
        if 'norm_layer' in kwargs:
            print("norm_layer argument flag is not supported for da3. DA3 is used as is")
        self.training_salad = training_salad
        self.num_channels = self.dino.num_features
        self.dino_alt_start = self.dino.alt_start

    @staticmethod
    def from_pretrained(model_name: str = "da3-base", return_token: bool=False, **kwargs) -> "DepthAnything3Dino":
        da3 = da3_from_pretained(model_name, **kwargs)
        return DepthAnything3Dino(da3, return_token, **kwargs)

    def forward(
        self,
        x: torch.Tensor | List[str | Image.Image | np.ndarray],
        feat_layer: int = -1, #FUTURE: must be a backbone config, i.e., add to yaml and pass in __init__
        process_res: int = -1,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        image, extrinsics, intrinsics = self._prepare_inputs(x, extrinsics, intrinsics)

        if feat_layer == -1:
            feat_layer = self.dino.alt_start -1
        assert feat_layer < self.dino.alt_start, "Double check what's the last layer before alternate attention"

        if process_res == -1:
            H, W, _ = image[0].shape
            process_res = max(H, W)

        if self.training_salad:
            output = self.dino_only_inference(
                image, extrinsics, intrinsics, process_res,
                export_feat_layers=[feat_layer], **kwargs
            )
        else:
            output = self.da3_inference(
                image, extrinsics, intrinsics, process_res,
                export_feat_layers=[feat_layer], **kwargs
            )

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

        if self.return_token:
            return f_reshaped, t_reshaped
        return f_reshaped

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
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
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

        if extrinsics is not None:
            extrinsics = extrinsics.cpu().numpy()
        if intrinsics is not None:
            intrinsics = intrinsics.cpu().numpy()

        return image, extrinsics, intrinsics
        

def intermediate_features(
    model: DepthAnything3,
    image: list[np.ndarray | Image.Image | str],
    extrinsics: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    process_res: int = 504,
    export_feat_layers: list = []
) -> Prediction:
    if not export_feat_layers:
        dino: DinoVisionTransformer = model.model.backbone.pretrained
        export_feat_layers = [dino.alt_start - 1]
    prediction = model.inference(
        image, extrinsics, intrinsics,
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

