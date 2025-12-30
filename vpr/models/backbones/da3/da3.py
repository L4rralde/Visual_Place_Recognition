

from typing import Sequence

from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from addict import Dict

from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Prediction
from depth_anything_3.model.dinov2.vision_transformer import DinoVisionTransformer
from depth_anything_3.model.da3 import DepthAnything3Net


@torch.inference_mode()
class DepthAnything3Backbone(nn.Module):
    def __init__(self, model_name: str = "da3-base", from_block: int=1, **kwargs):
        self.from_block = from_block
        self.da3 = DepthAnything3(model_name, **kwargs)

    def forward(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        ref_view_strategy: str = "saddle_balanced",
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
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
        imgs_cpu, extrinsics, intrinsics = self.da3._preprocess_inputs(
            image, extrinsics, intrinsics, process_res, process_res_method
        )

        # Prepare tensors for model
        #This basically does: .to(device, non_blocking=True)[None].float() for each input
        imgs, ex_t, in_t = self.da3._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)

        # Normalize extrinsics
        # If ext_t is None, returns None.
        ex_t_norm = self.da3._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)

        # Run model forward pass

        #raw_output = self.da3._run_model_forward(
        #    imgs, ex_t_norm, in_t, export_feat_layers
        #)
        
        #Run model forward def
        feat_layers = list(export_feat_layers) if export_feat_layers is not None else []
        #output = self.da3.forward(imgs, ex_t, in_t, feat_layers)

        #def forward def
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad():
            with torch.autocast(device_type=imgs.device.type, dtype=autocast_dtype):
                #return self.da3.model(
                #    image, extrinsics, intrinsics, export_feat_layers
                #) #Returns a Prediction class object
            
                #Model forward def
                if ex_t_norm is not None:
                    with torch.autocast(device_type=imgs.device.type, enabled=False):
                        cam_token = self.cam_enc(ex_t_norm, in_t, imgs.shape[-2:])
                else:
                    cam_token = None
            
                #feats, aux_feats = self.backbone(
                #    x, cam_token=cam_token, export_feat_layers=export_feat_layers, ref_view_strategy=ref_view_strategy
                #)
                #H, W = x.shape[-2], x.shape[-1]

                #Creates a dictionary. Key is the extra layer, value are reshaped tokens without cls tokens.
                #output.aux = self._extract_auxiliary_features(aux_feats, export_feat_layers, H, W)

                #self.backbone.forward def

                #return self.pretrained.get_intermediate_layers(
                #    x,
                #    self.out_layers, #Out layers: n=1. Take the last layer
                #    **kwargs,
                #)#Here, the magic happens:
            
                #self.bakcbone.pretrained(dino).get_intermediate_layers def:

                #outputs, aux_outputs = self.da3.model.backbone.pretrained._get_intermediate_layers_not_chunked(
                #    x, n, export_feat_layers=export_feat_layers, **kwargs
                #)
                #camera_tokens = [out[0] for out in outputs]
                #if outputs[0][1].shape[-1] == self.embed_dim: #Idk y last dim could be self.embed_dim * 2. One way or the other, It doesnt matter since I only care about auxiliar layers.
                #    outputs = [self.norm(out[1]) for out in outputs]
                #elif outputs[0][1].shape[-1] == (self.embed_dim * 2):
                #    outputs = [
                #        torch.cat(
                #            [out[1][..., : self.embed_dim], self.norm(out[1][..., self.embed_dim :])],
                #            dim=-1,
                #        )
                #        for out in outputs
                #    ]
                #else:
                #    raise ValueError(f"Invalid output shape: {outputs[0][1].shape}")
                #aux_outputs = [self.norm(out) for out in aux_outputs] #We want to take this outputs (with the class tokens)
                #outputs = [out[..., 1 + self.num_register_tokens :, :] for out in outputs]
                #aux_outputs = [out[..., 1 + self.num_register_tokens :, :] for out in aux_outputs] #Here, cls tokens are dropped :(
                #return tuple(zip(outputs, camera_tokens)), aux_outputs

                # self.bakcbone.pretrained(dino)._get_intermediate_layers_not_chunked def
                #Let x be a batch of sequences. x.shape = (batch_size, sequence len, channels, height, width)
                dino: DinoVisionTransformer = self.da3.model.backbone.pretrained
                outputs, aux_outputs = dino._get_intermediate_layers_not_chunked(
                    imgs, self.from_block, feat_layers,
                    cam_token=cam_token,
                    ref_view_strategy=ref_view_strategy
                )
                camera_tokens = [out[0] for out in outputs]
                if outputs[0][1].shape[-1] == dino.embed_dim:
                    outputs = [dino.norm(out[1]) for out in outputs]
                elif outputs[0][1].shape[-1] == (dino.embed_dim * 2): #Idk why feature dim can be doubled, but one is normalized.
                    outputs = [
                        torch.cat(
                            [out[1][..., : dino.embed_dim], dino.norm(out[1][..., dino.embed_dim :])],
                            dim=-1,
                        )
                        for out in outputs
                    ]
                else:
                    raise ValueError(f"Invalid output shape: {outputs[0][1].shape}")
                aux_outputs = [dino.norm(out) for out in aux_outputs] #Applying final ViT norm layer
                outputs = [out[..., 1 + self.num_register_tokens :, :] for out in outputs] #Taking feat tokens only
                
                #Extra line of code
                cls_outputs = [out[..., 0, :] for out in aux_outputs]
                aux_outputs = [out[..., 1 + self.num_register_tokens :, :] for out in aux_outputs]

                feats = tuple(zip(outputs, camera_tokens))
                aux_feats = aux_outputs
                cls_token = cls_outputs

                da3_model: DepthAnything3Net = self.da3.model
                H, W = imgs.shape[-2], imgs.shape[-1]

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
                            output = da3_model._process_gs_head(feats, H, W, output, imgs, extrinsics, intrinsics)
                    
                    output = da3_model._process_mono_sky_estimation(output)    
                else:
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
        assert len(aux_cls) == len(feat_layers)
        for cls, feat_layer in zip(cls_token, feat_layers):
            cls_reshaped = cls.reshape(cls.shape[0], cls.shape[1], -1)
            aux_cls[f"feat_layer_{feat_layer}"] = cls_reshaped
    
        return aux_cls

    def inference(self) -> Prediction:
        #FUTURE
        raise NotImplementedError()
            
    
    #We better use 


def intermediate_features(
    model: DepthAnything3,
    image: list[np.ndarray | Image.Image | str],
    export_dir: str,
    export_feat_layers: Sequence[int] | None = [8], #Is 8 the last layer before alternate attention?
    extrinsics: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    ref_view_strategy: str = "saddle_balanced",
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    **kwargs
) -> Prediction:
    model.inference(
        image, extrinsics, intrinsics,
        export_dir=export_dir,
        ref_view_strategy=ref_view_strategy,
        process_res=process_res,
        process_res_method=process_res_method,
        export_feat_layers=export_feat_layers,
    )


#TODO:
# [ ] Modify code to not drop cls token from auxiliar outputs.
# [ ] Write code (with out modifying depth anything source code) which takes depth anything outputs before alternate attention blocks (including cls tokens)
# [ ] Compare if results are the same. (one to one).
# [ ] Train SALAD with ViT-Base.
# [ ] Adapt code to use other ViT confs.

