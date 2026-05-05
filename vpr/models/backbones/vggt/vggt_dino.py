from typing import List, Dict, Tuple
import gc

from PIL import Image
import torch
import torch.nn as nn

from .transforms import preprocess_image
from .dino_blocks import vit_large_blocks
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vggt.models.vggt import VGGT
from vggt.models.aggregator import slice_expand_and_flatten
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def load_vggt_state_dict():
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state_dict = torch.hub.load_state_dict_from_url(_URL, map_location='cpu')
    return state_dict


def load_pretrained_vggt() -> VGGT:
    vggt = VGGT()
    state_dict = load_vggt_state_dict()
    vggt.load_state_dict(state_dict)
    return vggt


# The following code was adaptef from VGGT-SPARK's fork of VGGT
# I don't intend to use this code in the main repo, but I anyway added it
# to remember I did implemented it.
# Deprecated:
#    "This function was defined to experiment with " \
#    "VGGT alternate attention blocks for image matching "\
#    "Following VGGT-SLAM 2.0 alpha score. "\
#    "Since results were not good enough, I discarded "
#    "the changes made in VGGT code in this repo and added "\
#    "them to a new fork: https://github.com/L4rralde/vggt_score_match"\
#    "Namely, you must access to (attention) tensors q,k to compute" \
#    "the score and that does required to dig deep into VGGT" \
def mean_top_quarter(arr):
    import numpy as np
    # flatten to 1D
    flat = arr.ravel()
    # find 75th percentile threshold
    thresh = np.percentile(flat, 75)
    # select values >= threshold
    top_vals = flat[flat >= thresh]
    # return their mean
    return top_vals.mean()

# Deprecated:
#    "This function was defined to experiment with " \
#    "VGGT alternate attention blocks for image matching "\
#    "Following VGGT-SLAM 2.0 alpha score. "\
#    "Since results were not good enough, I discarded "
#    "the changes made in VGGT code in this repo and added "\
#    "them to a new fork: https://github.com/L4rralde/vggt_score_match"\
#    "Namely, you must access to (attention) tensors q,k to compute" \
#    "the score and that does required to dig deep into VGGT" \
def xattn_similarity(k, q, token_offset=5):
    assert k.shape == q.shape
    B, H, T, d = k.shape # B, batch size. H: heads, T: all (images) concatenated tokens
    
    num_imgs = 2
    tokens_per_img = T//num_imgs

    q = q.clone()
    q[:, :, :token_offset] = 0 #Zeroing cam and reg tokens
    q[:, :, tokens_per_img: tokens_per_img+token_offset] = 0 #Zeroing cam and reg tokens

    #Extract keys for the first image excluding cam and reg tokens
    k = k[:,:,token_offset:tokens_per_img , :] #K(1) (1) Denotes from image 1. With out reg and cls tokens

    attn = q @ k.transpose(-2, -1) # [Q(1) \\ Q(2)] @ K(1)^T = [Q(1)K(1)^T \\ Q(2)K(1)^T ]
    #print(attn.shape) # B, H, T, N
    attn = attn.transpose(-2, -1) # [K(1)Q(1)^T  K(1)Q(2)^T]
    # print(attn.shape) # B, H, N, T
    attn = attn.softmax(dim=-1) #softmax across all q
 
    #print(attn.shape) #B, H, N, T
    attn = attn.mean(dim=1) #Average across all heads. New shape is B, N, T
    #print(attn.shape) #B, N, T

    all_token_to_first_frame = attn[..., :tokens_per_img]  # K(1)Q(1)^T. New sahpe is B, N, T/2
    all_token_to_second_frame = attn[..., tokens_per_img:] # # K(1)Q(2)^T. New shape is B, N, T/2
    #print(all_token_to_first_frame.shape) #B, N, T/2
    #print(all_token_to_second_frame.shape) #B, N, T/2

    max_per_token_first_img, _ = all_token_to_first_frame.max(dim=-1)
    #print(max_per_token_first_img.shape) # B, N
    # max_per_token_second_img = all_token_to_second_frame.max(dim=-1)[0]  #What's 0 for? No, max returns a tuple: (values, idcs)

    attn_second_frame_normalized = all_token_to_second_frame / (max_per_token_first_img.unsqueeze(-1) + 1e-6) #B, N, T/2
    ratio, _ = attn_second_frame_normalized.max(dim=1) #B, T/2. For which token of Query

    #print(ratio.shape)

    # ratio = max_per_token_second_img / (max_per_token_first_img + 1e-8)

    ratio =  ratio.float().detach().cpu().numpy() #B, T/2
    ratio_list = [mean_top_quarter(r) for r in ratio] # Each r is of shape T/2

    #print("Average of top quarter attention values (all frames):", ratio_list)
    # print("First Frame, Second Frame:", avg_top_quarter_first_img, avg_top_quarter_second_img)
    # plt.figure(figsize=(6,6))
    # plt.imshow(max_attn[1].reshape(image_height, image_width))
    # plt.colorbar()
    # plt.show()

    return ratio_list


class VggtBase(nn.Module):
    PATCH_SIZE = 14
    def __init__(
        self,
        vggt: VGGT,
        **kwargs
    ):
        super().__init__()
        if 'num_trainable_blocks' in kwargs:
            print("num_trainable_blocks argument is not supported for VGGT backbone. VGGT is used as is")
        self.norm_layer = kwargs.get('norm_layer', True)
        self.probing_from_layer: int = kwargs.get('probing_from_layer', -1)
        self.adapter_depth: int = kwargs.get('adapter_depth', 0)
        self.num_channels = vggt.aggregator.patch_embed.embed_dim
        self._resnet_std = vggt.aggregator._resnet_std
        self._resnet_mean = vggt.aggregator._resnet_mean
        self._vggt: VGGT|None = None
        self._dino = None

    @property
    def vggt(self) -> VGGT:
        if self._vggt is None:
            raise RuntimeError("self.vggt is not set in this class")
        return self._vggt

    @property
    def dino(self) -> nn.Module:
        if self._dino is not None:
            return self._dino
        if self._vggt is not None:
            return self._vggt.aggregator.patch_embed
        raise RuntimeError("self.dino is not set in this class")

    def make_adapter(self, adapter_depth: int=0) -> nn.Module:
        assert adapter_depth >= 0

        if adapter_depth == 0:
            return nn.Identity()
        
        n_used_blocks = self.probing_from_layer + 1
        assert self.dino.n_blocks >= n_used_blocks + adapter_depth, \
            f"Model depth ({self.dino.n_blocks}) is too shallow for probing layer {self.probing_from_layer} and adapter depth {adapter_depth}."
        
        dino_blocks_idcs_for_adapter = [
            self.probing_from_layer + i + 1
            for i in range(adapter_depth)
        ]
        adapter = vit_large_blocks(self.dino, dino_blocks_idcs_for_adapter)
        return adapter

    def inference(self, img_path_list: List[str]) -> dict:
        assert torch.cuda.is_available(), "Sadly, only works with cuda"
        DEVICE = "cuda"
        gc.collect() #Collect garbage
        torch.cuda.empty_cache()

        images = load_and_preprocess_images(img_path_list).to(DEVICE)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast(DEVICE, dtype=dtype):
                predictions = self.forward(images)
        extrinsic, intrinsic = self.pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            images.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        torch.cuda.empty_cache()

        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                predictions[key] = value.cpu().numpy().squeeze(0)

        return predictions

    def dino_forward(self, images: torch.Tensor) -> torch.Tensor:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean.to('cuda')) / self._resnet_std.to('cuda')

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.dino_forward_features(images) #This is all we need.

        return patch_tokens

    def dino_forward_features(self, x, masks=None):
        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x, masks)
            for i, blk in enumerate(self.dino.blocks):
                x = blk(x)
                if i == self.probing_from_layer:
                    x_for_salad = x.clone()

        x_for_salad = self.adapter(x_for_salad)

        with torch.no_grad():
            if self.norm_layer:
                x_for_salad = self.dino.norm(x_for_salad)
            x_norm = self.dino.norm(x)
            
        
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.dino.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.dino.num_register_tokens + 1 :],
            "x_salad_clstoken": x_for_salad[:, 0],
            "x_salad_patchtokens": x_for_salad[:, self.dino.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def pair_patch_tokens_with_ref(self, images: torch. Tensor, patch_tokens: torch.Tensor) -> tuple:
        B, S, C_in, H, W = images.shape
        BS, P, C = patch_tokens.shape

        assert BS == B*S
        assert S > 1
        if S == 2:
            return images, patch_tokens
        #B, S, C_in, H, W -> B, S, 1, C_in, H, W, -> B, S-1, 2, C_in, H, W -> B*(S-1), 2, C_in, H, W
        #BS, P, C -> B, S, P, C -> B, S, P, C -> B, S, 1, P, C -> B, S-1, 2, P, C -> B*(S-1), 2, P, C

        first_image = images[:, 0:1, ...] #B, 1, C_in, H, W
        rest_images = images[:, 1:, ...] #B, S-1, C_in, H, W
        first_expanded = first_image.expand(-1, S-1, -1, -1, -1)
        image_pairs = torch.stack([first_expanded, rest_images], dim=2) #B, S-1, 2, C_in, H, W
        image_pairs = image_pairs.reshape(B*(S-1), 2, C_in, H, W) #B*(S-1), 2, C_in, H, W

        patch_tokens_batched = patch_tokens.view(B, S, P, C)
        first_image_tokens = patch_tokens_batched[:, 0:1, ...] #B, 1, P, C
        rest_tokens = patch_tokens_batched[:, 1:, ...] #B, S-1, P, C
        first_image_tokens_expanded = first_image_tokens.expand(-1, S-1, -1, -1) #B, S-1, P, C
        paired_tokens = torch.stack(
            [first_image_tokens_expanded, rest_tokens],
            dim=2
        ) #B, S-1, 2, P, C
        paired_tokens = paired_tokens.reshape(B*(S-1), 2, P, C) #B*(S-1), 2, P, C
        paired_tokens = paired_tokens.view(-1, P, C) #B*(S-1)*2, P, C
        return image_pairs, paired_tokens

    def pairwise_prediction(self, images: torch.Tensor) -> dict:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        patch_tokens = self.dino_forward(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        img_pairs, paired_tokens = self.pair_patch_tokens_with_ref(images, patch_tokens)

        aggregated_tokens_list, patch_start_idx = self.alternate_attention(img_pairs, paired_tokens)
        predictions = self.heads_forward(
            img_pairs, aggregated_tokens_list, patch_start_idx, query_points=None
        )
        return predictions

    def alternate_attention(self, images: torch.Tensor, patch_tokens: torch.Tensor) -> tuple:
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self._vggt.aggregator.camera_token, B, S)
        register_token = slice_expand_and_flatten(self._vggt.aggregator.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self._vggt.aggregator.rope is not None:
            pos = self._vggt.aggregator.position_getter(
                B * S,
                H // self._vggt.aggregator.patch_size,
                W // self._vggt.aggregator.patch_size,
                device=images.device
            )

        if self._vggt.aggregator.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self._vggt.aggregator.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self._vggt.aggregator.aa_block_num):
            for attn_type in self._vggt.aggregator.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._vggt.aggregator._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._vggt.aggregator._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self._vggt.aggregator.patch_start_idx

    def heads_forward(self, images: torch.Tensor, aggregated_tokens_list: List[torch.Tensor], patch_start_idx: int, query_points:torch.Tensor) -> Dict[str, torch.Tensor]:
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self._vggt.camera_head is not None:
                pose_enc_list = self._vggt.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self._vggt.depth_head is not None:
                depth, depth_conf = self._vggt.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self._vggt.point_head is not None:
                pts3d, pts3d_conf = self._vggt.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self._vggt.track_head is not None and query_points is not None:
            track_list, vis, conf = self._vggt.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        
        patch_tokens = self.dino_forward(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        aggregated_tokens_list, patch_start_idx = self.alternate_attention(images, patch_tokens)
        predictions = self.heads_forward(images, aggregated_tokens_list, patch_start_idx, query_points)

        return predictions

    def prepare_tokens_for_salad(self, patch_tokens: Dict[str, torch.Tensor], images_shape: Tuple[int]) -> Tuple[torch.Tensor]:
        B, S, C_in, H, W = images_shape

        f = patch_tokens['x_salad_patchtokens']
        t = patch_tokens['x_salad_clstoken']
        
        f = f.reshape((B*S, H//14, W//14, self.num_channels)).permute(0, 3, 1, 2)

        return f, t

    def pose_encoding_to_extri_intri(self, pose_encoding: torch.Tensor, image_size_hw: tuple) -> tuple:
        return pose_encoding_to_extri_intri(pose_encoding, image_size_hw)

    def preprocess_image(self, img_list: Image.Image) -> Image.Image:
        return preprocess_image(img_list)


class VggtBackbone(VggtBase):
    def __init__(self, vggt, **kwargs):
        super().__init__(vggt, **kwargs)
        self._vggt = vggt
        self.adapter = self.make_adapter(self.adapter_depth)
        if self.probing_from_layer < 0:
            self.probing_from_layer = self.dino.n_blocks + self.probing_from_layer
        assert 0 <= self.probing_from_layer < self.dino.n_blocks, \
            "Index probing_from_layer out of range"

    @staticmethod
    def from_pretrained(**kwargs) -> "VggtBackbone":
        vggt = load_pretrained_vggt()
        return VggtBackbone(vggt, **kwargs)


class VggtDino(VggtBase):
    def __init__(self, vggt: VGGT, norm_layer: bool=True, **kwargs):
        super().__init__(vggt, norm_layer=norm_layer, **kwargs)
        self._dino = vggt.aggregator.patch_embed
        self.adapter = self.make_adapter(self.adapter_depth)
        if self.probing_from_layer < 0:
            self.probing_from_layer = self.dino.n_blocks + self.probing_from_layer
        assert 0 <= self.probing_from_layer < self.dino.n_blocks, \
            "Index probing_from_layer out of range"

    @staticmethod
    def from_pretrained(**kwargs) -> "VggtDino":
        vggt = VGGT()        
        full_state = load_vggt_state_dict()

        # Filter weights: Keep ONLY the dino parts
        # This assumes the prefix in the state_dict matches the module structure
        prefix = "aggregator.patch_embed."
        dino_state = {k: v for k, v in full_state.items() if k.startswith(prefix)}
        
        # Free the huge full_state dictionary immediately
        del full_state
        gc.collect() 
        
        # Load strictly the filtered weights
        # We must use strict=False because we are intentionally missing the rest of the model
        keys = vggt.load_state_dict(dino_state, strict=False)
        #print(f"VRAM Optimization: Loaded only {len(dino_state)} keys. Discarded the rest.")

        # Create the backbone
        backbone = VggtDino(vggt, **kwargs)
        
        # Final Cleanup
        del vggt
        del dino_state
        gc.collect()
        
        return backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, S, C_in, H, W = images.shape
        assert C_in == 3, "Wrong torch image format"

        patch_tokens = self.dino_forward(images)
        
        #x_norm_patchtokens shape = B*S, Total patches, channles
        #x_norm_clstoken shape: B*S, channles
        f, t = self.prepare_tokens_for_salad(patch_tokens, (B, S, C_in, H, W))

        return f, t
