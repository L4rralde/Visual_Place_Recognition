# DINO + SALAD for Visual Place Recognition

This repo contains code to train SALAD with DINOv2/DINOv3 backbones. This repo is an evolution of my [fork](https://github.com/L4rralde/dinov3_salad/tree/cimat) of [DINO + SALAD](https://github.com/serizba/salad)

In addition to the original work "Optimal Transport Aggregation for Visual Place Recognition" I have added the following backbones:

- **DINOv3**. Check my results in this [link](https://www.linkedin.com/feed/update/urn:li:activity:7378879292006580224/).
- **Visual Geometry Grounded Trasnformer**.
- Not-metric configurations of **Depth Anything v3**: Small, Base, Large, Giant (with out nesting). I think recent versions of the repo that favor the image orsering scheme for the nested version have broken these versions, so this repo uses a checked out version.
- **MapAnything**. Trained with v1.1

Future work may include **Efficient
Universal Perception Encoder"**



## Available trained models:

| Model         | Backbone            | Patch Size | Embedding Dim | Hidden Dim | Pitts30k-val R1 | Pitts30k-val R5 | Ckpt size |
| ------------- | ------------------- | ---------- | ------------- | ---------- | --------------- | --------------- | --------- |
|**MapAnything-SALAD**|`mapanything` (ViT-G)| 14         | 1536          | **512**    | 0.93            | 0.98            |   9.88MB  | 
| **VGGT-SALAD**      | `vggt` (ViT-L)      | 14         | 1024          | **512**    | 0.92            | 0.98            |   6.89MB  |
| **DA3-SALAD Giant** | `da3-giant` (ViT-G) | 14         | 1536          | **1024**   | 0.89            | 0.96            |  19.80MB  |
| **DA3-SALAD Large** | `da3-large` (ViT-L) | 14         | 1024          | **512**    | 0.84            | 0.93            |   6.89MB  |
| **DA3-SALAD Base**  | `da3-base` (ViT-B)  | 14         | 768           | **512**    | 0.87            | 0.95            |   5.39MB  |
| **DA3-SALAD Small** | `da3-small` (ViT-S) | 14         | 384           | **512**    | 0.84            | 0.94            |   3.14MB  |


### Invoking the models:

```python
import torch
model = torch.hub.load(
    'L4rralde/Visual_Place_Recognition',
    <model_name>,
    <path_to_this_repo_clone>
)
```

Model names: `mapanything_salad`, `vggt_salad`, `da3_salad_giant`, `da3_salad_large`, `da3_salad_base`, `da3_salad_small`.

## My hardware limitations

My trained models haven't even reached the performance of DINOv2+SALAd. Even though I'm using the same training configuration and datasets whilst some of my backbones are even larger. The reason is I'm working with 16GB of VRAM. No attention blocks are trained (DINOv2+SALAD fine tunes 4 ViT Blocks). If possible, I'll migrate to a cloud provider (or borrow a workstation), and add ViT-blocks-like adapters and increase the resolution. The number of tokens used during training must be at least half the number of tokens using at inference.

Note that moving to another computer requires me to install all dependencies, and download and prepare the datasets, so it's not a small task.


## Installing

It's easter, I have no access to my computer to double check the requirements. I'll update this section soon. However, I must mention the installing procedure depends on the selected backbone.

## Training

python train_from_yaml.py --config "$file"

## TODO:
- [X] Da3 dino class does not match da3 auxiliar outputs. They do as long we do not preprocess the inputs.
- [x] Check if tokens' order remain constant. They do. A commit from december 2025 modified the order, but we will use an older version.
- [x] Latest version of da3 repo breaks some configurations. I think only nested works. Kept one before december.
- [x] File bug regarding da3, cpu memory and different image aspect ratios
- [ ] Clean up. Remove args that are not required, e.g., return_token. This must be always true.


## Contribute

If want to contribute, please send me an email. The current status will be part of my thesis, but further work would be omitted. You can help me to make this models reach (or set) the state of the art. Today, the bottleneck is RAM.
