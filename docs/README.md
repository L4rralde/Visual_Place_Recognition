# DINO + SALAD for Visual Place Recognition

This repo contains code to train SALAD with DINOv2/DINOv3 backbones.


## Available models:

| Model         | Backbone            | Patch Size | Embedding Dim | Hidden Dim | Pitts30k-val R1 | Pitts30k-val R5 | Ckpt size |
| ------------- | ------------------- | ---------- | ------------- | ---------- | --------------- | --------------- | --------- |
| **VGGT-SALAD**      | `vggt` (ViT-L)      | 14         | 1024          | **512**    | 0.92            | 0.98            |   6.89MB  |
| **DA3-SALAD Giant** | `da3-giant` (ViT-g) | 14         | 1536          | **1024**   | 0.89            | 0.96            |  19.80MB  |
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

Model names: `vggt_salad`, `da3_salad_giant`, `da3_salad_large`, `da3_salad_base`, `da3_salad_small`.



## Training

python train_from_yaml.py --config "$file"

## TODO:
- [X] Da3 dino class does not match da3 auxiliar outputs. They do as long we do not preprocess the inputs.
- [x] Check if tokens' order remain constant. They do. A commit from december 2025 modified the order, but we will use an older version.
- [x] Latest version of da3 repo breaks some configurations. I think only nested works. Kept one before december.
- [x] Write tests to check if aux features are constant independent of the images used.
- [x] Write tests to check if aux features from da3 api match those from my class.
- [x] File bug regarding da3, cpu memory and different image aspect ratios
- [x] Write tests to check if the predictions for 3D reconstruction match (da3_salad vs da3).
- [x] Clean out. Remove args that are not required, e.g., return_token. This must be always true.
