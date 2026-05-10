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

### v0.2 Release Notes

All models were retrained using half the number of tokens used at test time. 
All models improved, but DA3-SALAD Large and MapAnything-SALAD.
VGGT-SALAD became the best performing model
I'd expect better metrics at test stage (comparing against previous weights).
I'll benchmark all models when finished evaluating different configurations.
Currently I'm working on VGGT-SALAD++, which will make the unit-tests to fail
because of different features are being added.
All models were trained for 4 epochs.

Note. I must set all hidden dims to (max) 512. Using larger hiddnen dims increases the probability of easily overfitting.


| Model         | Backbone            | Patch Size | Embedding Dim | Hidden Dim | Pitts30k-val R1 | Pitts30k-val R5 | Ckpt size |
| ------------- | ------------------- | ---------- | ------------- | ---------- | --------------- | --------------- | --------- |
| **VGGT-SALAD**      | `vggt` (ViT-L)      | 14         | 1024          | **512**    | 0.932            | 0.987            |   6.89MB  |
|**MapAnything-SALAD**|`mapanything` (ViT-G)| 14         | 1536          | **512**    | 0.929            | 0.983            |   9.88MB  | 
| **DA3-SALAD Giant** | `da3-giant` (ViT-G) | 14         | 1536          | **1024**   | 0.90            | 0.97            |  19.80MB  |
| **DA3-SALAD Large** | `da3-large` (ViT-L) | 14         | 1024          | **512**    | 0.84            | 0.93            |   6.89MB  |
| **DA3-SALAD Base**  | `da3-base` (ViT-B)  | 14         | 768           | **512**    | 0.88            | 0.96            |   5.39MB  |
| **DA3-SALAD Small** | `da3-small` (ViT-S) | 14         | 384           | **512**    | 0.85            | 0.94            |   3.14MB  |


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

### VGGT probing Release Notes

DINO block 19 was used to produce input tokens for SALAD. Results are shown below:

| Model          | Adpater depth | Pitts30k-val R1 | Pitts30k-val R5 | Ckpt size |
| -------------- | ------------- | --------------- | --------------- | --------- |
| VGGT_L19       | NA            | 95.27           | 99.12           | 6.89MB    |
| VGGT_L19_A     | 2             | 95.39           | 99.04           | 104MB     |
| VGGT_L19_ADeep | 4             | 95.91           | 99.20           | 200MB     |

To use these models:

```python
import torch
model = torch.hub.load(
    'L4rralde/Visual_Place_Recognition',
    <vggt_probed_model_name>,
    <path_to_this_repo_clone>
)
```

vggt_probed_model_name: `vggt_l19_salad`, `vggt_l19_salad_adapters`, `vggt_l19_salad_deep_adapters`.




## My hardware limitations

My trained models haven't even reached the performance of DINOv2+SALAd. Even though I'm using the same training configuration and datasets whilst some of my backbones are even larger. The reason is I'm working with 16GB of VRAM. No attention blocks are trained (DINOv2+SALAD fine tunes 4 ViT Blocks). If possible, I'll migrate to a cloud provider (or borrow a workstation), and add ViT-blocks-like adapters and increase the resolution. The number of tokens used during training must be at least half the number of tokens using at inference.

Note that moving to another computer requires me to install all dependencies, and download and prepare the datasets, so it's not a small task.


## Installing

The instructions depend on the backbone you want to use.
As April the 20th, Code for DINOv2, DINOv3, VGGT, DepthAnythingv3 and MapAnything as backbones have been developed, yet EUPE is planned.

First, regardless the backbone you want to train, install base requirements in a new python virtual environment (tested on python3.10).

```bash
python3.10 -m venv salad.venv
source salad.venv/bin/activate
pip install -r requirements/base.txt
```

Then, install the specific requirements depending on the backbone, e.g., vggt:

```bash
pip install -r requirements/vggt.txt
```

By the moment I have added requirements for training SALAD with VGGT and DepthAnythingV3

## Downloading the Datasets:

For training and validation steps GSVCities and Pitts30k are required

- **GSVCities**

You may find it on [Kaggle](https://www.kaggle.com/datasets/amaralibey/gsv-cities).


- **Pittsburgh 250k**

Available on many websites inclusing [Kaggle](https://www.kaggle.com/datasets/duongoku/pittsburgh250k).
Make sure pittsburgh dataset tree is correct. For instance, when downloading from kaggle and unzipping,
you may find repeated hierarchies such as `pittsburgh/000/000` when it should be just `pittsburgh/000/`.
Also, move `pittsburgh/netvlad_v100_datasets/datasets/` to `pittsburgh/`


## Training

1. Set `$VPR_GIT_ROOT` environment variable:

```bash
source set_env_vars.sh
```

2. Run the trainer script. `$file` depends on the backbone you want to use and the configuration.
You may find available training configurations in `training_configs/` directory

```bash
python train_from_yaml.py --config "$file"
````

## TODO:
- [X] Da3 dino class does not match da3 auxiliar outputs. They do as long we do not preprocess the inputs.
- [x] Check if tokens' order remain constant. They do. A commit from december 2025 modified the order, but we will use an older version.
- [x] Latest version of da3 repo breaks some configurations. I think only nested works. Kept one before december.
- [x] File bug regarding da3, cpu memory and different image aspect ratios
- [ ] Clean up. Remove args that are not required, e.g., return_token. This must be always true.


## Contribute

If want to contribute, please send me an email. The current status will be part of my thesis, but further work would be omitted. You can help me to make this models reach (or set) the state of the art. Today, the bottleneck is RAM.
