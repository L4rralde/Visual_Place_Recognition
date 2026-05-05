import sys, os
from dataclasses import dataclass

import torch


DOWNLOAD_URL = "https://github.com/L4rralde/Visual_Place_Recognition/releases/download/v0.2_weights/"
dependencies = ['torch']


@dataclass
class _Config:
    backbone_arch: str
    backbone_config: dict
    salad_config: dict
    url: str


_da3_giant_config = _Config(
    backbone_arch='da3-giant',
    backbone_config={"return_token": True},
    salad_config={
        "cluster_dim": 128,
        "hidden_dim": 1024,
        "num_clusters": 64,
        "token_dim": 256
    },
    url=f"{DOWNLOAD_URL}/da3_salad_giant.pth"
)

_da3_large_config = _Config(
    backbone_arch='da3-large',
    backbone_config={"return_token": True},
    salad_config={
        "cluster_dim": 128,
        "num_clusters": 64,
        "token_dim": 256
    },
    url=f"{DOWNLOAD_URL}/da3_salad_large.pth"
)

_da3_base_config = _Config(
    backbone_arch='da3-base',
    backbone_config={"return_token": True},
    salad_config={
        'cluster_dim': 128,
        'num_clusters': 64,
        'token_dim': 256,
    },
    url=f"{DOWNLOAD_URL}/da3_salad_base.pth"
)

_da3_small_config = _Config(
    backbone_arch='da3-small',
    backbone_config={"return_token": True},
    salad_config={
        'cluster_dim': 128,
        'num_clusters': 64,
        'token_dim': 256,
    },
    url=f"{DOWNLOAD_URL}/da3_salad_small.pth"
)

_vggt_config = _Config(
    backbone_arch='vggt',
    backbone_config={"return_token": True},
    salad_config={
        "cluster_dim": 128,
        "num_clusters": 64,
        "token_dim": 256
    },
    url=f"{DOWNLOAD_URL}/vggt_salad.pth"
)

_mapanything_config = _Config(
    backbone_arch='map_anything',
    backbone_config={"return_token": True},
    salad_config={
        "cluster_dim": 128,
        "num_clusters": 64,
        "token_dim": 256
    },
    url="{DOWNLOAD_URL}/mapanything_salad.pth"
)


def _da3_salad(config: _Config, vpr_repo_path, **kwargs) -> torch.nn.Module:
    if vpr_repo_path not in sys.path:
        sys.path.insert(0, vpr_repo_path)
        sys.path.insert(0, os.path.join(vpr_repo_path, "submodules", "Depth-Anything-3"))
    from model_flavors.da3_salad import DA3Salad
    from vpr.models.backbones.da3.da3 import da3_from_pretained
    backbone_arch = config.backbone_arch
    backbone_config = config.backbone_config
    salad_config = config.salad_config
    url = config.url
    da3 = da3_from_pretained(backbone_arch, **kwargs)
    da3_salad = DA3Salad(da3, backbone_config, salad_config)
    salad_state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    da3_salad.aggregator.load_state_dict(salad_state_dict)

    return da3_salad


def da3_salad_giant(vpr_repo_path: str, **kwargs) -> torch.nn.Module:
    return _da3_salad(_da3_giant_config, vpr_repo_path, **kwargs)


def da3_salad_large(vpr_repo_path: str, **kwargs) -> torch.nn.Module:
    return _da3_salad(_da3_large_config, vpr_repo_path, **kwargs)


def da3_salad_base(vpr_repo_path: str, **kwargs) -> torch.nn.Module:
    return _da3_salad(_da3_base_config, vpr_repo_path, **kwargs)


def da3_salad_small(vpr_repo_path: str, **kwargs) -> torch.nn.Module:
    return _da3_salad(_da3_small_config, vpr_repo_path, **kwargs)


def vggt_salad(vpr_repo_path: str, **kwargs) -> torch.nn.Module:
    if vpr_repo_path not in sys.path:
        sys.path.insert(0, vpr_repo_path)
        sys.path.insert(0, os.path.join(vpr_repo_path, "submodules", "vggt"))
    from model_flavors.vggt_salad import VggtSalad
    from vpr.models.backbones.vggt import load_pretrained_vggt

    backbone_arch = _vggt_config.backbone_arch
    backbone_config = _vggt_config.backbone_config
    salad_config = _vggt_config.salad_config
    vggt = load_pretrained_vggt()
    vggt_salad = VggtSalad(vggt, backbone_config, salad_config)
    url = _vggt_config.url
    salad_state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    vggt_salad.aggregator.load_state_dict(salad_state_dict)

    return vggt_salad


def mapanything_salad(vpr_repo_path: str, **kwargs) -> torch.nn.Module:
    if vpr_repo_path not in sys.path:
        sys.path.insert(0, vpr_repo_path)
        sys.path.insert(0, os.path.join(vpr_repo_path, "submodules", "map-anything"))
    from model_flavors.mapanything_salad import MapAnythingSalad
    from vpr.models.backbones.mapanything import load_pretrained_mapanything

    backbone_arch = _mapanything_config.backbone_arch
    backbone_config = _mapanything_config.backbone_config
    salad_config = _mapanything_config.salad_config
    mapanything = load_pretrained_mapanything()
    mapanything_salad = MapAnythingSalad(mapanything, backbone_config, salad_config)
    url = _mapanything_config.url
    salad_state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    mapanything_salad.aggregator.load_state_dict(salad_state_dict)

    return mapanything_salad
