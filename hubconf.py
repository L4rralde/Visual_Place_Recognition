import sys, os

import torch


DONWLOAD_URL = "https://github.com/L4rralde/Visual_Place_Recognition/releases/download/Da3-Vggt-Salad/"


def da3_salad_giant(vpr_repo_path: str, **kwargs) -> torch.nn.Module:
    dependencies = ['torch', 'DepthAnything3']
    sys.path.append(vpr_repo_path)
    sys.path.append(os.path.join(vpr_repo_path, "submodules", "Depth-Anything-3"))
    from model_flavors.da3_salad import DA3Salad
    from vpr.models.backbones.da3.da3 import da3_from_pretained

    backbone_arch = 'da3-giant'
    backbone_config = {
        "return_token": True
    }
    salad_config = {
        "cluster_dim": 128,
        "hidden_dim": 1024,
        "num_clusters": 64,
        "token_dim": 256
    }

    da3 = da3_from_pretained(backbone_arch, **kwargs)
    da3_salad = DA3Salad(da3, backbone_config, salad_config)
    url = f"{DONWLOAD_URL}/da3_salad_giant.pth"
    salad_state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    da3_salad.aggregator.load_state_dict(salad_state_dict)

    return da3_salad


#https://drive.google.com/file/d/1Bt7VM8uyayb2QTwvspau71ejjt6_aBpI/view?usp=drive_link
def vggt_salad(vpr_repo_path: str, **kwargs) -> torch.nn.Module:
    dependencies = ['torch', 'VGGT']
    sys.path.append(vpr_repo_path)
    sys.path.append(os.path.join(vpr_repo_path, "submodules", "vggt"))
    from model_flavors.vggt_salad import VggtSalad
    from vpr.models.backbones.vggt import load_pretrained_vggt

    backbone_arch = 'vggt'
    backbone_config = {
        "return_token": True
    }
    salad_config = {
        "cluster_dim": 128,
        "num_clusters": 64,
        "token_dim": 256
    }
    vggt = load_pretrained_vggt()
    vggt_salad = VggtSalad(vggt, backbone_config, salad_config)
    url = f"{DONWLOAD_URL}/vggt_salad.pth"
    salad_state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
    vggt_salad.aggregator.load_state_dict(salad_state_dict)

    return vggt_salad
