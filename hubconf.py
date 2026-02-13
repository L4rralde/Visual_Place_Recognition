import torch


#https://drive.google.com/file/d/11SJJsdlW6_vLdKNR7zYbeJkN-NVDQC8N/view?usp=drive_link
def da3_salad_giant(**kwargs) -> torch.nn.Module:
    dependencies = ['torch', 'DepthAnything3']
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
    path = "/home/emmanuel/Desktop/tesis/Visual_Place_Recognition/logs/lightning_logs/version_46/salad.ckpt"
    salad_state_dict = torch.load(
        path,
        weights_only=False,
        map_location=torch.device('cpu')
    )
    da3_salad.aggregator.load_state_dict(salad_state_dict)

    return da3_salad


#https://drive.google.com/file/d/1Bt7VM8uyayb2QTwvspau71ejjt6_aBpI/view?usp=drive_link
def vggt_salad(**salad) -> torch.nn.Module:
    dependencies = ['torch', 'VGGT']
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
    path = "/home/emmanuel/Desktop/tesis/Visual_Place_Recognition/logs/lightning_logs/version_43/salad.ckpt"
    salad_state_dict = torch.load(
        path,
        weights_only=False,
        map_location=torch.device('cpu')
    )
    vggt_salad.aggregator.load_state_dict(salad_state_dict)

    return vggt_salad
