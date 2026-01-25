#This code is of mine, not from DINOv2+SALAD.

import yaml


def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Validate required fields
    required_fields = [
        'backbone_arch', 'backbone_config', 'agg_config',
        'input_config', 'max_epochs',
    ]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in YAML: {field}")
    
    is_pure_dino = config['backbone_arch'].lower().startswith('dino')

    # Validate backbone_config subfields
    backbone_config_fields = ['return_token']
    backbone_config = config['backbone_config']
    if 'frozen' in backbone_config:
        if not backbone_config['frozen']:
            raise ValueError("If frozen is passed, it must be set to true. Other values are not allowed")
    else:
        backbone_config_fields.append('num_trainable_blocks')

    if is_pure_dino: #TODO. Add normalization layer to DA3
        backbone_config_fields.append('norm_layer')

    for field in backbone_config_fields:
        if field not in backbone_config:
            raise ValueError(f"Missing required backbone_config field in YAML: {field}")

    # Validate input_config subfields
    input_config_fields = ['img_size']
    for field in input_config_fields:
        if field not in config['input_config']:
            raise ValueError(f"Missing required input_config field in YAML: {field}")

    agg_config_fields = ['num_clusters', 'cluster_dim', 'token_dim']
    for field in agg_config_fields:
        if field not in config['agg_config']:
            raise ValueError(f"Missing required agg_config field in YAML: {field}")

    return config
