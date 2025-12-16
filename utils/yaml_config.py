import yaml


def load_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Validate required fields
    required_fields = ['backbone_arch', 'input_config', 'backbone_config', 'max_epochs']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in YAML: {field}")

    # Validate backbone_config subfields
    backbone_config_fields = ['num_trainable_blocks', 'return_token', 'norm_layer']
    for field in backbone_config_fields:
        if field not in config['backbone_config']:
            raise ValueError(f"Missing required backbone_config field in YAML: {field}")

    # Validate input_config subfields
    backbone_config_fields = ['img_size', 'mean_std']
    for field in backbone_config_fields:
        if field not in config['input_config']:
            raise ValueError(f"Missing required input_config field in YAML: {field}")

    return config

