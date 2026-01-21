
python train_from_yaml.py --config "$file"

## TODO:
- [X] Da3 dino class does not match da3 auxiliar outputs. They do as long we do not preprocess the inputs.
- [x] Check if tokens' order remain constant. They do. A commit from december 2025 modified the order, but we will use an older version.
- [x] Latest version of da3 repo breaks some configurations. I think only nested works. Kept one before december.
