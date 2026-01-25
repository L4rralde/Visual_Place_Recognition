
python train_from_yaml.py --config "$file"

## TODO:
- [X] Da3 dino class does not match da3 auxiliar outputs. They do as long we do not preprocess the inputs.
- [x] Check if tokens' order remain constant. They do. A commit from december 2025 modified the order, but we will use an older version.
- [x] Latest version of da3 repo breaks some configurations. I think only nested works. Kept one before december.
- [x] Write tests to check if aux features are constant independent of the images used.
- [x] Write tests to check if aux features from da3 api match those from my class.
- [x] File bug regarding da3, cpu memory and different image aspect ratios
- [ ] Write tests to check if the predictions for 3D reconstruction match (da3_salad vs da3).
- [ ] Clean out. Remove args that are not required, e.g., return_token. This must be always true.
