
python train_from_yaml.py --config "$file"

## TODO:
- [X] Da3 dino class does not match da3 auxiliar outputs. They do as long we do not preprocess the inputs.
- [x] Check if tokens' order remain constant. They do. A commit from december 2025 modified the order, but we will use an older version.
- [x] Latest version of da3 repo breaks some configurations. I think only nested works. Kept one before december.
- [ ] Write tests to check if aux features are constant independent of the images used.
- [ ] Writes tests to check if aux features from da3 api match those from my class.
- [ ] Write preprocessing functions to replace that from da3 and fix major bug for training da3dino + salad.
