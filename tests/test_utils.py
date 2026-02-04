import os, sys
from typing import List, Callable
import glob

from PIL import Image
from torch.utils.data import Dataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

supported_configs = {'SMALL', 'BASE', 'LARGE', 'GIANT'}


def load_da3_as_is(config_name: str='BASE'):
    import vpr.models.backbones.da3.da3 as da3
    from vpr.models.backbones.da3.depth_anything_3.api import DepthAnything3
    config_name = config_name.upper()
    if not config_name in supported_configs:
        raise ValueError(f"Configuration {config_name} is not supported. Try one of the followings: {supported_configs}")
    da3 = DepthAnything3.from_pretrained(f"depth-anything/DA3-{config_name}")
    freeze_model(da3)

    return da3


def load_da3_dino(config_name: str='BASE', return_token: bool=True):
    import vpr.models.backbones.da3.da3 as da3
    if not config_name in supported_configs:
        raise ValueError(f"Configuration {config_name} is not supported. Try one of the followings: {supported_configs}")
    da3_dino = da3.DepthAnything3Dino.from_pretrained(
        model_name=f'da3-{config_name.lower()}',
        return_token=return_token,
    )
    freeze_model(da3_dino)

    return da3_dino


def freeze_model(model) -> None:
    for param in model.parameters():
        param.requires_grad = False
    model.eval()


class ImgDirDataset(Dataset):
    def __init__(self, img_dir: str, transform: Callable|None = None):        
        self.img_pahts = ImgDirDataset.scan_dir(img_dir)
        assert len(self.img_pahts) > 0, "Found no valid image"
        self.transform = transform

    @staticmethod
    def scan_dir(img_dir: str) -> List[str]:
        valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        all_img_paths = []
        for ext in valid_exts:
            all_img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        
        return all_img_paths

    def __len__(self) -> int:
        return len(self.img_pahts)

    def __getitem__(self, index) -> Image.Image:
        img_path = self.img_pahts[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_path