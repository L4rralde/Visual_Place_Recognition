import os
from pathlib import Path
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms as T


class ImageDataset(Dataset):
    def __init__(self, input_dir: os.PathLike, transform: Optional[Callable] = None) -> None:
        self.input_dir = Path(input_dir)
        self.paths = sorted(
            p for p in self.input_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.transform = transform or T.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = self.transform(img)

        return img
