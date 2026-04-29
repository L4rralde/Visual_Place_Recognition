from typing import List

import torch
import numpy as np
import cv2
from PIL import Image
from lightglue import SuperPoint


# Helper to convert PIL to torch.Tensor usable by LightGlue / SuperPoint
def pil_to_torch_gray(
    pil_img: Image.Image,
    resize_max: int = 2048
) -> torch.Tensor:
    """
    Convert PIL Image to grayscale torch.Tensor (1,H,W), normalized [0,1], optionally resize so max edge <= resize_max.
    """
    img = pil_img.convert('L')  # Convert to grayscale
    img_np = np.array(img).astype(np.float32) / 255.0  # H x W

    # Resize if needed
    if resize_max is not None:
        h, w = img_np.shape[:2]
        scale = resize_max / max(h, w)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            img_np = cv2.resize(
                img_np, new_size,
                interpolation=cv2.INTER_LINEAR
            )

    # Convert to torch tensor (1, 1, H, W)
    img_t = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    return img_t


class SuperPoint:
    def __init__(self, max_num_keypoints: int=2048):
        self.superpoint = None
        self.max_num_keypoints = max_num_keypoints
        self.wakeup()
        self.device = (
            'cuda' if torch.cuda.is_available()
            else 'cpu'
        )

    def wakeup(self) -> None:
        self.superpoint = SuperPoint(max_num_keypoints=self.max_num_keypoints)
        self.superpoint = self.superpoint.eval().to(self.device)

    def run(self, image_list: List[Image.Image]) -> List[dict]:
        features_list = []
        for img in image_list:
            img_tensor = pil_to_torch_gray(img)

            with torch.no_grad():
                img_tensor = img_tensor.to(self.device)
                feats = self.superpoint.extract(img_tensor)

            feats = {k: v.cpu() for k, v in feats.items()}

            features_list.append(feats)

        return features_list
