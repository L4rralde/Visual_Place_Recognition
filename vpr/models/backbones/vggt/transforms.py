from typing import Tuple, Callable

from torchvision import transforms as T
from PIL import Image
import torch

#TODO List:
# - [ ] Does vggt work with smaller target sizes?
# - [ ] Does mode='pad' works as good as 'crop'?
# - [ ] Check if this function actually works without converting imgs to torch tensor in the middle of the operations.


def preprocess_image(img: Image.Image, mode: str="crop", target_size: int=518) -> Image.Image:
    # If there's an alpha channel, blend onto white background:
    if img.mode == "RGBA":
        # Create white background
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        # Alpha composite onto the white background
        img = Image.alpha_composite(background, img)

    # Now convert to "RGB" (this step assigns white for transparent areas)
    img = img.convert("RGB")

    width, height = img.size

    if mode == "pad":
        # Make the largest dimension 518px while maintaining aspect ratio
        if width >= height:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
        else:
            new_height = target_size
            new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
    else:  # mode == "crop"
        # Original behavior: set width to 518px
        new_width = target_size
        # Calculate height maintaining aspect ratio, divisible by 14
        new_height = round(height * (new_width / width) / 14) * 14

    # Resize with new dimensions (width, height)
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    #img = to_tensor(img)  # Convert to tensor (0, 1)

    # Center crop height if it's larger than 518 (only in crop mode)
    if mode == "crop" and new_height > target_size:
        start_y = (new_height - target_size) // 2
        img = img[:, start_y : start_y + target_size, :]

    # For pad mode, pad to make a square of target_size x target_size
    if mode == "pad":
        h_padding = target_size - img.shape[1]
        w_padding = target_size - img.shape[2]

        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left

            # Pad with white (value=1.0)
            img = torch.nn.functional.pad(
                img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )

    return img

    #The following code can't be executed since this class modifies images individually.
    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images


class VggtTransform:
    def __init__(self, img_size: int=518, mode: str="crop") -> None:
        self.img_size = img_size
        self.mode = mode
        #assert self.img_size == 518, "I think it won't work with another size"

    def __call__(self, img: Image.Image, mode: str|None = None) -> Image.Image:
        if mode is None:
            mode = self.mode
        return preprocess_image(img, mode, self.img_size)


def get_transforms(input_config: dict) -> Tuple[Callable]:
    img_size = input_config.get('img_size', 518)
    mode = input_config.get('mode', 'crop')
    if isinstance(img_size, int):
        resize = VggtTransform(img_size, mode)
    elif isinstance(img_size, list) and len(img_size) == 2:
        #same in dino
        resize = T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR)
    else:
        raise ValueError(f"Unexpected format for img_size: {img_size}")

    train_transform = T.Compose([
        resize, #Bug: https://github.com/L4rralde/Visual_Place_Recognition/issues/1?reload=1
        T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])
    valid_transform = T.Compose([
        resize, #Bug: https://github.com/L4rralde/Visual_Place_Recognition/issues/1?reload=1
        T.ToTensor(),
    ])

    return train_transform, valid_transform
