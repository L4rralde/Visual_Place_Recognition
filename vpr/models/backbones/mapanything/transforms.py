def load_images(
    folder_or_list,
    resize_mode="fixed_mapping",
    size=None,
    norm_type="dinov2",
    patch_size=14,
    verbose=False,
    bayer_format=False,
    resolution_set=518,
    stride=1,
):
    """
    Open and convert all images in a list or folder to proper input format for model

    Args:
        folder_or_list (str or list): Path to folder or list of image paths.
        resize_mode (str): Resize mode - "fixed_mapping", "longest_side", "square", or "fixed_size". Defaults to "fixed_mapping".
        size (int or tuple, optional): Required for "longest_side", "square", and "fixed_size" modes.
                                      - For "longest_side" and "square": int value for resize dimension
                                      - For "fixed_size": tuple of (width, height)
        norm_type (str, optional): Image normalization type. See UniCeption IMAGE_NORMALIZATION_DICT keys. Defaults to "dinov2".
        patch_size (int, optional): Patch size for image processing. Defaults to 14.
        verbose (bool, optional): If True, print progress messages. Defaults to False.
        bayer_format (bool, optional): If True, read images in Bayer format. Defaults to False.
        resolution_set (int, optional): Resolution set to use for "fixed_mapping" mode (518 or 512). Defaults to 518.
        stride (int, optional): Load every nth image from the input. stride=1 loads all images, stride=2 loads every 2nd image, etc. Defaults to 1.

    Returns:
        list: List of dictionaries containing image data and metadata
    """
    # Validate resize_mode and size parameter requirements
    valid_resize_modes = ["fixed_mapping", "longest_side", "square", "fixed_size"]
    if resize_mode not in valid_resize_modes:
        raise ValueError(
            f"Resize_mode must be one of {valid_resize_modes}, got '{resize_mode}'"
        )

    if resize_mode in ["longest_side", "square", "fixed_size"] and size is None:
        raise ValueError(f"Size parameter is required for resize_mode='{resize_mode}'")

    # Validate size type based on resize mode
    if resize_mode in ["longest_side", "square"]:
        if not isinstance(size, int):
            raise ValueError(
                f"Size must be an int for resize_mode='{resize_mode}', got {type(size)}"
            )
    elif resize_mode == "fixed_size":
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError(
                f"Size must be a tuple/list of (width, height) for resize_mode='fixed_size', got {size}"
            )
        if not all(isinstance(x, int) for x in size):
            raise ValueError(
                f"Size values must be integers for resize_mode='fixed_size', got {size}"
            )

    # Get list of image paths
    if isinstance(folder_or_list, str):
        # If folder_or_list is a string, assume it's a path to a folder
        if verbose:
            print(f"Loading images from {folder_or_list}")
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))
    elif isinstance(folder_or_list, list):
        # If folder_or_list is a list, assume it's a list of image paths
        if verbose:
            print(f"Loading a list of {len(folder_or_list)} images")
        root, folder_content = "", folder_or_list
    else:
        # If folder_or_list is neither a string nor a list, raise an error
        raise ValueError(f"Bad {folder_or_list=} ({type(folder_or_list)})")

    # Define supported image extensions
    supported_images_extensions = [".jpg", ".jpeg", ".png"]
    if heif_support_enabled:
        supported_images_extensions += [".heic", ".heif"]
    supported_images_extensions = tuple(supported_images_extensions)

    # First pass: Load all images and collect aspect ratios
    loaded_images = []
    aspect_ratios = []
    for i, path in enumerate(folder_content):
        # Skip images based on stride
        if i % stride != 0:
            continue

        # Check if the file has a supported image extension
        if not path.lower().endswith(supported_images_extensions):
            continue

        try:
            # Otherwise, read the image normally
            img = exif_transpose(PIL.Image.open(os.path.join(root, path))).convert(
                "RGB"
            )

            W1, H1 = img.size
            aspect_ratios.append(W1 / H1)
            loaded_images.append((path, img, W1, H1))

        except Exception as e:
            if verbose:
                print(f"Warning: Could not load {path}: {e}")
            continue

    # Check if any images were loaded
    if not loaded_images:
        raise ValueError("No valid images found")

    # Calculate average aspect ratio and determine target size
    average_aspect_ratio = sum(aspect_ratios) / len(aspect_ratios)
    if verbose:
        print(
            f"Calculated average aspect ratio: {average_aspect_ratio:.3f} from {len(aspect_ratios)} images"
        )

    # Determine target size for all images based on resize mode
    if resize_mode == "fixed_mapping":
        # Resolution mappings are already compatible with their respective patch sizes
        # 518 mappings are divisible by 14, 512 mappings are divisible by 16
        target_width, target_height = find_closest_aspect_ratio(
            average_aspect_ratio, resolution_set
        )
        target_size = (target_width, target_height)
    elif resize_mode == "square":
        target_size = (
            round((size // patch_size)) * patch_size,
            round((size // patch_size)) * patch_size,
        )
    elif resize_mode == "longest_side":
        # Use average aspect ratio to determine size for all images
        # Longest side should be the input size
        if average_aspect_ratio >= 1:  # Landscape or square
            # Width is the longest side
            target_size = (
                size,
                round((size // patch_size) / average_aspect_ratio) * patch_size,
            )
        else:  # Portrait
            # Height is the longest side
            target_size = (
                round((size // patch_size) * average_aspect_ratio) * patch_size,
                size,
            )
    elif resize_mode == "fixed_size":
        # Use exact size provided, aligned to patch_size
        target_size = (
            (size[0] // patch_size) * patch_size,
            (size[1] // patch_size) * patch_size,
        )

    if verbose:
        print(
            f"Using target resolution {target_size[0]}x{target_size[1]} (W x H) for all images"
        )

    # Get the image normalization function based on the norm_type
    if norm_type in IMAGE_NORMALIZATION_DICT.keys():
        img_norm = IMAGE_NORMALIZATION_DICT[norm_type]
        ImgNorm = tvf.Compose(
            [tvf.ToTensor(), tvf.Normalize(mean=img_norm.mean, std=img_norm.std)]
        )
    else:
        raise ValueError(
            f"Unknown image normalization type: {norm_type}. Available options: {list(IMAGE_NORMALIZATION_DICT.keys())}"
        )

    # Second pass: Resize all images to the same target size
    imgs = []
    for path, img, W1, H1 in loaded_images:
        # Resize and crop the image to the target size
        img = crop_resize_if_necessary(img, resolution=target_size)[0]

        # Normalize image and add it to the list
        W2, H2 = img.size
        if verbose:
            print(f" - Adding {path} with resolution {W1}x{H1} --> {W2}x{H2}")

        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=len(imgs),
                instance=str(len(imgs)),
                data_norm_type=[norm_type],
            )
        )

    assert imgs, "No images foud at " + root
    if verbose:
        print(f" (Found {len(imgs)} images)")

    return imgs