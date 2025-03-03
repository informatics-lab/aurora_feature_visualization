from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
from torchvision.transforms import Normalize
import numpy as np
import kornia
from kornia.geometry.transform import translate

# try:
#     from kornia import warp_affine, get_rotation_matrix2d
# except ImportError:
from kornia.geometry.transform import (
    warp_affine,
    get_rotation_matrix2d,
    warp_affine3d,
)


KORNIA_VERSION = kornia.__version__


def jitter(d):
    assert d > 1, "Jitter parameter d must be more than 1, currently {}".format(d)

    def inner(image_t):
        dx = np.random.choice(d)
        dy = np.random.choice(d)
        return translate(image_t, torch.tensor([[dx, dy]]).float().to(image_t.device))

    return inner


def jitter_3d(displacement_range, dsize):
    assert displacement_range > 1, (
        "Jitter parameter displacement_range must be more than 1, currently {}".format(
            displacement_range
        )
    )

    def inner(image_t):
        device = image_t.device
        batch_size = image_t.shape[0]

        dx = (torch.rand(batch_size, 1, device=device) * 2 - 1) * displacement_range
        dy = (torch.rand(batch_size, 1, device=device) * 2 - 1) * displacement_range
        dz = (torch.rand(batch_size, 1, device=device) * 2 - 1) * displacement_range

        M = torch.eye(3, 4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        M[:, 0, 3] = dx.squeeze(1)
        M[:, 1, 3] = dy.squeeze(1)
        M[:, 2, 3] = dz.squeeze(1)

        jittered_image = warp_affine3d(image_t, M, dsize)
        return jittered_image, dx, dy, dz

    return inner


def pad(w, mode="reflect", constant_value=0.5):
    if mode != "constant":
        constant_value = 0

    def inner(image_t):
        return F.pad(
            image_t,
            [w] * 4,
            mode=mode,
            value=constant_value,
        )

    return inner


def random_scale(scales):
    def inner(image_t):
        scale = np.random.choice(scales)
        shp = image_t.shape[2:]
        scale_shape = [int(_roundup(scale * d)) for d in shp]
        pad_x = max(0, _roundup((shp[1] - scale_shape[1]) / 2))
        pad_y = max(0, _roundup((shp[0] - scale_shape[0]) / 2))
        upsample = torch.nn.Upsample(
            size=scale_shape, mode="bilinear", align_corners=True
        )
        return F.pad(upsample(image_t), [pad_y, pad_x] * 2)

    return inner


def random_rotate(angles, units="degrees"):
    def inner(image_t):
        b, _, h, w = image_t.shape
        # kornia takes degrees
        alpha = _rads2angle(np.random.choice(angles), units)
        angle = torch.ones(b) * alpha
        if KORNIA_VERSION < "0.4.0":
            scale = torch.ones(b)
        else:
            scale = torch.ones(b, 2)
        center = torch.ones(b, 2)
        center[..., 0] = (image_t.shape[3] - 1) / 2
        center[..., 1] = (image_t.shape[2] - 1) / 2
        M = get_rotation_matrix2d(center, angle, scale).to(image_t.device)
        rotated_image = warp_affine(image_t.float(), M, dsize=(h, w))
        return rotated_image

    return inner


def compose(transforms):
    def inner(x):
        for transform in transforms:
            x = transform(x)
        return x

    return inner


def _roundup(value):
    return np.ceil(value).astype(int)


def _rads2angle(angle, units):
    if units.lower() == "degrees":
        return angle
    if units.lower() in ["radians", "rads", "rad"]:
        angle = angle * 180.0 / np.pi
    return angle


def normalize():
    # ImageNet normalization for torchvision models
    # see https://pytorch.org/docs/stable/torchvision/models.html
    normal = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def inner(image_t):
        return torch.stack([normal(t) for t in image_t])

    return inner


def preprocess_inceptionv1():
    # Original Tensorflow's InceptionV1 model
    # takes in [-117, 138]
    # See https://github.com/tensorflow/lucid/blob/master/lucid/modelzoo/other_models/InceptionV1.py#L56
    # Thanks to ProGamerGov for this!
    return lambda x: x * 255 - 117


def color_jitter(mean=1.0, std=1.0):
    """
    Randomly shifts and scales each image’s channels.

    For an input tensor of shape (B, C, H, W) this function generates a random offset
    for each image and each channel in the range [-mean, mean] and a random scale factor computed
    as the exponential of a value in the range [-std, std]. Then it applies:
       output = (input - offset) / scale

    Args:
        mean (float): Maximum magnitude of shift. Default is 1.0.
        std (float): Maximum magnitude (pre-exponentiation) of scaling factor. Default is 1.0.
                      Note that the actual scale is computed as exp(random_value).
    Returns:
        A function that takes an image tensor and returns the color-jittered version.
    """

    def inner(image_t):
        b, c, h, w = image_t.shape
        offset = (torch.rand(b, c, 1, 1, device=image_t.device) - 0.5) * 2 * mean
        scale = ((torch.rand(b, c, 1, 1, device=image_t.device) - 0.5) * 2 * std).exp()
        return (image_t - offset) / scale

    return inner


def color_jitter_r(mean=1.0, std=1.0):
    """
    Reverse the color jitter transformation.

    Given an input tensor of shape (B, C, H, W), this transform computes random
    per-image offsets (in the range [-mean, mean]) and per-image scaling factors (via exp(),
    from a value in [-std, std]), and applies the reverse transformation:

         output = (input * scale) + offset

    This is designed to invert a corresponding forward jitter transform of the form:
         output = (input - offset) / scale

    Args:
        mean (float): Maximum magnitude for the random offset. Default is 1.0.
        std (float): Maximum magnitude (before exponentiation) for the random scale factor. Default is 1.0.
    Returns:
        A function that takes an image tensor and returns the reversed jitter version.
    """

    def inner(image_t):
        b, c, h, w = image_t.shape
        # Generate random offsets for each image and channel in [-mean, mean]
        offset = (torch.rand(b, c, 1, 1, device=image_t.device) - 0.5) * 2 * mean
        # Generate random values for scale in [-std, std] and exponentiate so that scale is in (exp(-std), exp(std))
        scale = ((torch.rand(b, c, 1, 1, device=image_t.device) - 0.5) * 2 * std).exp()
        return (image_t * scale) + offset

    return inner


def random_scale_vit(scales, target_size=(224, 224)):
    def inner(image_t):
        scale = np.random.choice(scales)
        scale_shape = [int(scale * d) for d in target_size]
        upsample = torch.nn.Upsample(
            size=scale_shape, mode="bilinear", align_corners=True
        )
        scaled_image = upsample(image_t)
        # Resize back to the target size required by ViT
        resize_back = torch.nn.Upsample(
            size=target_size, mode="bilinear", align_corners=True
        )
        return resize_back(scaled_image)

    return inner


def focus(size: int, std: float):
    """
    Randomly crops a square region of the given size from the center of the image,
    with a random perturbation controlled by std.

    Args:
        size (int): The size of the square to crop.
        std (float): Standard deviation for the random offset.

    Returns:
        A function that takes an image tensor of shape (B, C, H, W) and returns a cropped version.
    """

    def inner(img: torch.Tensor) -> torch.Tensor:
        # Generate random perturbations in the range [-std, std] for x and y directions.
        pert = (torch.rand(2, device=img.device) * 2 - 1) * std
        w, h = img.shape[-2:]
        # Compute starting indices for the crop ensuring they are within the image boundaries.
        x = (pert[0] + w // 2 - size // 2).long().clamp(min=0, max=w - size)
        y = (pert[1] + h // 2 - size // 2).long().clamp(min=0, max=h - size)
        return img[:, :, x : x + size, y : y + size]

    return inner


def center(size: int):
    def inner(img: torch.Tensor) -> torch.Tensor:
        return img

    return inner


def zoom(out_size: int = 224):
    """
    Returns a function that zooms (resizes) an image tensor to the given output size.

    Args:
        out_size (int): The desired height and width of the output image.

    Returns:
        A function that takes an image tensor of shape (B, C, H, W) and returns it
        zoomed (resized) to (B, C, out_size, out_size), using bilinear interpolation.
    """

    def inner(image_t: torch.Tensor) -> torch.Tensor:
        # Use the functional interface to interpolate to (out_size, out_size)
        return F.interpolate(
            image_t, size=(out_size, out_size), mode="bilinear", align_corners=False
        )

    return inner


def do_nothing():
    def inner(image_tf):
        return image_tf

    return inner


standard_transforms = [
    pad(12, mode="constant", constant_value=0.5),
    jitter(8),
    random_scale([1 + (i - 5) / 50.0 for i in range(11)]),
    random_rotate(list(range(-10, 11)) + 5 * [0]),
    jitter(4),
]


pii_transforms = [
    focus(0, 0),  # This is will be updated in the inversion loop
    jitter(8),
    color_jitter_r(1, True),
]
