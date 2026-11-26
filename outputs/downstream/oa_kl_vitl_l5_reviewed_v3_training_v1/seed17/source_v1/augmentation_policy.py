import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F


DEFAULT_AUGMENTATION_PARAMETERS = {
    "brightness": [0.9, 1.1],
    "contrast": [0.9, 1.1],
    "gamma": [0.9, 1.1],
    "horizontal_flip_probability": 0.5,
    "rotation_degrees": [-5.0, 5.0],
    "translation_fraction": [-0.03, 0.03],
    "gaussian_blur_probability": 0.15,
    "gaussian_blur_sigma": [0.1, 0.7],
    "gaussian_blur_kernel": 3,
    "padding": "border",
    "interpolation": "bilinear",
    "align_corners": False,
}


def _parameters(parameters):
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping")
    unknown = set(parameters) - set(DEFAULT_AUGMENTATION_PARAMETERS)
    if unknown:
        raise ValueError(f"unknown augmentation parameter: {sorted(unknown)}")
    p = {**DEFAULT_AUGMENTATION_PARAMETERS, **parameters}
    for key in ("brightness", "contrast", "gamma", "rotation_degrees",
                "translation_fraction", "gaussian_blur_sigma"):
        value = p[key]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{key} must be a range of length 2")
        lo, hi = map(float, value)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ValueError(f"{key} invalid range")
        if key in ("brightness", "contrast", "gamma", "gaussian_blur_sigma") and lo <= 0:
            raise ValueError(f"{key} must be strictly greater than 0")
        p[key] = (lo, hi)
    for key in ("horizontal_flip_probability", "gaussian_blur_probability"):
        value = float(p[key])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be in [0,1]")
        p[key] = value
    kernel = p["gaussian_blur_kernel"]
    if isinstance(kernel, bool) or not isinstance(kernel, int) or kernel < 1 or kernel % 2 != 1:
        raise ValueError("gaussian_blur_kernel must be a positive odd number")
    if p["padding"] != "border" or p["interpolation"] != "bilinear" or p["align_corners"] is not False:
        raise ValueError("border padding, bilinear interpolation and align_corners=False are required")
    return p


def _uniform(bounds, batch, device):
    lo, hi = bounds
    if lo == hi:
        return torch.full((batch,), lo, device=device, dtype=torch.float32)
    return torch.rand(batch, device=device, dtype=torch.float32) * (hi - lo) + lo


def _mask(probability, batch, device):
    if probability == 1:
        return torch.ones((batch, 1, 1, 1), device=device, dtype=torch.bool)
    return (torch.rand((batch, 1, 1, 1), device=device) < probability)


@torch.no_grad()
def augment_knee(images, parameters, enabled=True):
    if not enabled:
        return images
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 1:
        raise ValueError("images must be a [B,1,H,W] tensor")
    if not images.is_floating_point():
        raise TypeError("images must be a float tensor")
    batch, _, height, width = images.shape
    if height < 1 or width < 1:
        raise ValueError("image height and width must be positive")
    p = _parameters(parameters)
    if batch == 0:
        return images.clone()
    device, original_dtype = images.device, images.dtype
    # Affine and convolution run in float32 even if the caller is under BF16/FP16 autocast.
    with torch.autocast(device_type=device.type, enabled=False):
        x = images.to(dtype=torch.float32).clone()
        if p["brightness"] != (1.0, 1.0):
            x = (x * _uniform(p["brightness"], batch, device)[:, None, None, None]).clamp(0, 1)
        if p["contrast"] != (1.0, 1.0):
            mean = x.mean(dim=(-2, -1), keepdim=True)
            factor = _uniform(p["contrast"], batch, device)[:, None, None, None]
            x = ((x - mean) * factor + mean).clamp(0, 1)
        if p["gamma"] != (1.0, 1.0):
            x = x.clamp(0, 1).pow(_uniform(p["gamma"], batch, device)[:, None, None, None])

        flip_probability = p["horizontal_flip_probability"]
        if flip_probability:
            x = torch.where(_mask(flip_probability, batch, device), x.flip(-1), x)

        if p["rotation_degrees"] != (0.0, 0.0) or p["translation_fraction"] != (0.0, 0.0):
            angle = _uniform(p["rotation_degrees"], batch, device) * (math.pi / 180.0)
            cosine, sine = angle.cos(), angle.sin()
            tx = 2 * _uniform(p["translation_fraction"], batch, device)
            ty = 2 * _uniform(p["translation_fraction"], batch, device)
            height_over_width = height / width
            width_over_height = width / height
            theta = torch.empty((batch, 2, 3), device=device, dtype=torch.float32)
            theta[:, 0, 0] = cosine
            theta[:, 0, 1] = sine * height_over_width
            theta[:, 1, 0] = -sine * width_over_height
            theta[:, 1, 1] = cosine
            theta[:, 0, 2] = -cosine * tx - sine * height_over_width * ty
            theta[:, 1, 2] = sine * width_over_height * tx - cosine * ty
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

        blur_probability = p["gaussian_blur_probability"]
        if blur_probability:
            blur_mask = _mask(blur_probability, batch, device)
            sigma = _uniform(p["gaussian_blur_sigma"], batch, device)
            kernel_size = p["gaussian_blur_kernel"]
            radius = kernel_size // 2
            coordinate = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
            weights = torch.exp(-0.5 * (coordinate[None, :] / sigma[:, None]).square())
            weights = weights / weights.sum(dim=1, keepdim=True)
            kernels = (weights[:, :, None] * weights[:, None, :])[:, None, :, :]
            # Treat the batch as channels so a grouped convolution applies a separate sigma per image; edges are replicated.
            padded = F.pad(x.reshape(1, batch, height, width), (radius,) * 4, mode="replicate")
            blurred = F.conv2d(padded, kernels, groups=batch).reshape(batch, 1, height, width)
            x = torch.where(blur_mask, blurred, x)
        return x.clamp(0, 1).to(dtype=original_dtype)
