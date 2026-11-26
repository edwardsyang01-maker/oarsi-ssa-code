from __future__ import annotations

import copy
from typing import Any, Mapping, Tuple

import numpy as np
from PIL import Image

PREPROCESSING_VERSION = "cache448_to_model_input_v1"
CACHE_IMAGE_SIZE = 448
MODEL_IMAGE_SIZES = (224, 448)
MIN_VALID_STD = 0.01
PIXEL_CENTER_MAPPING = "x_out=(x_in+0.5)*s-0.5"
PIXEL_CENTER_MAPPING_Y = "y_out=(y_in+0.5)*s-0.5"

__all__ = [
    "CACHE_IMAGE_SIZE",
    "MIN_VALID_STD",
    "MODEL_IMAGE_SIZES",
    "ModelInputPreprocessingError",
    "PIXEL_CENTER_MAPPING",
    "PREPROCESSING_VERSION",
    "cache_448_xy_to_model_xy",
    "model_xy_to_cache_448_xy",
    "prepare_model_input",
]


class ModelInputPreprocessingError(ValueError):
    pass


def _validate_image_size(image_size: Any) -> int:
    if isinstance(image_size, (bool, np.bool_)) or not isinstance(
        image_size, (int, np.integer)
    ):
        raise ModelInputPreprocessingError("image_size must be the integer 224 or 448")
    size = int(image_size)
    if size not in MODEL_IMAGE_SIZES:
        raise ModelInputPreprocessingError("image_size must be 224 or 448")
    return size


def _validate_cache_arrays(gray: Any, mask: Any) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(gray, np.ndarray):
        raise ModelInputPreprocessingError("gray must be a numpy array")
    if not isinstance(mask, np.ndarray):
        raise ModelInputPreprocessingError("valid_mask must be a numpy array")
    if gray.dtype != np.float32:
        raise ModelInputPreprocessingError("gray dtype must be float32")
    if mask.dtype != np.bool_:
        raise ModelInputPreprocessingError("valid_mask dtype must be bool")
    expected = (1, CACHE_IMAGE_SIZE, CACHE_IMAGE_SIZE)
    if gray.shape != expected or mask.shape != expected:
        raise ModelInputPreprocessingError("gray/valid_mask shape must be (1,448,448)")
    if not np.isfinite(gray).all():
        raise ModelInputPreprocessingError("gray has non-finite values")
    if float(gray.min()) < 0.0 or float(gray.max()) > 1.0:
        raise ModelInputPreprocessingError("gray must be in [0,1]")
    if not bool(mask.any()):
        raise ModelInputPreprocessingError("valid_mask needs at least one valid pixel")
    # Padding in the 448 cache must be exactly 0; reject, never repair.
    if bool(np.any(gray[~mask] != 0.0)):
        raise ModelInputPreprocessingError("padding of the 448 cache must be 0")
    std = _valid_std(gray, mask)
    if not np.isfinite(std) or std < MIN_VALID_STD:
        raise ModelInputPreprocessingError(
            f"gray std in the valid region must be >= {MIN_VALID_STD}"
        )
    return gray, mask


def _valid_std(gray: np.ndarray, mask: np.ndarray) -> float:
    values = gray[mask]
    if values.size == 0:
        return float("nan")
    return float(np.std(values.astype(np.float64), ddof=0))


def _validate_output(gray: np.ndarray, mask: np.ndarray, size: int) -> None:
    expected = (1, size, size)
    if gray.dtype != np.float32 or gray.shape != expected:
        raise ModelInputPreprocessingError(f"output image must be float32 {expected}")
    if mask.dtype != np.bool_ or mask.shape != expected:
        raise ModelInputPreprocessingError(f"output mask must be bool {expected}")
    if not np.isfinite(gray).all():
        raise ModelInputPreprocessingError("output image has non-finite values")
    if float(gray.min()) < 0.0 or float(gray.max()) > 1.0:
        raise ModelInputPreprocessingError("output image must be in [0,1]")
    if not bool(mask.any()):
        raise ModelInputPreprocessingError(
            f"{size} valid_mask is empty after downscaling"
        )
    if bool(np.any(gray[~mask] != 0.0)):
        raise ModelInputPreprocessingError(f"{size} output padding must be 0")
    std = _valid_std(gray, mask)
    if not np.isfinite(std) or std < MIN_VALID_STD:
        raise ModelInputPreprocessingError(
            f"{size} gray std in the valid region after downscaling must be >= {MIN_VALID_STD}"
        )


def _validate_cache_transform(cache_transform: Any) -> Tuple[dict, list]:
    if not isinstance(cache_transform, Mapping):
        raise ModelInputPreprocessingError("cache_transform must be a mapping")
    try:
        copied = copy.deepcopy(dict(cache_transform))
    except (TypeError, ValueError) as exc:  # pragma: no cover
        raise ModelInputPreprocessingError("cache_transform cannot be copied") from exc
    bbox = copied.get("source_bbox_xyxy_exclusive")
    if not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) != 4:
        raise ModelInputPreprocessingError(
            "cache_transform.source_bbox_xyxy_exclusive must be a 4-element sequence"
        )
    values = []
    for value in bbox:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ModelInputPreprocessingError(
                "cache_transform source bbox must be strict integers"
            )
        values.append(int(value))
    x0, y0, x1, y1 = values
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ModelInputPreprocessingError(
            "cache_transform source bbox must be a positive-area box in native coordinates"
        )
    return copied, values


def _resize_float_image(array: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(
        np.ascontiguousarray(array, dtype=np.float32), mode="F"
    )
    return np.asarray(
        image.resize((size, size), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def _resize_bool_mask(mask: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(
        (np.ascontiguousarray(mask, dtype=np.uint8) * np.uint8(255)), mode="L"
    )
    return (
        np.asarray(
            image.resize((size, size), resample=Image.Resampling.NEAREST),
            dtype=np.uint8,
        )
        > 0
    )


def _build_transform(
    cache_transform: dict, native_bbox: list, size: int
) -> dict:
    scale = float(size) / float(CACHE_IMAGE_SIZE)
    return {
        "coordinate_space": "model_input_pixels",
        "image_size": int(size),
        "model_image_size": int(size),
        "model_input_shape": [1, int(size), int(size)],
        "source_bbox_coordinate_space": "native_pixels",
        "source_bbox_xyxy_exclusive": [int(v) for v in native_bbox],
        "native_bbox_xyxy_exclusive": [int(v) for v in native_bbox],
        "cache_coordinate_space": "cache_448_pixels",
        "cache_transform": cache_transform,
        "composition": {
            "from_space": "cache_448_pixels",
            "to_space": "model_input_pixels",
            "source_size_wh": [CACHE_IMAGE_SIZE, CACHE_IMAGE_SIZE],
            "target_size_wh": [int(size), int(size)],
            "scale": scale,
            "scale_x": scale,
            "scale_y": scale,
            "pixel_center_mapping": PIXEL_CENTER_MAPPING,
            "pixel_center_mapping_y": PIXEL_CENTER_MAPPING_Y,
            "resampling_image": "PIL_F_BILINEAR",
            "resampling_mask": "PIL_NEAREST",
            "padding_outside_mask": 0.0,
            "identity_448": bool(size == CACHE_IMAGE_SIZE),
        },
    }


def prepare_model_input(
    gray: Any,
    mask: Any,
    *,
    image_size: Any,
    cache_transform: Any,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    size = _validate_image_size(image_size)
    source, source_mask = _validate_cache_arrays(gray, mask)
    cache_transform_copy, native_bbox = _validate_cache_transform(cache_transform)

    if size == CACHE_IMAGE_SIZE:
        # At 448 there is no resampling: cached pixels are kept element for element.
        image = np.array(source, dtype=np.float32, copy=True, order="C")
        out_mask = np.array(source_mask, dtype=np.bool_, copy=True, order="C")
    else:
        try:
            image_2d = _resize_float_image(source[0], size)
            out_mask_2d = _resize_bool_mask(source_mask[0], size)
        except (OSError, TypeError, ValueError) as exc:
            raise ModelInputPreprocessingError(
                f"{size} resize failed: {exc}"
            ) from exc
        image = np.array(image_2d, dtype=np.float32, copy=True).reshape(
            1, size, size
        )
        out_mask = np.array(out_mask_2d, dtype=np.bool_, copy=True).reshape(
            1, size, size
        )
        # After NEAREST upsampling of the mask, zero the input outside the mask again.
        image[~out_mask] = 0.0

    _validate_output(image, out_mask, size)
    transform = _build_transform(cache_transform_copy, native_bbox, size)
    return (
        np.ascontiguousarray(image, dtype=np.float32),
        np.ascontiguousarray(out_mask, dtype=np.bool_),
        transform,
    )


def _composition_scale(transform: Any) -> float:
    if not isinstance(transform, Mapping):
        raise ValueError("transform must be a mapping")
    composition = transform.get("composition")
    if not isinstance(composition, Mapping):
        raise ValueError("transform is missing composition")
    scale = composition.get("scale")
    if isinstance(scale, (bool, np.bool_)) or not isinstance(
        scale, (int, float, np.integer, np.floating)
    ):
        raise ValueError("composition.scale must be a number")
    value = float(scale)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("composition.scale must be a positive finite number")
    return value


def cache_448_xy_to_model_xy(x: Any, y: Any, transform: Any):
    scale = _composition_scale(transform)
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    return (
        (x_values + 0.5) * scale - 0.5,
        (y_values + 0.5) * scale - 0.5,
    )


def model_xy_to_cache_448_xy(x: Any, y: Any, transform: Any):
    scale = _composition_scale(transform)
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    return (
        (x_values + 0.5) / scale - 0.5,
        (y_values + 0.5) / scale - 0.5,
    )
