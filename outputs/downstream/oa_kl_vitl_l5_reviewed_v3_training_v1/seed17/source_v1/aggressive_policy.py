from __future__ import annotations

import math
import numbers

try:  # keep the module importable on hosts without torch installed
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

__all__ = [
    "NUM_CLASSES",
    "NUM_ORDINAL_BINS",
    "DEFAULT_FOCAL_GAMMA",
    "corn_log_probabilities",
    "corn_terms",
    "augment_aggressive",
]

NUM_CLASSES = 5
# : Number of conditional binary tasks (all classes but the last).
NUM_ORDINAL_BINS = NUM_CLASSES - 1
# : Default focal exponent used by :func:`corn_terms`.
DEFAULT_FOCAL_GAMMA = 2.5

PHOTOMETRIC_RANGE = (0.85, 1.15)
MAX_ROTATION_DEGREES = 7.0
MAX_TRANSLATION_FRACTION = 0.05


def _require_torch():
    if torch is None:  # pragma: no cover
        raise RuntimeError("this function requires PyTorch to be installed")




def corn_log_probabilities(logits):
    _require_torch()
    _check_logits(logits)

    logits_f = logits.float()
    log_q = F.logsigmoid(logits_f)  # log sigmoid(z_j)      = log q_j
    log_one_minus_q = F.logsigmoid(-logits_f)  # log sigmoid(-z_j)     = log(1 - q_j)
    log_survival = torch.cumsum(log_q, dim=1)  # logS_j, j = 0..3

    return torch.cat(
        (
            log_one_minus_q[:, 0:1],  # log p_0
            log_survival[:, 0:3] + log_one_minus_q[:, 1:4],  # log p_1 .. log p_3
            log_survival[:, 3:4],  # log p_4
        ),
        dim=1,
    )




def corn_terms(logits, y, weights=None, gamma=DEFAULT_FOCAL_GAMMA):
    _require_torch()
    _check_logits(logits)
    _check_targets(logits, y)
    gamma_value = _validated_gamma(gamma)
    weight_values = _validated_weights(weights)

    logits_f = logits.float()
    y_long = y.to(device=logits_f.device).long()
    log_probabilities = corn_log_probabilities(logits_f)

    batch = logits_f.shape[0]
    if batch == 0:
        empty = logits_f.reshape(-1)[:0]
        return {
            "log_probabilities": log_probabilities,
            "focal": empty.clone(),
            "corn_sum": empty.clone(),
            "eligible_count": torch.zeros(
                0, dtype=torch.long, device=logits_f.device
            ),
        }

    if weight_values is None:
        sample_weight = torch.ones(
            batch, dtype=torch.float32, device=logits_f.device
        )
    else:
        weight_tensor = torch.as_tensor(
            weight_values, dtype=torch.float32, device=logits_f.device
        )
        sample_weight = weight_tensor[y_long]

    log_pt = log_probabilities.gather(1, y_long.unsqueeze(1)).squeeze(1)
    if gamma_value == 0.0:
        # Exactly the weighted NLL; skipping pow() also avoids 0 * inf in the
        # backward pass when p_y == 1.
        focal = -sample_weight * log_pt
    else:
        pt = log_pt.exp().clamp(max=1.0)
        focal = -sample_weight * (1.0 - pt).pow(gamma_value) * log_pt

    bins = torch.arange(NUM_ORDINAL_BINS, device=logits_f.device)
    # Task j is only trained for samples with y > j - 1, i.e. j <= y.
    eligible = y_long.unsqueeze(1) >= bins.unsqueeze(0)
    bin_targets = (y_long.unsqueeze(1) > bins.unsqueeze(0)).to(logits_f.dtype)
    bce = F.binary_cross_entropy_with_logits(logits_f, bin_targets, reduction="none")
    zeros = torch.zeros_like(bce)
    bce = torch.where(eligible, bce, zeros)
    corn_sum = sample_weight * bce.sum(dim=1)
    eligible_count = eligible.sum(dim=1).long()

    return {
        "log_probabilities": log_probabilities,
        "focal": focal.float(),
        "corn_sum": corn_sum.float(),
        "eligible_count": eligible_count,
    }


def _check_logits(logits):
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if not logits.is_floating_point():
        raise TypeError("logits must be a floating point tensor")
    if logits.dim() != 2:
        raise ValueError("logits must be 2D with shape (N, 4)")
    if logits.shape[1] != NUM_ORDINAL_BINS:
        raise ValueError(
            f"logits must have exactly {NUM_ORDINAL_BINS} ordinal bins"
        )


def _check_targets(logits, y):
    if not isinstance(y, torch.Tensor):
        raise TypeError("y must be a torch.Tensor")
    if y.is_floating_point() or y.is_complex():
        raise TypeError("y must be an integer tensor")
    if y.dtype == torch.bool:
        raise TypeError("y must be an integer tensor, not boolean")
    if y.dim() != 1:
        raise ValueError("y must be 1D with shape (N,)")
    if y.shape[0] != logits.shape[0]:
        raise ValueError("y length must match the number of logits rows")
    if y.numel() > 0:
        if bool((y < 0).any()) or bool((y >= NUM_CLASSES).any()):
            raise ValueError(f"y must be class indices in [0, {NUM_CLASSES - 1}]")


def _validated_gamma(gamma):
    if isinstance(gamma, bool) or not isinstance(gamma, numbers.Real):
        raise TypeError("gamma must be a real number")
    value = float(gamma)
    if not math.isfinite(value):
        raise ValueError("gamma must be finite")
    if value < 0.0:
        raise ValueError("gamma must be >= 0")
    return value


def _validated_weights(weights):
    if weights is None:
        return None
    if isinstance(weights, torch.Tensor):
        if weights.dim() != 1 or weights.shape[0] != NUM_CLASSES:
            raise ValueError(f"weights must have exactly {NUM_CLASSES} entries")
        values = weights.detach().to("cpu", torch.float64).tolist()
    elif isinstance(weights, (str, bytes)):
        raise TypeError(f"weights must be a sequence of {NUM_CLASSES} numbers")
    else:
        try:
            values = list(weights)
        except TypeError:
            raise TypeError(f"weights must be a sequence of {NUM_CLASSES} numbers")

    if len(values) != NUM_CLASSES:
        raise ValueError(f"weights must have exactly {NUM_CLASSES} entries")

    clean = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise TypeError("weights must contain real numbers")
        fvalue = float(value)
        if not math.isfinite(fvalue):
            raise ValueError("weights must be finite")
        if fvalue < 0.0:
            raise ValueError("weights must be non-negative")
        clean.append(fvalue)
    return clean




def augment_aggressive(images, enabled=True):
    if not enabled:
        return images

    _require_torch()
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor")
    if not images.is_floating_point():
        raise TypeError("images must be a floating point tensor")
    if images.dim() != 4:
        raise ValueError("images must be 4D with shape (B, 1, H, W)")
    if images.shape[1] != 1:
        raise ValueError("images must have a single channel")

    if images.numel() == 0:
        # Empty shard: nothing to augment, no RNG consumed.
        return images.detach().clone()

    low = float(images.detach().min())
    high = float(images.detach().max())
    if low < 0.0 or high > 1.0:
        raise ValueError("images must have values in [0, 1]")

    device = images.device
    compute_dtype = torch.float64 if images.dtype == torch.float64 else torch.float32

    with torch.no_grad():
        work = images.detach().to(compute_dtype)
        batch = work.shape[0]

        # --- photometric (fixed draw order: brightness, contrast, gamma) ---
        factor_shape = (batch, 1, 1, 1)
        brightness = _uniform_range(factor_shape, device, compute_dtype)
        contrast = _uniform_range(factor_shape, device, compute_dtype)
        gamma = _uniform_range(factor_shape, device, compute_dtype)

        mean = work.mean(dim=(2, 3), keepdim=True)
        work = (work - mean) * contrast + mean
        work = work * brightness
        work = work.clamp(0.0, 1.0)
        work = work.pow(gamma)

        height = int(work.shape[2])
        width = int(work.shape[3])

        angle = _uniform((batch,), device, compute_dtype) * MAX_ROTATION_DEGREES
        angle = angle * (math.pi / 180.0)
        shift_x = _uniform((batch,), device, compute_dtype) * (
            MAX_TRANSLATION_FRACTION * width
        )
        shift_y = _uniform((batch,), device, compute_dtype) * (
            MAX_TRANSLATION_FRACTION * height
        )

        cos = torch.cos(angle)
        sin = torch.sin(angle)
        # ``affine_grid`` maps output normalised coordinates back to input ones.
        # Using the transposed rotation (R^T) makes the *content* appear rotated
        # by ``+angle``; negating the translation makes the content move by
        # ``(+shift_x, +shift_y)`` pixels.
        theta = torch.stack(
            (
                cos,
                sin * float(height) / float(width),
                -2.0 * (cos * shift_x + sin * shift_y) / float(width),
                -sin * float(width) / float(height),
                cos,
                2.0 * (sin * shift_x - cos * shift_y) / float(height),
            ),
            dim=1,
        ).reshape(batch, 2, 3)

        grid = F.affine_grid(
            theta, [int(size) for size in work.shape], align_corners=False
        )
        work = F.grid_sample(
            work,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        work = work.clamp(0.0, 1.0)

    return work.to(images.dtype)


def _uniform(shape, device, dtype):
    return (2.0 * torch.rand(shape, dtype=torch.float32, device=device) - 1.0).to(dtype)


def _uniform_range(shape, device, dtype):
    low, high = PHOTOMETRIC_RANGE
    span = high - low
    return (
        low + span * torch.rand(shape, dtype=torch.float32, device=device)
    ).to(dtype)
