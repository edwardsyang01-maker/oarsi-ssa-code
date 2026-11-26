from __future__ import annotations

import math
import numbers
import operator

try:  # keep the module importable on hosts without torch installed
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None

__all__ = ["NUM_CLASSES", "class_weights", "loss_terms", "augment_gray"]

NUM_CLASSES = 5
NUM_ORDINAL_BINS = NUM_CLASSES - 1




def class_weights(counts, cap=3.0):
    values = _validated_counts(counts)
    cap_value = _validated_cap(cap)

    total = float(sum(values))
    peak = float(max(values))

    raw = [min(math.sqrt(peak / float(c)), cap_value) for c in values]
    weighted_mean = sum(float(c) * r for c, r in zip(values, raw)) / total
    return [r / weighted_mean for r in raw]


def _require_torch():
    if torch is None:  # pragma: no cover
        raise RuntimeError("this function requires PyTorch to be installed")


def _validated_counts(counts):
    if counts is None or isinstance(counts, (str, bytes)):
        raise TypeError("counts must be a sequence of five positive integers")
    try:
        values = list(counts)
    except TypeError:
        raise TypeError("counts must be a sequence of five positive integers")

    if len(values) != NUM_CLASSES:
        raise ValueError(f"counts must contain exactly {NUM_CLASSES} entries")

    clean = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("counts must contain integers, not booleans")
        try:
            ivalue = operator.index(value)
        except TypeError:
            raise TypeError("counts must contain integers")
        if ivalue <= 0:
            raise ValueError("counts must all be strictly positive")
        clean.append(int(ivalue))
    return clean


def _validated_cap(cap):
    if isinstance(cap, bool) or not isinstance(cap, numbers.Real):
        raise TypeError("cap must be a real number")
    cap_value = float(cap)
    if not math.isfinite(cap_value):
        raise ValueError("cap must be finite")
    if cap_value < 1.0:
        raise ValueError("cap must be >= 1")
    return cap_value




def loss_terms(logits, targets, weights=None, ordinal_lambda=0.0):
    _require_torch()
    _check_logits(logits)
    _check_targets(logits, targets)
    lam = _validated_ordinal_lambda(ordinal_lambda)
    weight_values = _validated_weights(weights)

    logits_f = logits.float()
    targets = targets.to(device=logits_f.device)

    if logits_f.shape[0] == 0:
        # Empty shard (possible on the tail of a distributed batch).  Return
        # empty per-example tensors that still participate in autograd.
        empty = logits_f.reshape(-1)[:0]
        return {
            "objective": empty,
            "cross_entropy": empty.clone(),
            "ordinal_rps": empty.clone(),
        }

    cross_entropy = F.cross_entropy(logits_f, targets.long(), reduction="none")

    probs = torch.softmax(logits_f, dim=1)
    cdf = torch.cumsum(probs[:, :NUM_ORDINAL_BINS], dim=1)
    bins = torch.arange(NUM_ORDINAL_BINS, device=logits_f.device, dtype=torch.long)
    target_cdf = (bins.unsqueeze(0) >= targets.long().unsqueeze(1)).to(probs.dtype)
    ordinal_rps = ((cdf - target_cdf) ** 2).mean(dim=1)

    base = cross_entropy + lam * ordinal_rps
    if weight_values is None:
        objective = base
    else:
        weight_tensor = torch.as_tensor(
            weight_values, dtype=torch.float32, device=logits_f.device
        )
        objective = weight_tensor[targets.long()] * base

    return {
        "objective": objective.float(),
        "cross_entropy": cross_entropy.float(),
        "ordinal_rps": ordinal_rps.float(),
    }


def _check_logits(logits):
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if not logits.is_floating_point():
        raise TypeError("logits must be a floating point tensor")
    if logits.dim() != 2:
        raise ValueError("logits must be 2D with shape (N, 5)")
    if logits.shape[1] != NUM_CLASSES:
        raise ValueError(f"logits must have exactly {NUM_CLASSES} classes")


def _check_targets(logits, targets):
    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a torch.Tensor")
    if targets.is_floating_point() or targets.is_complex():
        raise TypeError("targets must be an integer tensor")
    if targets.dtype == torch.bool:
        raise TypeError("targets must be an integer tensor, not boolean")
    if targets.dim() != 1:
        raise ValueError("targets must be 1D with shape (N,)")
    if targets.shape[0] != logits.shape[0]:
        raise ValueError("targets length must match the number of logits rows")
    if targets.numel() > 0:
        if bool((targets < 0).any()) or bool((targets >= NUM_CLASSES).any()):
            raise ValueError(f"targets must be class indices in [0, {NUM_CLASSES - 1}]")


def _validated_ordinal_lambda(ordinal_lambda):
    if isinstance(ordinal_lambda, bool) or not isinstance(ordinal_lambda, numbers.Real):
        raise TypeError("ordinal_lambda must be a real number")
    lam = float(ordinal_lambda)
    if not math.isfinite(lam):
        raise ValueError("ordinal_lambda must be finite")
    if lam < 0.0:
        raise ValueError("ordinal_lambda must be >= 0")
    return lam


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




def augment_gray(images, enabled):
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

    if images.numel() > 0:
        low = float(images.detach().min())
        high = float(images.detach().max())
        if low < 0.0 or high > 1.0:
            raise ValueError("images must have values in [0, 1]")

    device = images.device
    batch = images.shape[0]
    factor_shape = (batch, 1, 1, 1)

    brightness = _uniform_factor(factor_shape, device)
    contrast = _uniform_factor(factor_shape, device)
    gamma = _uniform_factor(factor_shape, device)

    work = images.to(torch.float64) if images.dtype == torch.float64 else images.float()
    mean = work.mean(dim=(2, 3), keepdim=True)
    out = (work - mean) * contrast.to(work.dtype) + mean
    out = out * brightness.to(work.dtype)
    out = out.clamp(0.0, 1.0)
    out = out.pow(gamma.to(work.dtype))

    return out.to(dtype=images.dtype)


def _uniform_factor(shape, device):
    return 0.9 + 0.2 * torch.rand(shape, dtype=torch.float32, device=device)
