from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Subset
except ImportError:  # pragma: no cover
    torch = None
    F = None
    DataLoader = None
    Subset = None

PACKAGE = "downstream.oa_kl_unified_v1"
OUTPUT_SUBTREE = "outputs/downstream/oa_kl_unified_v1"
CHECKPOINT_SUBTREE = "checkpoints/oa_kl_unified_v1"
RUN_PREFIX = "oa_kl_unified_v1_"
BACKBONES = ("dinov3_small", "skelex")
MODES = ("frozen_head", "last2")
IMAGE_SIZES = (224, 448)
TARGET = "single_knee_clinical_KL5"
INPUT_SHAPE = ["B", 1, None, None]
OUTPUT_SHAPE = ["B", 5]
SCHEMA_META = "oa_kl_unified_v1_training_core_v1"
DATA_STATE_LOADER_POLICY = "workers0_independent_generator_no_prefetch"
SYNTHETIC_MAX_SAMPLES = 1000
SYNTHETIC_MAX_STOP_AFTER = 20
SYNTHETIC_MAX_PARAMETERS = 1_000_000
REAL_MIN_PARAMETERS = 1_000_000
AMP_INIT_SCALE = 1024.0
DEFAULT_AMP_MAX_RETRIES = 5
OLD_SCHEME_MARKERS = ("wholefilm", "oa_kl_compare", "oa_kl_knee_v1", "oa_kl_knee_v2")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEVICE_PATTERN = re.compile(r"^(cpu|cuda:[0-9]+)$")


def _require_torch():
    if torch is None:
        raise RuntimeError("this operation needs the server torch environment")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def file_sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, writer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Any, value: Any) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    ).encode()
    _atomic_write(Path(path), lambda stream: stream.write(payload))


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(item) for item in value]
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    return value

@dataclass(frozen=True)
class TrainingConfig:
    root: Any
    run_name: str
    seed: int = 17
    epochs: int = 20
    image_size: int = 448
    backbone: str = "skelex"
    mode: str = "last2"
    micro_batch_size: int = 8
    effective_batch_size: int = 64
    workers: int = 0
    cpu_threads: int = 1
    head_lr: float = 1e-3
    encoder_lr: float = 1e-5
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    warmup_epochs: int = 1
    min_lr_ratio: float = 0.01
    checkpoint_every_steps: int = 400
    keep_last: int = 3
    amp_max_retries: int = DEFAULT_AMP_MAX_RETRIES
    device: str = "cuda:1"
    output_dir: Any = None
    checkpoint_dir: Any = None
    loss: str = "cross_entropy"
    augment: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": str(Path(os.fspath(self.root))),
            "run_name": self.run_name,
            "seed": self.seed,
            "epochs": self.epochs,
            "image_size": self.image_size,
            "backbone": self.backbone,
            "mode": self.mode,
            "micro_batch_size": self.micro_batch_size,
            "effective_batch_size": self.effective_batch_size,
            "workers": self.workers,
            "cpu_threads": self.cpu_threads,
            "head_lr": self.head_lr,
            "encoder_lr": self.encoder_lr,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "warmup_epochs": self.warmup_epochs,
            "min_lr_ratio": self.min_lr_ratio,
            "checkpoint_every_steps": self.checkpoint_every_steps,
            "keep_last": self.keep_last,
            "amp_max_retries": self.amp_max_retries,
            "device": self.device,
            "output_dir": (
                None if self.output_dir is None else str(Path(os.fspath(self.output_dir)))
            ),
            "checkpoint_dir": (
                None
                if self.checkpoint_dir is None
                else str(Path(os.fspath(self.checkpoint_dir)))
            ),
            "loss": self.loss,
            "augment": self.augment,
        }


def _boundary(root: Path, subtree: str) -> Path:
    boundary = root / Path(*PurePosixPath(subtree).parts)
    if boundary.exists() and boundary.resolve() != boundary:
        raise ValueError("output/checkpoint subtree boundary must not be a symlink: " + subtree)
    return boundary


def _path(root: Path, value: Any, subtree: str) -> Path:
    candidate = Path(os.fspath(value))
    raw_parts = list(candidate.parts)
    if candidate.is_absolute():
        base = candidate
    else:
        base = root / candidate
    if any(part == ".." for part in raw_parts):
        raise ValueError("output/checkpoint path must not contain '..'")
    if any(marker in part for part in raw_parts for marker in OLD_SCHEME_MARKERS):
        raise ValueError("path must not point into the old whole-film/comparison subtrees")
    try:
        resolved = base.resolve()
    except (OSError, ValueError) as error:  # pragma: no cover
        raise ValueError("output/checkpoint path cannot be resolved") from error
    boundary = _boundary(root, subtree)
    if resolved == boundary or not resolved.is_relative_to(boundary):
        raise ValueError(f"path must stay inside the designated project subtree {subtree}")
    return resolved


def _validate_run_name(value: Any) -> None:
    if not isinstance(value, str) or not value.startswith(RUN_PREFIX):
        raise ValueError(f"run_name must explicitly use the {RUN_PREFIX} prefix")
    if len(value) <= len(RUN_PREFIX):
        raise ValueError("run_name needs content after the prefix")
    if "/" in value or "\\" in value or ".." in value:
        raise ValueError("run_name must not contain path separators or '..'")
    if any(not (ch.isalnum() or ch in "_.-") for ch in value):
        raise ValueError("run_name allows only letters, digits and _ . -")


def validate_config(config: TrainingConfig) -> Tuple[Path, Path]:
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    try:
        root = Path(os.fspath(config.root)).resolve(strict=True)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError("config.root must be an existing, resolvable directory") from error
    if not root.is_dir():
        raise ValueError("config.root must be a directory")

    for key in ("seed", "epochs", "image_size", "micro_batch_size",
                "effective_batch_size", "workers", "cpu_threads",
                "warmup_epochs", "checkpoint_every_steps", "keep_last",
                "amp_max_retries"):
        value = getattr(config, key)
        if type(value) is not int:
            raise ValueError(f"{key} must be an integer")
    if config.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if config.epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if config.image_size not in IMAGE_SIZES:
        raise ValueError("image_size must be explicitly 224 or 448")
    if config.backbone not in BACKBONES:
        raise ValueError("backbone must be dinov3_small or skelex")
    if config.mode not in MODES:
        raise ValueError("mode must be frozen_head or last2")
    if config.micro_batch_size < 1 or config.effective_batch_size < 1:
        raise ValueError("micro/effective batch must be positive integers")
    if config.effective_batch_size % config.micro_batch_size:
        raise ValueError("only integer gradient accumulation is supported (effective % micro == 0)")
    if config.workers != 0 or config.cpu_threads != 1:
        raise ValueError("only workers=0 and cpu_threads=1 are supported")
    for key in ("head_lr", "encoder_lr", "weight_decay", "max_grad_norm"):
        value = getattr(config, key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{key} must be a number")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{key} must be a finite positive number")
    if not 0 <= config.warmup_epochs <= config.epochs:
        raise ValueError("warmup_epochs must be in [0, epochs]")
    if not isinstance(config.min_lr_ratio, (int, float)) or isinstance(
        config.min_lr_ratio, bool
    ):
        raise ValueError("min_lr_ratio must be a number")
    if not 0 <= float(config.min_lr_ratio) <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")
    if config.checkpoint_every_steps < 1:
        raise ValueError("checkpoint_every_steps must be a positive integer")
    if config.keep_last < 2:
        raise ValueError("keep_last must be at least 2 (required by CheckpointManager)")
    if config.amp_max_retries < 1:
        raise ValueError("amp_max_retries must be a positive integer")
    if not isinstance(config.device, str) or DEVICE_PATTERN.fullmatch(config.device) is None:
        raise ValueError("device must be 'cpu' or 'cuda:N'")
    if config.loss != "cross_entropy":
        raise ValueError("only unweighted cross_entropy on the original distribution is supported")
    if config.augment is not False:
        raise ValueError("augmentation is not allowed (augment must be False)")
    _validate_run_name(config.run_name)

    output_value = (
        config.output_dir
        if config.output_dir is not None
        else (Path(*PurePosixPath(OUTPUT_SUBTREE).parts) / config.run_name)
    )
    checkpoint_value = (
        config.checkpoint_dir
        if config.checkpoint_dir is not None
        else (Path(*PurePosixPath(CHECKPOINT_SUBTREE).parts) / config.run_name)
    )
    output = _path(root, output_value, OUTPUT_SUBTREE)
    checkpoint = _path(root, checkpoint_value, CHECKPOINT_SUBTREE)
    if output == checkpoint or checkpoint.is_relative_to(output) or output.is_relative_to(checkpoint):
        raise ValueError("output and checkpoint directories must differ")
    return output, checkpoint


def _validate_relative_sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{label} must be a project-relative path")
    if ".." in Path(value).parts:
        raise ValueError(f"{label} must not contain '..'")


def _validate_sha_binding(value: Any, label: str) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty binding dict")
    for relative, digest in value.items():
        _validate_relative_sha(relative, label)
        if not _is_sha256(digest):
            raise ValueError(f"{label} SHA must be 64-character lowercase hex")


def _validate_manifest_binding(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != {"train", "val"}:
        raise ValueError(f"{label} must contain only train/val")
    for split in ("train", "val"):
        item = value[split]
        if _is_sha256(item):
            continue
        if isinstance(item, dict) and set(item) == {"path", "sha256"}:
            _validate_relative_sha(item["path"], f"{label}.{split}.path")
            if not _is_sha256(item["sha256"]):
                raise ValueError(f"{label}.{split}.sha256 has an invalid format")
            continue
        raise ValueError(f"{label}.{split} must be a SHA string or {{path, sha256}}")


def _gate_identity(identity: Any, synthetic: bool) -> None:
    if not isinstance(identity, dict):
        raise ValueError("identity must be a dict")
    if synthetic:
        if identity.get("formal_training_authorized") is True:
            raise ValueError("a synthetic run must not claim formal training authorization")
        if identity.get("synthetic_only") is not True:
            raise ValueError("a synthetic identity must set synthetic_only=true explicitly")
        return
    if identity.get("formal_training_authorized") is not True:
        raise ValueError("real training lacks the identity.formal_training_authorized=true contract")
    if identity.get("synthetic_only") is True:
        raise ValueError("a real training identity must not be marked synthetic_only")
    if not _is_sha256(identity.get("plan_sha256")):
        raise ValueError("identity lacks a verified plan_sha256 (64-character lowercase SHA256)")
    _validate_sha_binding(identity.get("source_sha256"), "identity.source_sha256")
    _validate_manifest_binding(identity.get("manifest_sha256"), "identity.manifest_sha256")


def _dataset_identity(dataset: Any) -> Optional[Dict[str, Any]]:
    for attribute in ("identity", "aggregate"):
        value = getattr(dataset, attribute, None)
        if callable(value):
            try:
                result = value()
            except Exception:  # noqa: BLE001
                result = None
            if isinstance(result, dict):
                return result
        elif isinstance(value, dict):
            return value
    return None


def _gate_synthetic(
    config: TrainingConfig,
    train_dataset: Any,
    val_dataset: Any,
    stop_after: Optional[int],
) -> None:
    if config.device != "cpu" and config.device != "cuda:1":
        raise ValueError("synthetic injection allows only cpu or cuda:1")
    if stop_after is None or type(stop_after) is not int:
        raise ValueError("synthetic needs an explicit stop_after")
    if not 1 <= stop_after <= SYNTHETIC_MAX_STOP_AFTER:
        raise ValueError(f"synthetic stop_after must be in [1, {SYNTHETIC_MAX_STOP_AFTER}]")
    for name, dataset in (("train", train_dataset), ("val", val_dataset)):
        ident = _dataset_identity(dataset)
        if ident is None:
            raise ValueError(f"{name} Dataset lacks identity()/aggregate()")
        if ident.get("allow_synthetic") is not True and ident.get("contains_synthetic") is not True:
            raise ValueError(f"{name} Dataset must declare a synthetic identity explicitly")
        if not 1 <= len(dataset) < SYNTHETIC_MAX_SAMPLES:
            raise ValueError(f"synthetic {name} allows only a few samples")


def _gate_real_datasets(
    train_dataset: Any,
    val_dataset: Any,
    identity: Dict[str, Any],
    config: TrainingConfig,
) -> None:
    manifest = identity.get("manifest_sha256")
    if not isinstance(manifest, dict):
        raise ValueError("identity.manifest_sha256 must be a train/val binding dict")
    for name, dataset, split in (
        ("train", train_dataset, "train"),
        ("val", val_dataset, "val"),
    ):
        ident = _dataset_identity(dataset)
        if ident is None:
            raise ValueError(f"{name} Dataset lacks the NativeKneeDataset identity() interface")
        if ident.get("split") != split:
            raise ValueError(
                f"formal training {name} Dataset must declare split={split}, "
                "test or mixed identities are not allowed"
            )
        if ident.get("allow_synthetic") is not False or ident.get("contains_synthetic") is not False:
            raise ValueError(f"formal training {name} dataset must not contain synthetic/test-double rows")
        for flag in ("is_test", "test_only"):
            if ident.get(flag) is True:
                raise ValueError(f"formal training {name} dataset must not be test")
        if ident.get("image_size") != config.image_size:
            raise ValueError(
                f"{name} Dataset image_size must match config ({config.image_size})"
            )
        binding = manifest.get(split)
        if isinstance(binding, str):
            digest = binding
        elif isinstance(binding, dict) and set(binding) == {"path", "sha256"}:
            digest = binding["sha256"]
        else:
            digest = None
        if not _is_sha256(digest):
            raise ValueError(f"identity.manifest_sha256.{split} has no valid SHA")
        if ident.get("manifest_sha256") != digest:
            raise ValueError(
                f"{name} Dataset manifest_sha256 must match "
                f"identity.manifest_sha256.{split}."
            )


def verify_source_bindings(
    config: TrainingConfig, identity: Dict[str, Any], *, synthetic: bool
) -> None:
    bindings = identity.get("source_sha256", {})
    if not isinstance(bindings, dict):
        raise ValueError("identity.source_sha256 must be a dict")
    if not bindings:
        return
    root = Path(os.fspath(config.root)).resolve(strict=True)
    for relative, digest in bindings.items():
        _validate_relative_sha(relative, "identity.source_sha256")
        if not _is_sha256(digest):
            raise ValueError("identity.source_sha256 values must be 64-character lowercase hex")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("bound source file not found in the project: " + relative)
        if file_sha256(path) != digest:
            raise ValueError("frozen source changed: " + relative)


# RNG: touch only the selected device (no manual_seed_all); replay restores Python, numpy, CPU and that CUDA state.
def _seed_all(seed: int, device: Any) -> None:
    _require_torch()
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _capture_rng(device: Any) -> Dict[str, Any]:
    _require_torch()
    state: Dict[str, Any] = {"python": random.getstate()}
    if np is not None:
        state["numpy"] = np.random.get_state()
    state["torch_cpu"] = torch.get_rng_state()
    device = torch.device(device)
    if torch.cuda.is_available() and device.type == "cuda":
        state["torch_cuda"] = {int(device.index): torch.cuda.get_rng_state(device.index)}
    else:
        state["torch_cuda"] = {}
    return state


def _restore_rng(state: Dict[str, Any], device: Any) -> None:
    _require_torch()
    random.setstate(state["python"])
    if np is not None and "numpy" in state:
        np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    for index, value in state.get("torch_cuda", {}).items():
        if not torch.cuda.is_available():
            raise ValueError("the CUDA device needed for resuming does not exist")
        if int(index) >= torch.cuda.device_count():
            raise ValueError("the CUDA device needed for resuming does not exist")
        torch.cuda.set_rng_state(value, int(index))


def _order(length: int, seed: int, epoch: int) -> Tuple[List[int], str]:
    _require_torch()
    if np is None:
        raise RuntimeError("epoch permutation digest needs numpy")
    generator = torch.Generator().manual_seed(seed + 104729 * (epoch + 1))
    order = torch.randperm(length, generator=generator).tolist()
    digest = hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest()
    return order, digest


def _data_state(dataset: Any, progress: Dict[str, Any], seed: int) -> Dict[str, Any]:
    _, order_sha = _order(len(dataset), seed, progress["epoch"])
    return {
        "epoch": progress["epoch"],
        "next_index": progress["next_index"],
        "epoch_order_sha256": order_sha,
        "loader_policy": DATA_STATE_LOADER_POLICY,
    }


def _loader(dataset: Any, indices: List[int], batch_size: int, seed: int):
    _require_torch()
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )


def _batch(batch: Any, device: Any, image_size: int):
    _require_torch()
    if not isinstance(batch, dict) or "image" not in batch or "kl" not in batch:
        raise ValueError("batch must contain image and kl")
    images, labels = batch["image"], batch["kl"]
    if (
        not isinstance(images, torch.Tensor)
        or not isinstance(labels, torch.Tensor)
        or images.ndim != 4
        or tuple(images.shape[1:]) != (1, image_size, image_size)
        or len(images) == 0
        or not images.is_floating_point()
        or not torch.isfinite(images).all()
        or images.min() < 0
        or images.max() > 1
        or labels.dtype != torch.int64
        or labels.shape != (len(images),)
        or labels.min() < 0
        or labels.max() > 4
    ):
        raise ValueError(
            f"batch must be finite gray [B,1,{image_size},{image_size}] with KL labels [B] (0..4)"
        )
    return images.to(device), labels.to(device)


def optimizer_bundle(model: Any, config: TrainingConfig, steps_per_epoch: int, device: Any):
    _require_torch()
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if type(steps_per_epoch) is not int or steps_per_epoch < 1:
        raise ValueError("steps_per_epoch must be a positive integer")
    device = torch.device(device)
    if not hasattr(model, "head") or not hasattr(model, "encoder"):
        raise ValueError("the model must expose encoder/head attributes")
    head_params: List[Any] = []
    encoder_params: List[Any] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top = name.split(".", 1)[0]
        if top == "head" and (name == "head" or name.startswith("head.")):
            head_params.append(param)
        elif top == "encoder" and (name == "encoder" or name.startswith("encoder.")):
            encoder_params.append(param)
        else:
            raise ValueError(f"trainable parameter prefixes must be encoder.*/head.*: {name}")
    groups = []
    if head_params:
        groups.append(
            {
                "params": head_params,
                "lr": config.head_lr,
                "weight_decay": config.weight_decay,
                "group_name": "head",
            }
        )
    if encoder_params:
        groups.append(
            {
                "params": encoder_params,
                "lr": config.encoder_lr,
                "weight_decay": config.weight_decay,
                "group_name": "encoder",
            }
        )
    if not groups:
        raise ValueError("the model has no trainable parameters")
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, foreach=False)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = steps_per_epoch * config.warmup_epochs

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        position = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * (
            1.0 + math.cos(math.pi * position)
        ) / 2.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda"), init_scale=AMP_INIT_SCALE
    )
    return optimizer, scheduler, scaler


def update_group(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    batches: Any,
    config: TrainingConfig,
    device: Any,
) -> Dict[str, Any]:
    _require_torch()
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    device = torch.device(device)
    if not isinstance(batches, (list, tuple)) or not batches:
        raise ValueError("update_group needs a non-empty list of micro-batches")
    prepared = []
    group_knees = 0
    for batch in batches:
        images, labels = _batch(batch, device, config.image_size)
        prepared.append((images, labels))
        group_knees += int(len(labels))
    if group_knees <= 0:
        raise ValueError("the effective group has no samples")
    start_rng = _capture_rng(device)
    limit = int(config.amp_max_retries)
    retries = 0
    params = [p for group in optimizer.param_groups for p in group["params"]]
    while True:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        correct = 0
        finite_loss = True
        for images, labels in prepared:
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(images)
            if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (
                len(images),
                5,
            ):
                raise ValueError("single-knee model output must be [B,5]")
            loss = F.cross_entropy(logits.float(), labels, reduction="sum")
            if not bool(torch.isfinite(loss)):
                finite_loss = False
            loss_sum += float(loss.detach().cpu())
            correct += int((logits.detach().argmax(1) == labels).sum().item())
            scaler.scale(loss / group_knees).backward()
        # Always unscale, even for a non-finite loss, so GradScaler records found_inf and update() lowers the scale.
        scaler.unscale_(optimizer)
        # Pass the parameter list to clip_grad_norm_; bare grad tensors are not clipped at all.
        # Non-finite gradients, or a non-finite norm after clipping, fail the step: it is replayed and no samples are consumed.
        finite_grads = finite_loss and all(
            param.grad is not None and bool(torch.isfinite(param.grad).all())
            for param in params
        )
        grad_norm = None
        if finite_grads:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, float(config.max_grad_norm))
            finite_grads = grad_norm is not None and bool(torch.isfinite(grad_norm))
        if finite_grads:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            return {
                "knees": group_knees,
                "loss_sum": loss_sum,
                "correct": correct,
                "gradient_norm": None
                if grad_norm is None
                else float(grad_norm.detach().cpu()),
                "amp_retries": retries,
                "amp_scale": float(scaler.get_scale()),
            }
        scaler.update()
        retries += 1
        if retries >= limit:
            raise FloatingPointError(
                f"the effective group still had no successful update after {limit} AMP/non-finite-gradient replays; "
                "the group is not counted as a step and consumes no samples"
            )
        _restore_rng(start_rng, device)


def classification_metrics(labels: Any, probabilities: Any, cross_entropies: Any) -> Dict[str, Any]:
    if np is None:
        raise RuntimeError("classification_metrics needs numpy")
    raw = np.asarray(labels)
    if raw.dtype.kind == "b":
        raise ValueError("labels must not be boolean")
    if raw.dtype.kind not in "iuf":
        raise ValueError("labels must be an integer array or a float array of integer values")
    if raw.dtype.kind == "f":
        values = np.asarray(labels, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("labels must not contain NaN/Inf")
        if not np.array_equal(values, np.floor(values)):
            raise ValueError("float labels must hold integer values (no silent truncation)")
    y = raw.astype(np.int64).reshape(-1)
    probability = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or probability.shape != (len(y), 5) or not len(y):
        raise ValueError("invalid shape for per-knee metric input")
    if (
        not np.isfinite(probability).all()
        or (probability < 0).any()
        or (probability > 1).any()
        or ((y < 0) | (y > 4)).any()
        or not np.allclose(probability.sum(1), 1, atol=1e-6)
    ):
        raise ValueError("invalid per-knee probabilities")
    ce = np.asarray(cross_entropies, dtype=np.float64).reshape(-1)
    if ce.shape != y.shape or not np.isfinite(ce).all() or (ce < 0).any():
        raise ValueError("per-knee cross-entropy must be finite and non-negative")
    hard = probability.argmax(1)
    cm = np.zeros((5, 5), dtype=np.int64)
    for row in range(5):
        for col in range(5):
            cm[row, col] = int(((y == row) & (hard == col)).sum())
    expected = np.outer(cm.sum(1), cm.sum(0)) / len(y)
    distances = np.square(np.arange(5)[:, None] - np.arange(5)[None, :])
    supports = cm.sum(1)
    if int((supports > 0).sum()) <= 1:
        # A single true class makes the expected matrix degenerate, so QWK is undefined; return None with a reason.
        qwk = None
        qwk_reason = "single_true_class_not_estimable"
    else:
        denominator = float((expected * distances).sum())
        qwk = 1.0 - float((cm * distances).sum()) / denominator if denominator else None
        qwk_reason = None if qwk is not None else "zero_expectation_denominator"
    per_class: Dict[str, Dict[str, Any]] = {}
    f1_values: List[float] = []
    for level in range(5):
        tp = int(cm[level, level])
        fp = int(cm[:, level].sum()) - tp
        fn = int(cm[level, :].sum()) - tp
        support = int(cm[level, :].sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if support > 0 else None
        f1 = (
            2.0 * precision * float(recall) / (precision + float(recall))
            if recall is not None and (precision + recall) > 0
            else 0.0
        )
        f1_values.append(f1)
        per_class[str(level)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return _clean(
        {
            "knees": len(y),
            "cross_entropy": float(ce.mean()),
            "qwk": qwk,
            "qwk_reason": qwk_reason,
            "mae": float(np.abs(y - hard).mean()),
            "macro_f1": float(np.mean(f1_values)),
            "accuracy": float(np.mean(hard == y)),
            "abs_error_ge2_rate": float(np.mean(np.abs(y - hard) >= 2)),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
        }
    )


def validate(model: Any, dataset: Any, config: TrainingConfig, device: Any) -> Dict[str, Any]:
    _require_torch()
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    device = torch.device(device)
    was_training = bool(model.training)
    model.eval()
    targets: List[Any] = []
    probabilities: List[Any] = []
    losses: List[Any] = []
    try:
        with torch.inference_mode():
            for batch in _loader(
                dataset,
                list(range(len(dataset))),
                config.micro_batch_size,
                config.seed + 9,
            ):
                images, labels = _batch(batch, device, config.image_size)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(images)
                if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (
                    len(images),
                    5,
                ):
                    raise ValueError("single-knee model output must be [B,5]")
                logits = logits.float()
                loss = F.cross_entropy(logits, labels, reduction="none")
                if not torch.isfinite(logits).all() or not torch.isfinite(loss).all():
                    raise FloatingPointError("validation logits/loss are non-finite")
                probabilities.append(logits.softmax(-1).cpu().numpy())
                targets.append(labels.cpu().numpy())
                losses.append(loss.cpu().numpy())
    finally:
        model.train(was_training)
    if not targets:
        raise ValueError("validation Dataset must not be empty")
    return classification_metrics(
        np.concatenate(targets),
        np.concatenate(probabilities),
        np.concatenate(losses),
    )


def _best_update(model: Any, progress: Dict[str, Any], metrics: Dict[str, Any], epoch: int) -> bool:
    qwk = metrics.get("qwk")
    mae = metrics.get("mae")
    if qwk is None or mae is None:
        return False
    key = [float(qwk), float(-mae)]
    if progress["best_key"] is not None and tuple(key) <= tuple(progress["best_key"]):
        return False
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    buffers = set(dict(model.named_buffers()))
    progress["best_state"] = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in trainable or name in buffers
    }
    progress["best_key"] = key
    progress["best_epoch"] = epoch
    progress["best_step"] = progress["step"]
    progress["best_metrics"] = metrics
    return True


def _materialize_best(
    model: Any,
    progress: Dict[str, Any],
    identity: Dict[str, Any],
    output: Path,
    config: TrainingConfig,
    synthetic: bool,
) -> Optional[Dict[str, Any]]:
    _require_torch()
    if progress["best_state"] is None:
        return None
    model_path = output / "best_model.pt"
    meta_path = output / "best_selection.json"
    if model_path.exists() and meta_path.exists():
        try:
            old = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            old = {}
        if (
            old.get("identity") == identity
            and old.get("selection_key") == progress["best_key"]
            and old.get("step") == progress["best_step"]
            and old.get("model_sha256") == file_sha256(model_path)
        ):
            return old
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    state.update(progress["best_state"])
    payload = {
        "model": state,
        "identity": identity,
        "image_size": config.image_size,
        "epoch": progress["best_epoch"],
        "step": progress["best_step"],
        "selection_key": progress["best_key"],
        "validation_metrics": progress["best_metrics"],
        "test_evaluated": False,
        "synthetic_only": synthetic,
        "clinical_validation": False,
    }
    _atomic_write(model_path, lambda stream: torch.save(payload, stream))
    restored = torch.load(model_path, map_location="cpu", weights_only=True)
    if (
        restored["selection_key"] != progress["best_key"]
        or restored["identity"] != identity
        or restored["synthetic_only"] != synthetic
    ):
        raise ValueError("atomic read-back of the best model does not match")
    metadata = {key: value for key, value in payload.items() if key != "model"}
    metadata["model_sha256"] = file_sha256(model_path)
    metadata["selection_rule"] = (
        "val QWK, then lower val MAE; undefined single-class QWK is not selectable"
    )
    atomic_json(meta_path, metadata)
    return metadata


def _trim_metrics(path: Path, step: int) -> None:
    if not path.exists():
        return
    retained: List[Dict[str, Any]] = []
    seen = set()
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if row["step"] <= step and row["step"] not in seen:
            retained.append(row)
            seen.add(row["step"])
    data = "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in retained
    )
    _atomic_write(path, lambda stream: stream.write(data.encode()))


def synthetic_backup(directory: Any, *, run_name: str, keep_last: int) -> Dict[str, Any]:
    return {
        "state": "synthetic_hook_only",
        "independent_storage": False,
        "nas_write_performed": False,
        "run_name": run_name,
        "keep_last": keep_last,
    }


def _default_checkpoint_factory(directory: Any, keep_last: int = 3):
    try:
        from drai.training.checkpoint import CheckpointManager
    except ImportError as error:  # pragma: no cover
        raise RuntimeError(
            "formal training needs drai.training.checkpoint.CheckpointManager; "
            "synthetic acceptance must inject checkpoint_factory explicitly"
        ) from error
    return CheckpointManager(Path(directory), keep_last=keep_last)


def _validate_loaded_progress(progress: Any, config: TrainingConfig, train_count: int) -> None:
    if not isinstance(progress, dict):
        raise ValueError("checkpoint progress is not a dict")
    required = {
        "step", "successful_step", "epoch", "next_index", "epoch_loss_sum",
        "epoch_correct", "epoch_knees", "knees_seen", "best_key", "best_epoch",
        "best_step", "best_metrics", "best_state", "history",
    }
    if not required.issubset(progress):
        raise ValueError("checkpoint progress lacks required fields")
    if (
        type(progress["step"]) is not int
        or type(progress["epoch"]) is not int
        or type(progress["next_index"]) is not int
        or type(progress["epoch_knees"]) is not int
        or type(progress["knees_seen"]) is not int
        or progress["step"] < 0
        or progress["successful_step"] != progress["step"]
    ):
        raise ValueError("invalid checkpoint step/epoch/next_index")
    if not 0 <= progress["epoch"] <= config.epochs:
        raise ValueError("checkpoint epoch out of range")
    if not 0 <= progress["next_index"] <= train_count:
        raise ValueError("checkpoint sample cursor out of range")
    steps_per_epoch = math.ceil(train_count / config.effective_batch_size)
    epoch = progress["epoch"]
    next_index = progress["next_index"]
    if epoch == config.epochs and next_index != 0:
        raise ValueError("a checkpoint that completed all epochs must not have an extra sample cursor")
    if next_index not in (0, train_count) and next_index % config.effective_batch_size:
        raise ValueError("checkpoint sample cursor must be an effective-group boundary or the end of the epoch")
    completed = (
        steps_per_epoch
        if next_index == train_count
        else next_index // config.effective_batch_size
    )
    if progress["step"] != epoch * steps_per_epoch + completed:
        raise ValueError("checkpoint step does not match epoch/completed groups")
    if progress["epoch_knees"] != next_index:
        raise ValueError("checkpoint epoch_knees must equal next_index")
    if progress["knees_seen"] != epoch * train_count + next_index:
        raise ValueError("checkpoint knees_seen does not match epoch/next_index")
    history = progress["history"]
    if not isinstance(history, list):
        raise ValueError("invalid checkpoint history")
    if len(history) != epoch:
        raise ValueError("checkpoint history length must equal epoch")
    for index, entry in enumerate(history):
        if not isinstance(entry, dict) or entry.get("epoch") != index:
            raise ValueError("checkpoint history entries must be in epoch order 0..epoch-1")


def fit(
    *,
    config: TrainingConfig,
    model_factory: Any,
    train_dataset: Any,
    val_dataset: Any,
    identity: Dict[str, Any],
    resume: bool = False,
    stop_after: Optional[int] = None,
    synthetic: bool = False,
    checkpoint_factory: Any = None,
    backup_fn: Any = None,
) -> Dict[str, Any]:
    _require_torch()
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if not callable(model_factory):
        raise TypeError("model_factory must be callable")
    if type(resume) is not bool or type(synthetic) is not bool:
        raise TypeError("resume/synthetic must be booleans")

    _gate_identity(identity, synthetic)
    if not synthetic and checkpoint_factory is not None:
        raise ValueError(
            "formal training must use the production CheckpointManager; checkpoint_factory must not be injected"
        )
    device = torch.device(config.device)
    output_dir, checkpoint_dir = validate_config(config)
    if synthetic:
        _gate_synthetic(config, train_dataset, val_dataset, stop_after)
    else:
        _gate_real_datasets(train_dataset, val_dataset, identity, config)
        if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
            raise ValueError("stop_after must be None or a positive integer")
    verify_source_bindings(config, identity, synthetic=synthetic)
    if backup_fn is None:
        if synthetic:
            backup_fn = synthetic_backup
        else:
            raise ValueError("formal training needs backup_fn (NAS backup)")
    if not callable(backup_fn):
        raise TypeError("backup_fn must be callable")

    torch.set_num_threads(config.cpu_threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("train/val Dataset must not be empty")
    import fcntl

    with (output_dir / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("the output directory is locked by another training run") from error
        identity_path = output_dir / "execution_identity.json"
        if identity_path.exists():
            try:
                old = json.loads(identity_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("cannot resolve the existing run identity") from error
            if not resume or old != identity:
                raise ValueError("an existing run needs an explicit resume with an identical identity")
        elif resume:
            raise ValueError("no existing run to resume")
        if not resume and any(checkpoint_dir.glob("step_*.json")):
            raise ValueError("the checkpoint directory already has artifacts; a fresh run must not overwrite them")
        if not identity_path.exists():
            atomic_json(identity_path, identity)

        if device.type == "cuda":
            torch.cuda.set_device(device)
        _seed_all(config.seed, device)
        model = model_factory().to(device)
        _seed_all(config.seed + 20000, device)
        parameter_count = sum(p.numel() for p in model.parameters())
        if synthetic and parameter_count >= SYNTHETIC_MAX_PARAMETERS:
            raise ValueError("synthetic acceptance must not inject the full patient training model")
        if not synthetic and parameter_count < REAL_MIN_PARAMETERS:
            raise ValueError("formal training needs the full backbone model; test injection refused")
        steps_per_epoch = math.ceil(len(train_dataset) / config.effective_batch_size)
        optimizer, scheduler, scaler = optimizer_bundle(model, config, steps_per_epoch, device)
        factory = checkpoint_factory or _default_checkpoint_factory
        manager = factory(checkpoint_dir, keep_last=config.keep_last)
        metadata = {
            "schema": SCHEMA_META,
            "identity": identity,
            "config": config.to_dict(),
            "train_count": len(train_dataset),
            "val_count": len(val_dataset),
            "image_size": config.image_size,
            "input_shape": ["B", 1, config.image_size, config.image_size],
            "output_shape": list(OUTPUT_SHAPE),
            "target": TARGET,
            "train_dataset_identity": _dataset_identity(train_dataset),
            "val_dataset_identity": _dataset_identity(val_dataset),
        }
        progress: Dict[str, Any] = {
            "step": 0,
            "successful_step": 0,
            "epoch": 0,
            "next_index": 0,
            "knees_seen": 0,
            "epoch_loss_sum": 0.0,
            "epoch_correct": 0,
            "epoch_knees": 0,
            "best_key": None,
            "best_epoch": None,
            "best_step": None,
            "best_metrics": None,
            "best_state": None,
            "history": [],
        }
        if resume:
            progress, data_state = manager.load_latest(
                model, optimizer, scheduler, scaler, expected_metadata=metadata
            )
            _validate_loaded_progress(progress, config, len(train_dataset))
            expected_state = _data_state(train_dataset, progress, config.seed)
            if data_state != expected_state:
                raise ValueError("resumed data progress/epoch order does not match")
            _trim_metrics(output_dir / "metrics.jsonl", progress["step"])
            _materialize_best(model, progress, identity, output_dir, config, synthetic)
        last_backup: Optional[Dict[str, Any]] = None

        def publish(state: str, error: Optional[str] = None) -> Dict[str, Any]:
            status = {
                "state": state,
                "step": progress["step"],
                "epoch": progress["epoch"],
                "next_index": progress["next_index"],
                "knees_seen": progress["knees_seen"],
                "best_epoch": progress["best_epoch"],
                "best_metrics": progress["best_metrics"],
                "selected_on": "val",
                "test_evaluated": False,
                "synthetic_only": synthetic,
                "clinical_validation": False,
                "image_size": config.image_size,
                "plan_sha256": identity.get("plan_sha256"),
                "checkpoint_rejected": list(
                    getattr(manager, "last_rejected", [])
                ),
                "epoch_history": progress["history"],
            }
            if error is not None:
                status["error"] = error
            if last_backup is not None:
                status["nas_backup"] = last_backup
            atomic_json(output_dir / "status.json", status)
            atomic_json(output_dir / "epoch_history.json", progress["history"])
            return status

        def save() -> None:
            nonlocal last_backup
            verify_source_bindings(config, identity, synthetic=synthetic)
            manager.save(
                model,
                optimizer,
                scheduler,
                scaler,
                progress,
                metadata,
                _data_state(train_dataset, progress, config.seed),
            )
            receipt = backup_fn(
                checkpoint_dir, run_name=config.run_name, keep_last=config.keep_last
            )
            last_backup = receipt
            atomic_json(output_dir / "nas_backup.json", receipt)

        try:
            verify_source_bindings(config, identity, synthetic=synthetic)
            if not resume:
                save()
            publish("running")
            while progress["epoch"] < config.epochs:
                epoch = progress["epoch"]
                order, _ = _order(len(train_dataset), config.seed, epoch)
                cursor = progress["next_index"]
                if not 0 <= cursor <= len(order):
                    raise ValueError("resumed sample cursor out of range")
                remaining = order[cursor:]
                iterator = iter(
                    _loader(
                        train_dataset,
                        remaining,
                        config.micro_batch_size,
                        config.seed + epoch,
                    )
                )
                model.train()
                while progress["next_index"] < len(order):
                    target_count = min(
                        config.effective_batch_size,
                        len(order) - progress["next_index"],
                    )
                    batches, loaded = [], 0
                    while loaded < target_count:
                        batch = next(iterator)
                        batches.append(batch)
                        loaded += len(batch["kl"])
                    if loaded != target_count:
                        raise ValueError("micro-batch grouping does not match the sample group boundaries")
                    record = update_group(
                        model, optimizer, scheduler, scaler, batches, config, device
                    )
                    progress["step"] += 1
                    progress["successful_step"] += 1
                    progress["next_index"] += loaded
                    progress["knees_seen"] += record["knees"]
                    progress["epoch_loss_sum"] += record["loss_sum"]
                    progress["epoch_correct"] += record["correct"]
                    progress["epoch_knees"] += record["knees"]
                    row = {
                        "step": progress["step"],
                        "epoch": epoch,
                        "epoch_fraction": epoch + progress["next_index"] / len(order),
                        "loss": record["loss_sum"] / record["knees"],
                        "gradient_norm": record["gradient_norm"],
                        "group_knees": record["knees"],
                        "knees_seen": progress["knees_seen"],
                        "learning_rate": {
                            group["group_name"]: group["lr"]
                            for group in optimizer.param_groups
                        },
                        "amp_scale": record["amp_scale"],
                        "amp_retries": record["amp_retries"],
                        "wall_time_ms": int(time.time() * 1000),
                    }
                    with (output_dir / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    at_end = progress["next_index"] == len(order)
                    stop = stop_after is not None and progress["step"] >= stop_after
                    if not at_end and (
                        stop
                        or progress["step"] % config.checkpoint_every_steps == 0
                    ):
                        save()
                    if stop and not at_end:
                        return publish("stopped_at_successful_update")
                    publish("running")
                metrics = validate(model, val_dataset, config, device)
                improved = _best_update(model, progress, metrics, epoch)
                if improved:
                    _materialize_best(model, progress, identity, output_dir, config, synthetic)
                progress["history"].append(
                    {
                        "epoch": epoch,
                        "step": progress["step"],
                        "train_knees": progress["epoch_knees"],
                        "train_cross_entropy": progress["epoch_loss_sum"]
                        / progress["epoch_knees"],
                        "train_accuracy": progress["epoch_correct"]
                        / progress["epoch_knees"],
                        "validation": metrics,
                        "selected_as_best": improved,
                    }
                )
                progress["epoch"] += 1
                progress["next_index"] = 0
                progress["epoch_loss_sum"] = 0.0
                progress["epoch_correct"] = 0
                progress["epoch_knees"] = 0
                save()
                if (
                    stop_after is not None
                    and progress["step"] >= stop_after
                    and progress["epoch"] < config.epochs
                ):
                    return publish("stopped_at_successful_update")
            best = _materialize_best(model, progress, identity, output_dir, config, synthetic)
            result = publish("completed_train_val")
            result["best_selection"] = best
            result["bootstrap_performed"] = False
            atomic_json(output_dir / "result.json", result)
            return result
        except Exception as error:
            publish("failed", f"{type(error).__name__}: {error}")
            raise


__all__ = [
    "AMP_INIT_SCALE",
    "BACKBONES",
    "CHECKPOINT_SUBTREE",
    "IMAGE_SIZES",
    "INPUT_SHAPE",
    "MODES",
    "OUTPUT_SHAPE",
    "OUTPUT_SUBTREE",
    "PACKAGE",
    "RUN_PREFIX",
    "SCHEMA_META",
    "SYNTHETIC_MAX_SAMPLES",
    "TARGET",
    "TrainingConfig",
    "_batch",
    "_best_update",
    "_capture_rng",
    "_clean",
    "_data_state",
    "_default_checkpoint_factory",
    "_gate_identity",
    "_gate_real_datasets",
    "_gate_synthetic",
    "_loader",
    "_materialize_best",
    "_order",
    "_restore_rng",
    "_seed_all",
    "_trim_metrics",
    "_validate_loaded_progress",
    "atomic_json",
    "classification_metrics",
    "file_sha256",
    "fit",
    "optimizer_bundle",
    "synthetic_backup",
    "update_group",
    "validate",
    "validate_config",
    "verify_source_bindings",
]
