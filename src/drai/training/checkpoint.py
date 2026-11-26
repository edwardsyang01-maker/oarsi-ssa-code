import hashlib
import json
import os
import pickle
import random
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def capture_rng():
    kind, keys, pos, gauss, cached = np.random.get_state()
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    return {
        "python": random.getstate(),
        "numpy_kind": kind,
        "numpy_keys": torch.from_numpy(keys.astype(np.int64)),
        "numpy_pos": pos,
        "numpy_gauss": gauss,
        "numpy_cached": cached,
        "torch_cpu": torch.get_rng_state(),
        "cuda_devices": devices,
        "torch_cuda": [torch.cuda.get_rng_state(device) for device in devices],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(
        (
            state["numpy_kind"],
            state["numpy_keys"].numpy().astype(np.uint32),
            state["numpy_pos"],
            state["numpy_gauss"],
            state["numpy_cached"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    devices = state.get("cuda_devices", list(range(len(state["torch_cuda"]))))
    if any(device >= torch.cuda.device_count() for device in devices):
        raise ValueError("the CUDA device needed for resuming does not exist")
    for device, value in zip(devices, state["torch_cuda"], strict=True):
        torch.cuda.set_rng_state(value, device)


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(v) for v in value)
    return value


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _atomic_write(path, write):
    temporary = path.with_name(".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


class CheckpointManager:
    def __init__(self, directory, keep_last=3):
        if keep_last < 2:
            raise ValueError("keep at least two valid versions")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.last_rejected = []

    def _entries(self):
        entries = []
        for path in self.directory.glob("step_*.json"):
            try:
                value = json.loads(path.read_text())
                if Path(value["file"]).name != value["file"]:
                    raise ValueError("invalid checkpoint path")
                entries.append(
                    (int(value["step"]), path.stat().st_mtime_ns, path, value)
                )
            except (ValueError, KeyError, OSError):
                continue
        return sorted(entries, reverse=True)

    def save(self, model, optimizer, scheduler, scaler, progress, metadata, data_state):
        active = dist.is_initialized()
        rank = dist.get_rank() if active else 0
        world = dist.get_world_size() if active else 1
        status = [None]
        snapshot = None
        if rank == 0:
            try:
                underlying = model.module if hasattr(model, "module") else model
                snapshot = _to_cpu(
                    {
                        "model": underlying.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                    }
                )
            # Propagate any preparation failure at the save barrier so other ranks do not wait forever.
            except Exception as error:  # noqa: BLE001
                status[0] = f"state preparation failed: {type(error).__name__}: {error}"
        local = {"rng": capture_rng(), "data_state": data_state}
        per_rank = [None] * world if rank == 0 else None
        try:
            if active:
                dist.gather_object(local, per_rank, dst=0)
            else:
                per_rank = [local]
        # Root decode errors in object gather happen after the collective; notify the other ranks.
        except Exception as error:  # noqa: BLE001
            status[0] = f"state gathering failed: {type(error).__name__}: {error}"
        if rank == 0 and status[0] is None:
            try:
                value = {
                    "schema_version": 1,
                    "world_size": world,
                    **snapshot,
                    "progress": progress,
                    "metadata": metadata,
                    "per_rank": per_rank,
                }
                filename = f"step_{progress['step']:09d}_{uuid.uuid4().hex[:8]}.pt"
                path = self.directory / filename
                _atomic_write(path, lambda f: torch.save(value, f))
                torch.load(path, map_location="cpu", weights_only=True)
                info = {
                    "file": filename,
                    "step": progress["step"],
                    "sha256": sha256(path),
                    "schema_version": 1,
                }
                encoded = (json.dumps(info, indent=2) + "\n").encode()
                _atomic_write(path.with_suffix(".json"), lambda f: f.write(encoded))
                self._prune()
            # Broadcast a rank-0 save failure so the other ranks do not wait.
            except Exception as error:  # noqa: BLE001
                status[0] = f"{type(error).__name__}: {error}"
        if active:
            dist.broadcast_object_list(status, src=0)
        if status[0]:
            raise RuntimeError("checkpoint save failed: " + status[0])

    def _prune(self):
        valid = []
        for _, _, sidecar, info in self._entries():
            path = self.directory / info["file"]
            if path.exists() and sha256(path) == info["sha256"]:
                valid.append((sidecar, path))
        for sidecar, path in valid[self.keep_last :]:
            sidecar.unlink()
            path.unlink()

    def load_latest(self, model, optimizer, scheduler, scaler, expected_metadata):
        active = dist.is_initialized()
        rank = dist.get_rank() if active else 0
        world = dist.get_world_size() if active else 1
        self.last_rejected = []
        for _, _, _, info in self._entries():
            path = self.directory / info["file"]
            try:
                if sha256(path) != info["sha256"]:
                    raise ValueError("SHA256 mismatch")
                value = torch.load(path, map_location="cpu", weights_only=True)
                if value["schema_version"] != 1 or value["world_size"] != world:
                    raise ValueError("checkpoint version/world_size mismatch")
                if value["metadata"] != expected_metadata:
                    raise ValueError("data or configuration fingerprint mismatch")
                underlying = model.module if hasattr(model, "module") else model
                underlying.load_state_dict(value["model"], strict=True)
                optimizer.load_state_dict(value["optimizer"])
                scheduler.load_state_dict(value["scheduler"])
                scaler.load_state_dict(value["scaler"])
                restore_rng(value["per_rank"][rank]["rng"])
                return value["progress"], value["per_rank"][rank]["data_state"]
            except (
                ValueError,
                KeyError,
                OSError,
                RuntimeError,
                EOFError,
                pickle.UnpicklingError,
            ) as error:
                self.last_rejected.append({"file": path.name, "reason": str(error)})
        raise RuntimeError(
            "no resumable checkpoint with matching data/configuration: " + str(self.last_rejected)
        )
