import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .preprocess_io import file_sha256, project_path

FULL_PA_POLICY = "full-bilateral-pa-letterbox-photometric-v1"


def letterbox(gray, valid, size):
    if gray.ndim != 2 or valid.shape != gray.shape or valid.dtype != bool:
        raise ValueError("whole-film gray and bool mask need the same 2-D shape")
    if (
        gray.dtype != np.float32
        or not np.isfinite(gray).all()
        or gray.min() < 0
        or gray.max() > 1
    ):
        raise ValueError("whole-film input must be finite [0,1] float32")
    h, w = gray.shape
    ratio = size / max(h, w)
    nh, nw = max(1, round(h * ratio)), max(1, round(w * ratio))
    scaled = np.asarray(
        Image.fromarray(gray).resize((nw, nh), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    mask = np.asarray(
        Image.fromarray(valid.astype(np.uint8)).resize(
            (nw, nh), Image.Resampling.NEAREST
        ),
        dtype=bool,
    )
    output = np.zeros((size, size), dtype=np.float32)
    out_mask = np.zeros((size, size), dtype=bool)
    x, y = (size - nw) // 2, (size - nh) // 2
    output[y : y + nh, x : x + nw] = np.clip(scaled, 0, 1)
    out_mask[y : y + nh, x : x + nw] = mask
    output[~out_mask] = 0
    return output, out_mask, [x, y, x + nw, y + nh]


class FullPaTwoViewDataset(Dataset):
    def __init__(
        self, root, manifest, image_size=448, seed=20261007, allow_exploratory=False
    ):
        self.root = Path(root).resolve()
        self.seed, self.epoch, self.image_size = seed, 0, image_size
        self.intensity_policy_id = FULL_PA_POLICY
        if not allow_exploratory:
            raise ValueError("the whole-film policy currently allows only explicit exploratory pretraining; formal QC is not finished")
        with Path(manifest).open(newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        seen = set()
        for row in self.rows:
            if row["split"] != "train" or row["patient_id"] in seen:
                raise ValueError("the whole-film training pool must be train only, with one fixed image per patient")
            seen.add(row["patient_id"])
            if (
                row["input_policy_id"] != FULL_PA_POLICY
                or row["exploratory_ready"] != "1"
            ):
                raise ValueError("training-pool technical check or policy version failed")
            for path_key, sha_key in (
                ("gray01_npy", "gray01_sha256"),
                ("valid_mask_npy", "valid_mask_sha256"),
                ("input_evidence_json", "input_evidence_sha256"),
            ):
                path = project_path(self.root, row[path_key])
                if file_sha256(path) != row[sha_key]:
                    raise ValueError("whole-film input content fingerprint changed")
            evidence = json.loads(Path(row["input_evidence_json"]).read_text())
            if (
                evidence["input_policy_id"] != FULL_PA_POLICY
                or evidence["input_policy_status"] != "engineering_validated"
            ):
                raise ValueError("whole-film input evidence does not match")
        if not self.rows:
            raise ValueError("exploratory training pool is empty")

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _view(self, image, mask, sample_id, view):
        key = f"{self.seed}|{self.epoch}|{sample_id}|{view}"
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") & (
            (1 << 63) - 1
        )
        generator = torch.Generator().manual_seed(seed)

        def draw():
            return torch.rand((), generator=generator).item()

        gain, contrast, bias = (
            0.90 + 0.20 * draw(),
            0.90 + 0.20 * draw(),
            -0.03 + 0.06 * draw(),
        )
        mean = image[mask].mean()
        result = ((image - mean) * contrast + mean) * gain + bias
        if draw() < 0.2:
            result = TF.gaussian_blur(result, [5, 5], sigma=[0.1 + 0.5 * draw()] * 2)
        if draw() < 0.5:
            result = result + torch.randn(result.shape, generator=generator) * (
                0.003 + 0.007 * draw()
            )
        result = result.clamp(0, 1)
        result[~mask] = 0
        return (result - 0.5) / 0.5

    def __getitem__(self, index):
        row = self.rows[index]
        gray = np.load(row["gray01_npy"], allow_pickle=False)
        valid = np.load(row["valid_mask_npy"], allow_pickle=False)
        if gray.dtype != np.float32 or valid.dtype != bool or gray.shape != valid.shape:
            raise ValueError("invalid cache dtype/shape")
        resized, mask, _ = letterbox(gray, valid, self.image_size)
        image = torch.from_numpy(resized).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return {
            "view1": self._view(image, mask, row["roi_id"], 0),
            "view2": self._view(image, mask, row["roi_id"], 1),
            "patient_id": row["patient_id"],
            "roi_id": row["roi_id"],
        }
