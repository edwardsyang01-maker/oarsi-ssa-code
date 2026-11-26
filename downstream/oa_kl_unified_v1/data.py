from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from .preprocessing import ModelInputPreprocessingError, prepare_model_input
except ImportError:  # pragma: no cover
    if __package__:
        raise
    from preprocessing import ModelInputPreprocessingError, prepare_model_input

SCHEMA = "oa-kl-native-knee-cache-v1"
SUPPORTED_SPLITS = ("train", "val")
SUPPORTED_IMAGE_SIZES = (224, 448)
CACHE_SHAPE = (1, 448, 448)
VISITS = frozenset({"V00", "V01", "V03", "V05", "V06", "V08", "V10"})
MODALITIES = frozenset({"CR", "DX", "RG"})
IMAGE_POSITIONS = frozenset({"image_left", "image_right"})
CLINICAL_SIDES = frozenset({"RIGHT", "LEFT", "UNKNOWN"})
LABEL_POLICIES = frozenset({"header_mapped", "equal_kl_no_side_inference"})
SOURCE_MODES = frozenset({"native_dicom", "simulated_test_double"})
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

__all__ = [
    "NativeKneeCacheError",
    "NativeKneeDataset",
    "SCHEMA",
    "probe_io",
]


class NativeKneeCacheError(ValueError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _require_str(value: Any, where: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise NativeKneeCacheError(f"{where}: {field} must be a non-empty string")
    return value


def _require_sha256(value: Any, where: str, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise NativeKneeCacheError(
            f"{where}: {field} must be a 64-character lowercase hex SHA256"
        )
    return value


def _require_bool(value: Any, where: str, field: str, expected: Optional[bool] = None) -> bool:
    if type(value) is not bool:
        raise NativeKneeCacheError(f"{where}: {field} must be a boolean")
    if expected is not None and value is not expected:
        raise NativeKneeCacheError(f"{where}: {field} must be {expected}")
    return value


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class _DuplicateJsonKey(ValueError):
    pass


def _object_without_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


@dataclass(frozen=True)
class _SampleSpec:
    line: int
    candidate_id: str
    film_id: str
    patient_id: str
    visit: str
    split: str
    modality: str
    kl: int
    image_position: str
    clinical_side: str
    label_policy: str
    bbox: Tuple[int, int, int, int]
    transform: Dict[str, Any]
    gray_path: Path
    gray_sha256: str
    mask_path: Path
    mask_sha256: str
    evidence_path: Path
    evidence_sha256: str
    native_source: Dict[str, Any]
    source_mode: str
    test_double_used: bool


class NativeKneeDataset:
    def __init__(
        self,
        project_root: os.PathLike[str] | str,
        manifest_path: os.PathLike[str] | str,
        manifest_sha256: str,
        cache_root: os.PathLike[str] | str,
        split: str,
        image_size: int = 448,
        *,
        allow_synthetic: bool = False,
    ) -> None:
        if split not in SUPPORTED_SPLITS:
            raise NativeKneeCacheError("split must be train or val; test is not available")
        if not _is_int(image_size) or int(image_size) not in SUPPORTED_IMAGE_SIZES:
            raise NativeKneeCacheError("image_size must be 224 or 448")
        if type(allow_synthetic) is not bool:
            raise NativeKneeCacheError("allow_synthetic must be a boolean")
        _require_sha256(manifest_sha256, "constructor", "manifest_sha256")

        try:
            self.project_root = Path(os.fspath(project_root)).resolve(strict=True)
        except (TypeError, OSError, ValueError) as exc:
            raise NativeKneeCacheError("project_root is invalid or inaccessible") from exc
        if not self.project_root.is_dir():
            raise NativeKneeCacheError("project_root must be a directory")

        try:
            outputs_root = (self.project_root / "outputs").resolve(strict=True)
            processed_root = (self.project_root / "data" / "processed").resolve(strict=True)
        except (OSError, ValueError) as exc:
            raise NativeKneeCacheError("project outputs or data/processed does not exist") from exc
        if not outputs_root.is_dir() or not processed_root.is_dir():
            raise NativeKneeCacheError("project outputs or data/processed must be a directory")
        if not outputs_root.is_relative_to(self.project_root) or not processed_root.is_relative_to(
            self.project_root
        ):
            raise NativeKneeCacheError("project outputs or data/processed escapes project_root")

        try:
            manifest = Path(os.fspath(manifest_path))
        except (TypeError, ValueError) as exc:
            raise NativeKneeCacheError("invalid manifest_path") from exc
        if not manifest.is_absolute():
            manifest = self.project_root / manifest
        try:
            manifest = manifest.resolve(strict=True)
        except (OSError, ValueError) as exc:
            raise NativeKneeCacheError("manifest does not exist or cannot be resolved") from exc
        if not manifest.is_relative_to(outputs_root) or not manifest.is_file():
            raise NativeKneeCacheError("manifest must be a regular file inside project/outputs")

        try:
            cache = Path(os.fspath(cache_root))
        except (TypeError, ValueError) as exc:
            raise NativeKneeCacheError("invalid cache_root") from exc
        if not cache.is_absolute():
            cache = self.project_root / cache
        try:
            cache = cache.resolve(strict=True)
        except (OSError, ValueError) as exc:
            raise NativeKneeCacheError("cache_root does not exist or cannot be resolved") from exc
        if not cache.is_relative_to(processed_root) or not cache.is_dir():
            raise NativeKneeCacheError("cache_root must be inside project/data/processed")

        self.manifest_path = manifest
        self.manifest_sha256 = manifest_sha256
        self.cache_root = cache
        self.split = split
        self.image_size = int(image_size)
        self.allow_synthetic = allow_synthetic

        # Verify the SHA of the manifest bytes, then parse those same bytes.
        try:
            raw_manifest = manifest.read_bytes()
        except OSError as exc:
            raise NativeKneeCacheError("failed to read manifest") from exc
        actual_manifest_sha = _sha256_bytes(raw_manifest)
        if actual_manifest_sha != manifest_sha256:
            raise NativeKneeCacheError("manifest bytes SHA256 does not match")

        specs = self._parse_and_validate_manifest(raw_manifest, allow_synthetic)
        self._specs: Tuple[_SampleSpec, ...] = tuple(
            spec for spec in specs if spec.split == split
        )

        self._evidence_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._evidence_signature: Dict[
            Tuple[str, str], Tuple[int, int, int, int, int]
        ] = {}
        for spec in self._specs:
            self._verify_case_evidence(spec, cache_metadata=True)

        selected_synthetic = any(
            spec.source_mode == "simulated_test_double" or spec.test_double_used
            for spec in self._specs
        )
        manifest_synthetic = any(
            spec.source_mode == "simulated_test_double" or spec.test_double_used
            for spec in specs
        )
        self.contains_synthetic = selected_synthetic
        self.manifest_contains_synthetic = manifest_synthetic

    def _parse_and_validate_manifest(
        self, raw_manifest: bytes, allow_synthetic: bool
    ) -> List[_SampleSpec]:
        try:
            text = raw_manifest.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NativeKneeCacheError("manifest is not UTF-8") from exc

        lines = text.splitlines()
        if not lines:
            raise NativeKneeCacheError("manifest is empty")

        parsed: List[Tuple[int, Dict[str, Any]]] = []
        for lineno, line in enumerate(lines, 1):
            if not line.strip():
                raise NativeKneeCacheError(f"manifest line {lineno}: empty lines are not allowed")
            try:
                obj = json.loads(
                    line, object_pairs_hook=_object_without_duplicate_keys
                )
            except _DuplicateJsonKey as exc:
                raise NativeKneeCacheError(
                    f"manifest line {lineno}: JSON object has duplicate keys"
                ) from exc
            except json.JSONDecodeError as exc:
                raise NativeKneeCacheError(f"manifest line {lineno}: JSON parse error") from exc
            if not isinstance(obj, dict):
                raise NativeKneeCacheError(f"manifest line {lineno}: each line must be a JSON object")

            # Read only the split first: test rows are rejected before any label field is touched.
            if "split" not in obj:
                raise NativeKneeCacheError(f"manifest line {lineno}: missing split")
            row_split = obj["split"]
            if row_split == "test":
                raise NativeKneeCacheError(
                    f"manifest line {lineno}: test split found; refusing to load (labels not parsed)"
                )
            if row_split not in SUPPORTED_SPLITS:
                raise NativeKneeCacheError(f"manifest line {lineno}: split must be train/val")
            parsed.append((lineno, obj))

        specs = [self._validate_row(lineno, obj) for lineno, obj in parsed]
        self._validate_manifest_consistency(specs)

        if not allow_synthetic:
            for spec in specs:
                if spec.test_double_used or spec.source_mode != "native_dicom":
                    raise NativeKneeCacheError(
                        f"manifest line {spec.line}: synthetic/test-double rows are rejected outside test mode"
                    )
        return specs

    def _validate_row(self, lineno: int, obj: Mapping[str, Any]) -> _SampleSpec:
        where = f"manifest line {lineno}"
        if obj.get("schema") != SCHEMA:
            raise NativeKneeCacheError(f"{where}: schema mismatch")
        package = obj.get("package")
        if not isinstance(package, str) or not package:
            raise NativeKneeCacheError(f"{where}: package must be a non-empty string (any value)")

        film_id = _require_str(obj.get("film_id"), where, "film_id")
        patient_id = _require_str(obj.get("patient_id"), where, "patient_id")
        visit = _require_str(obj.get("visit"), where, "visit")
        if visit not in VISITS:
            raise NativeKneeCacheError(f"{where}: visit not in the frozen value set")
        split = obj.get("split")
        if split not in SUPPORTED_SPLITS:
            raise NativeKneeCacheError(f"{where}: split must be train/val")
        modality = _require_str(obj.get("modality"), where, "modality")
        if modality not in MODALITIES:
            raise NativeKneeCacheError(f"{where}: modality not in the frozen value set")
        image_position = _require_str(obj.get("image_position"), where, "image_position")
        if image_position not in IMAGE_POSITIONS:
            raise NativeKneeCacheError(f"{where}: image_position not in the frozen value set")
        candidate_id = _require_str(obj.get("candidate_id"), where, "candidate_id")
        if candidate_id != f"{film_id}::{image_position}":
            raise NativeKneeCacheError(f"{where}: candidate_id must equal film_id+'::'+image_position")

        kl = obj.get("kl")
        if not _is_int(kl) or not 0 <= int(kl) <= 4:
            raise NativeKneeCacheError(f"{where}: kl must be an integer in 0..4")
        clinical_side = _require_str(obj.get("clinical_side"), where, "clinical_side")
        if clinical_side not in CLINICAL_SIDES:
            raise NativeKneeCacheError(f"{where}: clinical_side not in the frozen value set")
        label_policy = _require_str(obj.get("label_policy"), where, "label_policy")
        if label_policy not in LABEL_POLICIES:
            raise NativeKneeCacheError(f"{where}: label_policy not in the frozen value set")
        if clinical_side == "UNKNOWN" and label_policy != "equal_kl_no_side_inference":
            raise NativeKneeCacheError(
                f"{where}: UNKNOWN side allows only equal_kl_no_side_inference; side must not be inferred"
            )
        if label_policy == "equal_kl_no_side_inference" and clinical_side != "UNKNOWN":
            raise NativeKneeCacheError(
                f"{where}: equal_kl_no_side_inference requires clinical_side=UNKNOWN, "
                f"got {clinical_side}"
            )
        if label_policy == "header_mapped" and clinical_side not in ("RIGHT", "LEFT"):
            raise NativeKneeCacheError(
                f"{where}: header_mapped requires clinical_side RIGHT or LEFT, "
                f"not {clinical_side}/UNKNOWN"
            )

        _require_bool(obj.get("label_eligible"), where, "label_eligible", True)
        _require_bool(obj.get("technical_readable"), where, "technical_readable", True)
        _require_bool(obj.get("localization_validated"), where, "localization_validated", False)
        _require_bool(obj.get("training_ready"), where, "training_ready", False)

        shape = obj.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(not _is_int(v) for v in shape)
            or tuple(int(v) for v in shape) != CACHE_SHAPE
        ):
            raise NativeKneeCacheError(f"{where}: shape must be [1,448,448]")

        gray_sha = _require_sha256(obj.get("gray01_sha256"), where, "gray01_sha256")
        mask_sha = _require_sha256(obj.get("valid_mask_sha256"), where, "valid_mask_sha256")
        evidence_sha = _require_sha256(
            obj.get("case_evidence_sha256"), where, "case_evidence_sha256"
        )
        gray_path = self._resolve_cache_rel(obj.get("gray01_npy"), where, "gray01_npy")
        mask_path = self._resolve_cache_rel(obj.get("valid_mask_npy"), where, "valid_mask_npy")
        evidence_path = self._resolve_cache_rel(
            obj.get("case_evidence_path"), where, "case_evidence_path"
        )

        bbox = obj.get("bbox_xyxy_exclusive")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not _is_int(v) for v in bbox)
        ):
            raise NativeKneeCacheError(f"{where}: bbox_xyxy_exclusive must be 4 integers")
        bbox_tuple = tuple(int(v) for v in bbox)

        native_source = obj.get("native_source")
        if not isinstance(native_source, dict):
            raise NativeKneeCacheError(f"{where}: native_source must be an object")
        _require_sha256(native_source.get("archive_sha256"), where, "native_source.archive_sha256")
        _require_sha256(
            native_source.get("input_evidence_sha256"),
            where,
            "native_source.input_evidence_sha256",
        )
        _require_sha256(
            native_source.get("native_pixel_sha256"),
            where,
            "native_source.native_pixel_sha256",
        )
        native_shape = native_source.get("source_native_shape")
        if (
            not isinstance(native_shape, list)
            or len(native_shape) != 2
            or any(not _is_int(v) or int(v) <= 0 for v in native_shape)
        ):
            raise NativeKneeCacheError(f"{where}: source_native_shape must be two positive integers")
        source_h, source_w = int(native_shape[0]), int(native_shape[1])
        x0, y0, x1, y1 = bbox_tuple
        if not (0 <= x0 < x1 <= source_w and 0 <= y0 < y1 <= source_h):
            raise NativeKneeCacheError(f"{where}: bbox must lie inside source_native_shape")

        transform = obj.get("transform")
        if not isinstance(transform, dict):
            raise NativeKneeCacheError(f"{where}: transform must be an object")
        transform_bbox = transform.get("source_bbox_xyxy_exclusive")
        if (
            not isinstance(transform_bbox, list)
            or len(transform_bbox) != 4
            or any(not _is_int(v) for v in transform_bbox)
            or tuple(int(v) for v in transform_bbox) != bbox_tuple
        ):
            raise NativeKneeCacheError(
                f"{where}: transform.source_bbox_xyxy_exclusive must equal bbox exactly"
            )

        source_mode = _require_str(obj.get("source_mode"), where, "source_mode")
        if source_mode not in SOURCE_MODES:
            raise NativeKneeCacheError(f"{where}: source_mode not in the frozen value set")
        test_double_used = _require_bool(
            obj.get("test_double_used"), where, "test_double_used"
        )
        if source_mode == "simulated_test_double" and not test_double_used:
            raise NativeKneeCacheError(
                f"{where}: simulated_test_double must set test_double_used=true"
            )
        if source_mode == "native_dicom" and test_double_used:
            raise NativeKneeCacheError(
                f"{where}: native_dicom must not set test_double_used=true"
            )

        return _SampleSpec(
            line=lineno,
            candidate_id=candidate_id,
            film_id=film_id,
            patient_id=patient_id,
            visit=visit,
            split=split,
            modality=modality,
            kl=int(kl),
            image_position=image_position,
            clinical_side=clinical_side,
            label_policy=label_policy,
            bbox=bbox_tuple,
            transform=copy.deepcopy(dict(transform)),
            gray_path=gray_path,
            gray_sha256=gray_sha,
            mask_path=mask_path,
            mask_sha256=mask_sha,
            evidence_path=evidence_path,
            evidence_sha256=evidence_sha,
            native_source=copy.deepcopy(dict(native_source)),
            source_mode=source_mode,
            test_double_used=test_double_used,
        )

    def _resolve_cache_rel(self, rel: Any, where: str, field: str) -> Path:
        if not isinstance(rel, str) or not rel:
            raise NativeKneeCacheError(f"{where}: {field} must be a path relative to cache_root")
        if "\\" in rel:
            raise NativeKneeCacheError(f"{where}: {field} must be a POSIX-style relative path")
        pure = PurePosixPath(rel)
        if pure.is_absolute():
            raise NativeKneeCacheError(f"{where}: {field} must not be an absolute path")
        if any(part == ".." for part in pure.parts):
            raise NativeKneeCacheError(f"{where}: {field} must not contain '..'")
        try:
            resolved = (self.cache_root / Path(*pure.parts)).resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise NativeKneeCacheError(f"{where}: {field} does not exist or cannot be resolved") from exc
        if not resolved.is_relative_to(self.cache_root):
            raise NativeKneeCacheError(f"{where}: {field} escapes cache_root after resolution (symlink)")
        if not resolved.is_file():
            raise NativeKneeCacheError(f"{where}: {field} must be a regular file")
        return resolved

    def _validate_manifest_consistency(self, specs: Sequence[_SampleSpec]) -> None:
        seen_candidates: Dict[str, int] = {}
        seen_film_positions: Dict[Tuple[str, str], int] = {}
        patient_splits: Dict[str, set] = {}
        film_identity: Dict[str, Tuple[Any, ...]] = {}
        patient_visit_films: Dict[Tuple[str, str], str] = {}
        film_candidates: Dict[str, List[_SampleSpec]] = {}
        archive_films: Dict[str, str] = {}

        for spec in specs:
            if spec.candidate_id in seen_candidates:
                raise NativeKneeCacheError(
                    f"manifest line {spec.line}: duplicate candidate_id"
                )
            seen_candidates[spec.candidate_id] = spec.line

            film_position = (spec.film_id, spec.image_position)
            if film_position in seen_film_positions:
                raise NativeKneeCacheError(
                    f"manifest line {spec.line}: duplicate (film_id, image_position)"
                )
            seen_film_positions[film_position] = spec.line

            identity = (
                spec.patient_id,
                spec.visit,
                spec.split,
                spec.modality,
                spec.label_policy,
                spec.source_mode,
                self._source_fingerprint(spec.native_source),
            )
            previous_identity = film_identity.get(spec.film_id)
            if previous_identity is None:
                film_identity[spec.film_id] = identity
            elif previous_identity != identity:
                raise NativeKneeCacheError(
                    f"manifest line {spec.line}: within one film the "
                    "patient/visit/split/modality/label_policy/source differ"
                )

            patient_visit = (spec.patient_id, spec.visit)
            previous_film = patient_visit_films.get(patient_visit)
            if previous_film is None:
                patient_visit_films[patient_visit] = spec.film_id
            elif previous_film != spec.film_id:
                raise NativeKneeCacheError(
                    f"manifest line {spec.line}: one patient+visit maps to several films"
                )

            patient_splits.setdefault(spec.patient_id, set()).add(spec.split)
            film_candidates.setdefault(spec.film_id, []).append(spec)

        for splits in patient_splits.values():
            if len(splits) > 1:
                raise NativeKneeCacheError("a patient appears in both train and val; refusing to load")

        for film_id, candidates in film_candidates.items():
            if len(candidates) < 2:
                continue
            if len(candidates) > 2:  # pragma: no cover
                raise NativeKneeCacheError(
                    f"film {film_id}: more than 2 candidates; refusing to load"
                )
            policy = candidates[0].label_policy
            if policy == "header_mapped":
                sides = {spec.clinical_side for spec in candidates}
                if len(sides) != 2:
                    raise NativeKneeCacheError(
                        f"film {film_id}: the two header_mapped candidates must have different clinical sides"
                    )
            elif policy == "equal_kl_no_side_inference":
                kls = {spec.kl for spec in candidates}
                if len(kls) != 1:
                    raise NativeKneeCacheError(
                        f"film {film_id}: the two equal_kl_no_side_inference candidates must have the same KL"
                    )

        for spec in specs:
            archive_sha = str(spec.native_source["archive_sha256"])
            previous_film = archive_films.get(archive_sha)
            if previous_film is None:
                archive_films[archive_sha] = spec.film_id
            elif previous_film != spec.film_id:
                raise NativeKneeCacheError(
                    f"manifest line {spec.line}: one archive_sha256 must not belong to different films"
                )

    @staticmethod
    def _source_fingerprint(native_source: Mapping[str, Any]) -> Tuple[Any, ...]:
        native_shape = native_source["source_native_shape"]
        return (
            native_source["archive_sha256"],
            native_source["input_evidence_sha256"],
            int(native_shape[0]),
            int(native_shape[1]),
        )

    @staticmethod
    def _evidence_key(spec: _SampleSpec) -> Tuple[str, str]:
        return (str(spec.evidence_path), spec.evidence_sha256)

    def _verify_case_evidence(
        self, spec: _SampleSpec, *, cache_metadata: bool
    ) -> Dict[str, Any]:
        key = self._evidence_key(spec)
        try:
            stat = spec.evidence_path.stat()
        except OSError as exc:
            raise NativeKneeCacheError(
                f"manifest line {spec.line}: case_evidence_path is not accessible"
            ) from exc
        signature = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
            int(stat.st_ino),
            int(stat.st_dev),
        )
        cached = self._evidence_meta.get(key)
        if cached is not None and self._evidence_signature.get(key) == signature:
            return cached

        try:
            raw = spec.evidence_path.read_bytes()
        except OSError as exc:
            raise NativeKneeCacheError(
                f"manifest line {spec.line}: failed to read case_evidence_path"
            ) from exc
        if _sha256_bytes(raw) != spec.evidence_sha256:
            raise NativeKneeCacheError(
                f"manifest line {spec.line}: case evidence SHA256 mismatch"
            )

        parsed: Any = None
        try:
            candidate = json.loads(raw.decode("utf-8"))
            if isinstance(candidate, dict):
                parsed = candidate
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = None
        meta: Dict[str, Any] = {
            "sha256": spec.evidence_sha256,
            "size": len(raw),
            "json": parsed,
        }
        if cache_metadata:
            self._evidence_meta[key] = meta
            self._evidence_signature[key] = signature
        return meta

    def _read_verified_bytes(
        self, path: Path, expected_sha256: str, where: str, field: str
    ) -> bytes:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise NativeKneeCacheError(f"{where}: {field} read failed") from exc
        # Hash exactly the bytes later passed to np.load (avoids TOCTOU).
        if _sha256_bytes(raw) != expected_sha256:
            raise NativeKneeCacheError(f"{where}: {field} bytes SHA256 mismatch")
        return raw

    @staticmethod
    def _decode_npy(raw: bytes, where: str, field: str) -> np.ndarray:
        try:
            array = np.load(io.BytesIO(raw), allow_pickle=False)
        except Exception as exc:  # np.load rejects object/pickle arrays here.
            raise NativeKneeCacheError(
                f"{where}: {field} cannot be decoded with allow_pickle=False"
            ) from exc
        if not isinstance(array, np.ndarray):
            raise NativeKneeCacheError(f"{where}: {field} is not a numpy array")
        return array

    @staticmethod
    def _validate_cache_arrays(
        gray: np.ndarray, mask: np.ndarray, where: str
    ) -> None:
        if gray.dtype != np.float32:
            raise NativeKneeCacheError(f"{where}: gray01_npy dtype must be float32")
        if mask.dtype != np.bool_:
            raise NativeKneeCacheError(f"{where}: valid_mask_npy dtype must be bool")
        if gray.shape != CACHE_SHAPE or mask.shape != CACHE_SHAPE:
            raise NativeKneeCacheError(f"{where}: gray/mask shape must be (1,448,448)")
        if not np.isfinite(gray).all():
            raise NativeKneeCacheError(f"{where}: gray has non-finite values")
        if float(gray.min()) < 0.0 or float(gray.max()) > 1.0:
            raise NativeKneeCacheError(f"{where}: gray must be in [0,1]")
        if not bool(mask.any()):
            raise NativeKneeCacheError(f"{where}: valid_mask needs at least one valid pixel")
        # Padding must be exactly 0; never repair a bad cache.
        if bool(np.any(gray[~mask] != 0.0)):
            raise NativeKneeCacheError(f"{where}: padding must be 0")
        valid_values = gray[mask]
        std = float(np.std(valid_values.astype(np.float64), ddof=0))
        if not np.isfinite(std) or std < 0.01:
            raise NativeKneeCacheError(f"{where}: gray std in the valid region must be >= 0.01")

    def _render_image(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
        where: str,
        cache_transform: Mapping[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        try:
            image, out_mask, transform = prepare_model_input(
                gray,
                mask,
                image_size=self.image_size,
                cache_transform=cache_transform,
            )
        except ModelInputPreprocessingError as exc:
            raise NativeKneeCacheError(f"{where}: {exc}") from exc
        return image, out_mask, transform

    def _load_sample(self, index: int) -> Tuple[Dict[str, Any], int]:
        spec = self._specs[index]
        where = f"manifest line {spec.line}"

        self._verify_case_evidence(spec, cache_metadata=True)

        gray_raw = self._read_verified_bytes(
            spec.gray_path, spec.gray_sha256, where, "gray01_npy"
        )
        mask_raw = self._read_verified_bytes(
            spec.mask_path, spec.mask_sha256, where, "valid_mask_npy"
        )
        gray = self._decode_npy(gray_raw, where, "gray01_npy")
        mask = self._decode_npy(mask_raw, where, "valid_mask_npy")
        self._validate_cache_arrays(gray, mask, where)
        image, _, transform = self._render_image(
            gray, mask, where, spec.transform
        )

        sample: Dict[str, Any] = {
            "image": image,
            "kl": np.int64(spec.kl),
            "candidate_id": spec.candidate_id,
            "film_id": spec.film_id,
            "patient_id": spec.patient_id,
            "visit": spec.visit,
            "split": spec.split,
            "modality": spec.modality,
            "image_position": spec.image_position,
            "clinical_side": spec.clinical_side,
            "label_policy": spec.label_policy,
            "bbox": [int(v) for v in spec.bbox],
            # cache_transform is in 448-cache space; composition is only the 448 -> model resize, not native or 224 coordinates.
            "transform": transform,
            "label_eligible": True,
            "technical_readable": True,
            "localization_validated": False,
            "training_ready": False,
            "source_mode": spec.source_mode,
            "test_double_used": spec.test_double_used,
            "case_evidence_sha256": spec.evidence_sha256,
            "clinical_validation": False,
        }
        return sample, len(gray_raw) + len(mask_raw)

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not _is_int(index):
            raise TypeError("index must be an integer")
        normalized = int(index)
        if normalized < 0:
            normalized += len(self._specs)
        if normalized < 0 or normalized >= len(self._specs):
            raise IndexError("sample index out of range")
        sample, _ = self._load_sample(normalized)
        return sample

    def identity(self) -> Dict[str, Any]:
        return {
            "module": "downstream.oa_kl_unified_v1.data",
            "schema": SCHEMA,
            "split": self.split,
            "image_size": self.image_size,
            "manifest_sha256": self.manifest_sha256,
            "num_samples": len(self._specs),
            "allow_synthetic": self.allow_synthetic,
            "contains_synthetic": self.contains_synthetic,
            "manifest_contains_synthetic": self.manifest_contains_synthetic,
            "clinical_validation": False,
        }

    def aggregate(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "split": self.split,
            "image_size": self.image_size,
            "num_samples": len(self._specs),
            "num_patients": len({spec.patient_id for spec in self._specs}),
            "num_films": len({spec.film_id for spec in self._specs}),
            "num_synthetic": sum(
                1
                for spec in self._specs
                if spec.source_mode == "simulated_test_double" or spec.test_double_used
            ),
            "allow_synthetic": self.allow_synthetic,
            "clinical_validation": False,
        }


def probe_io(
    dataset: NativeKneeDataset,
    *,
    passes: int = 2,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    if not isinstance(dataset, NativeKneeDataset):
        raise TypeError("probe_io needs a NativeKneeDataset")
    if not _is_int(passes) or int(passes) < 1:
        raise ValueError("passes must be a positive integer")
    if limit is not None and (not _is_int(limit) or int(limit) < 1):
        raise ValueError("limit must be None or a positive integer")

    passes_int = int(passes)
    total_available = len(dataset)
    sample_count = total_available if limit is None else min(int(limit), total_available)

    records: List[Dict[str, Any]] = []
    for pass_index in range(1, passes_int + 1):
        started = time.perf_counter()
        bytes_read = 0
        processed = 0
        for sample_index in range(sample_count):
            _, sample_bytes = dataset._load_sample(sample_index)
            bytes_read += int(sample_bytes)
            processed += 1
        elapsed = time.perf_counter() - started

        if processed:
            knees_per_second = processed / elapsed if elapsed > 0 else None
            mean_ms_per_knee = elapsed * 1000.0 / processed
        else:
            knees_per_second = None
            mean_ms_per_knee = None

        records.append(
            {
                "pass": pass_index,
                "phase": "initial_read" if pass_index == 1 else "repeat_read",
                "samples": processed,
                "total_bytes": bytes_read,
                "seconds": elapsed,
                "knees_per_second": knees_per_second,
                "mean_ms_per_knee": mean_ms_per_knee,
                "cache_state": "uncontrolled_os_page_cache",
                "hot_cache_guaranteed": False,
            }
        )

    return {
        "dataset": {
            "split": dataset.split,
            "image_size": dataset.image_size,
            "num_samples": total_available,
            "manifest_sha256": dataset.manifest_sha256,
            "allow_synthetic": dataset.allow_synthetic,
            "contains_synthetic": dataset.contains_synthetic,
            "clinical_validation": False,
        },
        "limit": None if limit is None else int(limit),
        "passes": records,
        "cache_state": "uncontrolled_os_page_cache",
        "cold_io_claimed": False,
        "note": "The first pass is not guaranteed to be cold I/O; later passes are repeat_read only, not guaranteed warm cache.",
    }
