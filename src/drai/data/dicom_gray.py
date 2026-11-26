import gzip
import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut

POLICY_ID = "pa-dicom-presentation-roi-percentile-v1"
NAS_ROOT = Path("/path/to/nas")
MAX_MEMBER_BYTES = 128 * 1024**2
MAX_EXPANDED_BYTES = 512 * 1024**2


def array_sha256(array):
    digest = hashlib.sha256(f"{array.dtype}|{array.shape}|".encode())
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


class _HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size=-1):
        chunk = self.stream.read(size)
        self.digest.update(chunk)
        self.count += len(chunk)
        return chunk


class _BoundedExpanded(io.RawIOBase):
    def __init__(self, stream):
        self.stream = stream
        self.count = 0

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            size = min(1024**2, MAX_EXPANDED_BYTES - self.count + 1)
        data = self.stream.read(size)
        self.count += len(data)
        if self.count > MAX_EXPANDED_BYTES:
            raise ValueError("archive expands beyond the 512 MiB limit; needs a separate audit")
        return data


def _decode_single(ds):
    if str(ds.PhotometricInterpretation) not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError("not a supported single-channel grayscale DICOM")
    if int(ds.SamplesPerPixel) != 1 or int(getattr(ds, "NumberOfFrames", 1)) != 1:
        raise ValueError("the current PA input needs a single frame and single channel")
    pixels = ds.pixel_array
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.ndim != 2 or pixels.shape != (int(ds.Rows), int(ds.Columns)):
        raise ValueError("decoded shape differs from DICOM Rows/Columns")
    if pixels.dtype.kind not in {"i", "u"}:
        raise ValueError("the current PA policy handles native integer gray only")
    bits = int(ds.BitsStored)
    signed = int(ds.PixelRepresentation) == 1
    if not 1 <= bits <= int(ds.BitsAllocated) <= 32:
        raise ValueError("unsupported pixel bit depth")
    lower = -(2 ** (bits - 1)) if signed else 0
    upper = 2 ** (bits - int(signed)) - 1
    if pixels.min() < lower or pixels.max() > upper:
        raise ValueError("decoded values exceed the BitsStored/PixelRepresentation range")
    return pixels


def read_pa_archive(source, expected_bytes=None, source_root=NAS_ROOT):
    source = Path(source).resolve()
    if not source.is_relative_to(Path(source_root).resolve()):
        raise ValueError("the source archive must be under the designated read-only root")
    before = source.stat()
    if expected_bytes is not None and before.st_size != int(expected_bytes):
        raise ValueError("source file size differs from the fixed list")
    image = None
    member_name = None
    nonimage_members = 0
    with source.open("rb") as raw:
        hashed = _HashReader(raw)
        with gzip.GzipFile(fileobj=hashed, mode="rb") as compressed:
            bounded = _BoundedExpanded(compressed)
            with tarfile.open(fileobj=bounded, mode="r|") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    if member.size > MAX_MEMBER_BYTES:
                        raise ValueError("tar member exceeds the 128 MiB limit")
                    with archive.extractfile(member) as handle:
                        try:
                            # pydicom seeks backwards while parsing the header; only the current size-bounded member is held in memory.
                            ds = pydicom.dcmread(io.BytesIO(handle.read()))
                        except pydicom.errors.InvalidDicomError:
                            nonimage_members += 1
                            continue
                    if "PixelData" not in ds:
                        nonimage_members += 1
                        continue
                    if image is not None:
                        raise ValueError("PA has several image members; an explicit multi-image policy is needed")
                    image = (ds, _decode_single(ds))
                    member_name = member.name
            # tar stops at the end-of-archive block; keep reading so trailing gzip damage or extra members are caught.
            while bounded.read(1024**2):
                pass
        if hashed.count != before.st_size:
            raise ValueError("compressed stream not fully read")
        archive_digest = hashed.digest.hexdigest()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source archive changed while being read")
    if image is None:
        raise ValueError("no decodable PA image found")
    ds, pixels = image
    return (
        ds,
        pixels,
        {
            "archive_sha256": archive_digest,
            "archive_bytes": before.st_size,
            "archive_mtime_ns": before.st_mtime_ns,
            "expanded_bytes": bounded.count,
            "dicom_member": member_name,
            "nonimage_members": nonimage_members,
            "gzip_crc_verified": True,
            "source_stat_unchanged": True,
            "native_pixel_sha256": array_sha256(pixels),
        },
    )


def valid_pixel_mask(pixels, ds):
    mask = np.ones(pixels.shape, dtype=bool)
    padding = getattr(ds, "PixelPaddingValue", None)
    if padding is not None:
        limit = getattr(ds, "PixelPaddingRangeLimit", padding)
        low, high = sorted((int(padding), int(limit)))
        mask &= (pixels < low) | (pixels > high)
    if not mask.any():
        raise ValueError("the whole image is padding")
    return mask


def presentation_values(pixels, ds):
    photo = str(getattr(ds, "PhotometricInterpretation", ""))
    if photo not in {"MONOCHROME1", "MONOCHROME2"}:
        raise ValueError("unsupported PhotometricInterpretation")
    valid = valid_pixel_mask(pixels, ds)
    # An explicitly empty rescale field is not an identity mapping; quarantine it instead of assuming 1/0.
    if "RescaleSlope" in ds or "RescaleIntercept" in ds:
        slope = getattr(ds, "RescaleSlope", None)
        intercept = getattr(ds, "RescaleIntercept", None)
        if slope is None or intercept is None:
            raise ValueError("RescaleSlope/Intercept explicitly empty or incomplete; quarantine for review")
        if not np.isfinite(float(slope)) or not np.isfinite(float(intercept)):
            raise ValueError("RescaleSlope/Intercept are not finite numbers")
        if float(slope) == 0:
            raise ValueError("RescaleSlope is zero; dynamic range degenerate")
    values = apply_modality_lut(pixels, ds)
    voi_luts = getattr(ds, "VOILUTSequence", [])
    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)
    if bool(center is not None) != bool(width is not None):
        raise ValueError("incomplete WindowCenter/WindowWidth")
    voi_method = "none"
    lut_info = None
    if voi_luts:
        if values.dtype.kind == "f":
            if not np.isfinite(values).all() or not np.array_equal(
                values, np.rint(values)
            ):
                raise ValueError("modality output for the VOI LUT has non-integers; needs a separate review")
            # Compare with a strict < 2**63: the int64 maximum overflows when cast to float64.
            if values.min() < -(2**63) or values.max() >= 2**63:
                raise ValueError("VOI LUT index exceeds the int64 range")
            values = values.astype(np.int64)
        item = voi_luts[0]
        values = apply_voi_lut(values, ds, index=0, prefer_lut=True)
        lut_info = {
            "descriptor": list(map(int, item.LUTDescriptor)),
            "sha256": hashlib.sha256(item.to_json().encode()).hexdigest(),
            "alternatives": len(voi_luts),
        }
        voi_method = "first_voi_lut"
    elif center is not None:
        values = apply_voi_lut(values, ds, index=0, prefer_lut=False)
        voi_method = "first_window"
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("gray conversion produced NaN/Inf")
    if getattr(ds, "PresentationLUTSequence", []):
        raise ValueError("explicit Presentation LUT Sequence not yet validated; needs separate handling")
    shape = str(getattr(ds, "PresentationLUTShape", "")).strip().upper()
    if shape not in {"", "IDENTITY", "INVERSE"}:
        raise ValueError("unsupported PresentationLUTShape")
    inverse = shape == "INVERSE" or (not shape and photo == "MONOCHROME1")
    # Polarity comes from Presentation LUT Shape; MONOCHROME1 is only the fallback.
    # PixelIntensityRelationshipSign is not applied, and no left-right flip is done.
    if inverse:
        values = values[valid].min() + values[valid].max() - values
    evidence = {
        "policy_id": POLICY_ID,
        "modality": str(getattr(ds, "Modality", "")),
        "photometric_interpretation": photo,
        "modality_lut_present": bool(getattr(ds, "ModalityLUTSequence", [])),
        "rescale_slope": float(getattr(ds, "RescaleSlope", 1)),
        "rescale_intercept": float(getattr(ds, "RescaleIntercept", 0)),
        "voi_method": voi_method,
        "voi_index": 0 if voi_method != "none" else None,
        "voi_lut": lut_info,
        "presentation_lut_shape": shape or None,
        "presentation_inverse_applied": inverse,
        "intensity_relationship_sign_used_to_invert": False,
        "window_center": str(center) if center is not None else None,
        "window_width": str(width) if width is not None else None,
        "input_policy_status": "engineering_validated_clinical_effect_pending",
    }
    return values, valid, evidence


def normalize_roi(values, valid, percentiles=(0.5, 99.5)):
    if values.ndim != 2 or values.shape != valid.shape or valid.dtype != bool:
        raise ValueError("gray and valid mask need the same 2-D shape")
    if not valid.any() or not np.isfinite(values).all():
        raise ValueError("ROI has no valid values or contains NaN/Inf")
    if not 0 <= percentiles[0] < percentiles[1] <= 100:
        raise ValueError("invalid percentile range")
    low, high = map(float, np.percentile(values[valid], percentiles))
    if high <= low:
        raise ValueError("ROI gray dynamic range is degenerate")
    output = np.clip((values - low) / (high - low), 0, 1).astype(np.float32)
    output[~valid] = 0
    return output, {
        "percentiles": list(percentiles),
        "low": low,
        "high": high,
        "valid_fraction": float(valid.mean()),
        "lower_clipped_fraction": float((values[valid] < low).mean()),
        "upper_clipped_fraction": float((values[valid] > high).mean()),
        "gray01_mean_valid": float(output[valid].mean()),
        "gray01_std_valid": float(output[valid].std()),
        "shape": list(output.shape),
        "dtype": str(output.dtype),
    }
