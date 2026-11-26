from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import sys
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from time import monotonic

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from drai.data.dicom_gray import normalize_roi, presentation_values, read_pa_archive
from drai.data.full_pa import FULL_PA_POLICY, letterbox
from drai.data.preprocess_io import (
    atomic_json,
    file_sha256,
    fsync_directory,
    project_path,
    read_sealed_record,
    sealed_record,
)

TASKS = (
    "baseline_max_kl5",
    "48m_any_compartment_jsn_ge1",
    "baseline_max_womac_pain_regression",
)
POLICY = "downstream-v00-fullfilm-technical-qc-v1"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    fsync_directory(path.parent)


def save_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    restored = np.load(path, allow_pickle=False)
    if restored.dtype != value.dtype or not np.array_equal(restored, value):
        raise ValueError("cached array read-back mismatch")


def specimen_key(patient: str) -> str:
    return hashlib.sha256((POLICY + "|" + patient).encode()).hexdigest()[:20]


def header_checks(ds, source: dict, patient: str) -> tuple[dict, list[str]]:
    warnings = []
    observed_id = str(getattr(ds, "PatientID", "")).strip()
    if observed_id and observed_id != patient:
        raise ValueError("DICOM PatientID does not match the fixed candidate patient")
    if not observed_id:
        warnings.append("patient_id_header_missing")
    raw_date = str(getattr(ds, "StudyDate", "")).strip()
    expected_date = source["exam_date_iso"]
    if raw_date:
        # DICOM DA holds a calendar date only; no time zone is added and incomplete dates are rejected.
        if len(raw_date) != 8 or not raw_date.isdigit():
            raise ValueError("DICOM StudyDate is not a valid YYYYMMDD date")
        parsed = date(
            int(raw_date[:4]), int(raw_date[4:6]), int(raw_date[6:8])
        ).isoformat()
        if parsed != expected_date:
            raise ValueError("DICOM StudyDate does not match the baseline candidate date")
    else:
        warnings.append("study_date_header_missing")
    if str(ds.Modality) != source["dicom_modality_first_instance"]:
        raise ValueError("DICOM acquisition code does not match the candidate list")
    for field, keyword in (
        ("Rows", "dicom_first_instance_rows"),
        ("Columns", "dicom_first_instance_columns"),
    ):
        if source.get(keyword) and int(getattr(ds, field)) != int(source[keyword]):
            raise ValueError("DICOM pixel size does not match the original header list")
    view = str(getattr(ds, "ViewPosition", "")).strip().upper()
    if view != "PA":
        warnings.append(
            "view_position_missing" if not view else "view_position_conflict"
        )
    laterality = str(getattr(ds, "ImageLaterality", "")).strip()
    if laterality in ("R", "L"):
        warnings.append("bilateral_candidate_has_unilateral_header")
    ratio = int(ds.Columns) / int(ds.Rows)
    if ratio < 0.75 or ratio > 2.5:
        warnings.append("unusual_native_aspect_ratio_requires_visual_review")
    return {
        "patient_id_matches": bool(observed_id),
        "study_date_matches": bool(raw_date),
        "view_position": view or None,
        "image_laterality": laterality or None,
        "native_rows": int(ds.Rows),
        "native_columns": int(ds.Columns),
        "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "clinical_side": "UNKNOWN",
    }, warnings


def prepare_one(task: tuple) -> dict:
    patient, source, output, retry = task
    torch.set_num_threads(1)
    output = Path(output)
    key = specimen_key(patient)
    folder = output / "patients" / key
    if folder.exists():
        previous = read_sealed_record(folder / "completion.json")
        if previous["status"] == "passed":
            row = previous["row"]
            for file_key, hash_key in (
                ("gray01_npy", "gray01_sha256"),
                ("valid_mask_npy", "valid_mask_sha256"),
                ("input_evidence_json", "input_evidence_sha256"),
            ):
                if file_sha256(row[file_key]) != row[hash_key]:
                    raise ValueError("published cache fingerprint changed on resume; repair it separately instead of silently reusing it")
            evidence = json.loads(Path(row["input_evidence_json"]).read_text())
            stat = Path(source["archive_server_path"]).stat()
            original = evidence["source_evidence"]
            if (stat.st_size, stat.st_mtime_ns) != (
                original["archive_bytes"],
                original["archive_mtime_ns"],
            ):
                raise ValueError("original source file stat changed on resume; needs a separate source audit")
            return {**previous, "reused": True}
        if not retry:
            return {**previous, "reused": True}
        history = output / "failure_history"
        history.mkdir(exist_ok=True)
        os.replace(folder, history / f"{key}_{uuid.uuid4().hex}")
    stage = folder.with_name(f".{key}.{uuid.uuid4().hex}.tmp")
    stage.mkdir()
    row, source_evidence, header, warnings = None, None, None, []
    failure = None
    started = monotonic()
    try:
        ds, native, source_evidence = read_pa_archive(
            source["archive_server_path"], source["archive_bytes"]
        )
        header, warnings = header_checks(ds, source, patient)
        values, valid, display = presentation_values(native, ds)
        # Percentiles are taken over the whole film; the field of view is not cropped, as for the frozen SSL input.
        gray, statistics = normalize_roi(values, valid)
        cached, cached_mask, box = letterbox(gray, valid, 512)
        model_input, model_mask, _ = letterbox(cached, cached_mask, 448)
        if (
            not cached_mask.any()
            or cached[cached_mask].std() < 0.01
            or not model_mask.any()
            or not np.isfinite(model_input).all()
        ):
            raise ValueError("gray degenerate or non-finite after normalization/resizing")
        save_array(stage / "gray01.npy", cached)
        save_array(stage / "valid_mask.npy", cached_mask)
        evidence = {
            "qc_policy_id": POLICY,
            "input_policy_id": FULL_PA_POLICY,
            "input_policy_status": "engineering_validated",
            "archive_relpath": source["archive_relpath"],
            "source_evidence": source_evidence,
            "header_checks": header,
            "warnings": warnings,
            "dicom_display_policy": display,
            "full_image_normalization": statistics,
            "source_native_shape": list(native.shape),
            "cached_shape": list(cached.shape),
            "model_shape": [1, 448, 448],
            "letterbox_xyxy_exclusive": box,
            "all_source_field_retained": True,
            "native_field_retained_not_anatomical_coverage_proof": True,
            "anatomical_coverage_review": "pending",
            "clinical_quality_approved": False,
            "disease_preservation_validated": False,
            "nas_source_modified": False,
        }
        atomic_json(stage / "input_evidence.json", evidence)
        row = {
            "sample_id": key,
            "patient_id": patient,
            "visit": "V00",
            "split": source["split"],
            "modality": str(ds.Modality),
            "input_archive_relpath": source["archive_relpath"],
            "gray01_npy": str(folder / "gray01.npy"),
            "gray01_sha256": file_sha256(stage / "gray01.npy"),
            "valid_mask_npy": str(folder / "valid_mask.npy"),
            "valid_mask_sha256": file_sha256(stage / "valid_mask.npy"),
            "input_evidence_json": str(folder / "input_evidence.json"),
            "input_evidence_sha256": file_sha256(stage / "input_evidence.json"),
            "input_policy_id": FULL_PA_POLICY,
            "technical_qc_passed": "1",
            "anatomical_coverage_review": "pending",
            "clinical_quality_approved": "0",
            "train_ready": "0",
            "warnings_json": json.dumps(warnings),
            "native_pixel_sha256": source_evidence["native_pixel_sha256"],
            "archive_sha256": source_evidence["archive_sha256"],
        }
    except Exception as error:  # noqa: BLE001
        failure = {"type": type(error).__name__, "message": str(error)}
    payload = {
        "patient_id": patient,
        "sample_id": key,
        "split": source["split"],
        "modality": source["dicom_modality_first_instance"],
        "archive_relpath": source["archive_relpath"],
        "status": "passed" if row else "failed_technical_qc",
        "decode_crc_passed": bool(source_evidence),
        "header": header,
        "warnings": warnings,
        "source_evidence": source_evidence,
        "row": row,
        "error": failure,
        "seconds": monotonic() - started,
    }
    atomic_json(stage / "completion.json", sealed_record(payload))
    fsync_directory(stage)
    os.replace(stage, folder)
    fsync_directory(folder.parent)
    return {**payload, "reused": False}


def make_visual_sample(cohort: list[dict], sources: dict) -> list[str]:
    groups = defaultdict(list)
    for row in cohort:
        source = sources[row["input_archive_relpath"]]
        groups[(source["dicom_modality_first_instance"], row["split"])].append(row)
    chosen = []
    for group in sorted(groups):
        rows = sorted(
            groups[group],
            key=lambda row: hashlib.sha256(
                ("baseline-coverage-sample-v1|" + row["patient_id"]).encode()
            ).hexdigest(),
        )[:8]
        chosen.extend(row["patient_id"] for row in rows)
    return chosen


def render_visual_samples(chosen: list[str], results: dict, report_dir: Path) -> None:
    size, margin = 448, 32
    for page in range((len(chosen) + 23) // 24):
        page_ids = chosen[page * 24 : (page + 1) * 24]
        sheet = Image.new("RGB", (4 * size, 6 * (size + margin)), "#202020")
        draw = ImageDraw.Draw(sheet)
        for index, patient in enumerate(page_ids):
            item = results.get(patient)
            x, y = index % 4 * size, index // 4 * (size + margin)
            title = f"QC {page * 24 + index + 1:02d}"
            if item:
                title += f" {item['modality']} {item['split']}"
                if item["status"] == "passed":
                    gray = np.load(item["row"]["gray01_npy"], allow_pickle=False)
                    display = Image.fromarray(
                        (np.clip(gray, 0, 1) * 255).astype(np.uint8)
                    )
                    display = display.resize((size, size), Image.Resampling.BILINEAR)
                    sheet.paste(display.convert("RGB"), (x, y))
                else:
                    draw.text(
                        (x + 10, y + size // 2), "TECHNICAL CHECK FAILED", fill="orange"
                    )
            else:
                draw.text((x + 10, y + size // 2), "NOT PROCESSED", fill="orange")
            draw.text((x + 8, y + size + 8), title, fill="white")
        sheet.save(report_dir / f"visual_sample_{page + 1:02d}.png")
    mapping = [
        {
            "preview_number": i + 1,
            "patient_id": patient,
            "sample_id": specimen_key(patient),
            "coverage_decision": "pending",
        }
        for i, patient in enumerate(chosen)
    ]
    write_csv(report_dir / "visual_review_sample.csv", mapping, list(mapping[0]))


def summarize(
    results: list[dict], all_count: int, output: Path, report_dir: Path, task_rows: dict
) -> dict:
    pixel_groups = defaultdict(list)
    for item in results:
        if item["status"] == "passed":
            pixel_groups[item["row"]["native_pixel_sha256"]].append(item)
    duplicates = [group for group in pixel_groups.values() if len(group) > 1]
    quarantined = {item["patient_id"] for group in duplicates for item in group}
    accepted = {
        item["patient_id"]: item["row"]
        for item in results
        if item["status"] == "passed" and item["patient_id"] not in quarantined
    }
    manifest_dir = ROOT / "data/manifests/downstream_qc_v1"
    if accepted:
        write_csv(
            manifest_dir / "baseline_input_technical_qc.csv",
            sorted(accepted.values(), key=lambda row: row["sample_id"]),
            list(next(iter(accepted.values()))),
        )
    task_summary = {}
    for task, candidates in task_rows.items():
        joined = []
        for candidate in candidates:
            image = accepted.get(candidate["patient_id"])
            if image:
                joined.append(
                    {**candidate, **image, "task": task, "label": candidate["label"]}
                )
        task_summary[task] = {}
        for split in ("train", "val", "test"):
            subset = [row for row in joined if row["split"] == split]
            counts = (
                Counter(row["label"] for row in subset) if task != TASKS[2] else None
            )
            task_summary[task][split] = {"total": len(subset)}
            if counts is not None:
                classes = range(5) if task == TASKS[0] else range(2)
                task_summary[task][split]["class_counts"] = {
                    str(k): counts[str(k)] for k in classes
                }
        if joined:
            write_csv(
                manifest_dir / f"{task}_technical_candidates.csv",
                joined,
                list(joined[0]),
            )
    summary = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "target_patients": all_count,
        "processed_patients": len(results),
        "all_candidates_complete": len(results) == all_count,
        "decode_crc_passed": sum(item["decode_crc_passed"] for item in results),
        "technical_passed_before_duplicate_quarantine": sum(
            item["status"] == "passed" for item in results
        ),
        "technical_accepted": len(accepted),
        "technical_failed": sum(item["status"] != "passed" for item in results),
        "failed_reasons": dict(
            Counter(item["error"]["message"] for item in results if item["error"])
        ),
        "failed_by_modality": dict(
            Counter(item["modality"] for item in results if item["status"] != "passed")
        ),
        "accepted_by_modality": dict(
            Counter(row["modality"] for row in accepted.values())
        ),
        "warning_counts": dict(
            Counter(warning for item in results for warning in item["warnings"])
        ),
        "duplicate_native_pixel_groups": len(duplicates),
        "quarantined_duplicate_patients": len(quarantined),
        "reused_completed_qc_patients": sum(item["reused"] for item in results),
        "task_counts": task_summary,
        "clinical_quality_approved": False,
        "anatomical_coverage_review": "pending",
        "training_started": False,
        "nas_source_modified": False,
        "output_directory": str(output),
        "manifest_directory": str(manifest_dir),
    }
    atomic_json(report_dir / "summary.json", summary)
    atomic_json(
        report_dir / "duplicate_quarantine.json",
        {
            "groups": [
                [
                    {"patient_id": item["patient_id"], "split": item["split"]}
                    for item in group
                ]
                for group in duplicates
            ],
            "rule": "same decoded native pixel fingerprint across distinct patients",
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--review-pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be in 1..8")
    output = project_path(ROOT, "data/processed/pa_baseline_qc_v1")
    report_dir = project_path(ROOT, "outputs/downstream/qc_baseline_v1")
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    (output / "patients").mkdir(exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        task_paths = {
            task: ROOT / f"data/manifests/downstream_v3/{task}_candidates.csv"
            for task in TASKS
        }
        task_rows = {task: read_csv(path) for task, path in task_paths.items()}
        cohort = task_rows[TASKS[0]]
        sources_path = ROOT / "data/manifests/pa_ssl_candidates_v1.csv"
        sources = {row["archive_relpath"]: row for row in read_csv(sources_path)}
        if len({row["patient_id"] for row in cohort}) != len(cohort):
            raise ValueError("baseline candidate patients are not unique")
        for row in cohort:
            source = sources[row["input_archive_relpath"]]
            if (
                row["input_visit"] != "V00"
                or source["visit"] != "V00"
                or row["split"] != source["split"]
                or row["patient_id"] != source["patient_id"]
                or not source["sequence_family"].startswith("Bilateral")
            ):
                raise ValueError("patient/visit/bilateral PA source does not match the fixed candidates")
        chosen = make_visual_sample(cohort, sources)
        identity = {
            "policy": POLICY,
            "input_policy": FULL_PA_POLICY,
            "cache_size": 512,
            "model_size": 448,
            "visual_sample_patients": chosen,
            "source_sha256": {
                str(path): file_sha256(path)
                for path in (sources_path, *task_paths.values())
            },
            "code_sha256": {
                str(path): file_sha256(path)
                for path in (
                    Path(__file__),
                    ROOT / "src/drai/data/dicom_gray.py",
                    ROOT / "src/drai/data/full_pa.py",
                    ROOT / "src/drai/data/preprocess_io.py",
                )
            },
        }
        identity_path = output / "run_identity.json"
        if identity_path.exists():
            if not args.resume or read_sealed_record(identity_path) != identity:
                raise ValueError("existing QC needs --resume with identical source list/code/policy identity")
        else:
            if args.resume:
                raise FileNotFoundError("no QC version to resume")
            atomic_json(identity_path, sealed_record(identity))
        selected = [
            row
            for row in cohort
            if not args.review_pilot or row["patient_id"] in chosen
        ]
        selected.sort(
            key=lambda row: hashlib.sha256(
                (POLICY + row["patient_id"]).encode()
            ).hexdigest()
        )
        tasks = [
            (
                row["patient_id"],
                sources[row["input_archive_relpath"]],
                str(output),
                args.retry_failures,
            )
            for row in selected
        ]
        results = []
        started = monotonic()
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context("spawn")
        ) as executor:
            pending = [executor.submit(prepare_one, task) for task in tasks]
            for future in as_completed(pending):
                results.append(future.result())
                if len(results) % 25 == 0 or len(results) == len(tasks):
                    progress = {
                        "processed": len(results),
                        "selected_this_run": len(tasks),
                        "target": len(cohort),
                        "passed": sum(item["status"] == "passed" for item in results),
                        "failed": sum(item["status"] != "passed" for item in results),
                        "elapsed_seconds": monotonic() - started,
                        "training_started": False,
                    }
                    atomic_json(report_dir / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        render_visual_samples(
            chosen, {item["patient_id"]: item for item in results}, report_dir
        )
        summary = summarize(results, len(cohort), output, report_dir, task_rows)
        summary["elapsed_seconds_this_run"] = monotonic() - started
        atomic_json(report_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
