from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_pain(value: float) -> bool:
    return bool(pd.notna(value) and math.isfinite(float(value)) and 0 <= value <= 20)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "data/manifests/downstream_v2"
    target = root / "data/manifests/downstream_v3"
    output = root / "outputs/downstream"
    clinical_path = Path(
        "/path/to/workspace/test/OAI_audit_20261007/"
        "cache/sas_tables/allclinical00.sas7bdat"
    )
    old_report_path = output / "task_design_label_audit_v2.json"
    protocol_path = root / "downstream/specs/task_protocol_v3.json"
    report = json.loads(old_report_path.read_text())
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for task in ("baseline_max_kl5", "48m_any_compartment_jsn_ge1"):
        original = source / f"{task}_candidates.csv"
        destination = target / original.name
        destination.write_bytes(original.read_bytes())
        if sha256(original) != sha256(destination):
            raise RuntimeError("SHA mismatch after copying the original five-class/progression candidate lists")
        copied.append(original)
    with (source / "baseline_max_kl5_candidates.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        original_columns = list(reader.fieldnames)
        base_candidates = list(reader)
    if len({row["patient_id"] for row in base_candidates}) != len(base_candidates):
        raise ValueError("duplicate patients in the baseline candidates")
    clinical = pd.read_sas(clinical_path, encoding="latin1")
    if clinical.duplicated("ID").any():
        raise ValueError("duplicate IDs in the baseline clinical table; refusing to pick one silently")
    clinical = clinical.set_index("ID")
    records = []
    exclusions = {split: 0 for split in ("train", "val", "test")}
    for row in base_candidates:
        if row["input_visit"] != "V00" or row["train_ready"] != "0":
            raise ValueError("input candidate visit/QC status does not match the design")
        patient = row["patient_id"]
        if patient not in clinical.index:
            exclusions[row["split"]] += 1
            continue
        scores = clinical.loc[patient, ["V00WOMKPR", "V00WOMKPL"]]
        if not all(valid_pain(value) for value in scores):
            exclusions[row["split"]] += 1
            continue
        right, left = (float(value) for value in scores)
        records.append(
            {
                **row,
                "task": "baseline_max_womac_pain_regression",
                "label": max(right, left),
                "max_jsn_increase": "",
                "womac_pain_right": right,
                "womac_pain_left": left,
                "label_source_table": "allclinical00",
            }
        )
    columns = original_columns + [
        "womac_pain_right",
        "womac_pain_left",
        "label_source_table",
    ]
    path = target / "baseline_max_womac_pain_regression_candidates.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    task = "baseline_max_womac_pain_regression"
    report["tasks"][task] = {}
    for split in ("train", "val", "test"):
        values = pd.Series(
            [row["label"] for row in records if row["split"] == split], dtype=float
        )
        report["tasks"][task][split] = {
            "total": len(values),
            "task_type": "regression",
            "valid_range": [0, 20],
            "zero_pain": int((values == 0).sum()),
            "fractional_targets": int((values % 1 != 0).sum()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "missing_or_invalid_bilateral_scores": exclusions[split],
        }
    report.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_id": "drai-downstream-v3-three-tasks",
            "upstream_audit_v2_sha256": sha256(old_report_path),
            "audit_script_sha256": sha256(Path(__file__)),
            "protocol_sha256": sha256(protocol_path),
            "clinical_quality_approved": False,
            "model_training_started": False,
            "pain_score_decimals_preserved": True,
            "pain_cohort_reuses_barcode_verified_kl5_baseline_candidates": True,
        }
    )
    report["source_sha256"].update(
        {str(path): sha256(path) for path in (clinical_path, *copied)}
    )
    (output / "task_design_label_audit_v3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "protocol": report["protocol_id"],
                "pain_counts": {
                    split: report["tasks"][task][split]["total"]
                    for split in ("train", "val", "test")
                },
                "excluded_missing": exclusions,
                "classification_candidates_copied_without_change": True,
                "training_started": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
