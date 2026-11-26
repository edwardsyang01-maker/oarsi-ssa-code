import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = "drai-pa-patient-split-v1:20261007"
KEEP = [
    "patient_id",
    "visit",
    "nominal_months",
    "exam_date_iso",
    "dicom_modality_first_instance",
    "side",
    "sequence_family",
    "archive_relpath",
    "archive_server_path",
    "archive_bytes",
    "file_nonempty",
    "manifest_dicom_date_disagree",
    "dicom_first_instance_rows",
    "dicom_first_instance_columns",
    "dicom_first_instance_pixel_spacing",
    "pixel_integrity_checked",
]


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_csv_once(path, fields, rows):
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8")
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"an existing list has different content; use a new version: {path}")
    else:
        path.write_bytes(data)


def main():
    source = ROOT / "data/manifests/xray_knee_pa_candidates.csv"
    # The CSV has a BOM; utf-8-sig keeps it out of the first column name.
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not set(KEEP) <= set(reader.fieldnames):
            raise ValueError("source list lacks required fields")
        rows = list(reader)
    if any(not row["patient_id"] for row in rows):
        raise ValueError("empty patient ID; patient-level separation cannot be guaranteed")
    patients = sorted({row["patient_id"] for row in rows})
    ranked = sorted(
        patients, key=lambda p: (hashlib.sha256(f"{SEED}|{p}".encode()).hexdigest(), p)
    )
    n_holdout = round(len(patients) * 0.1)
    groups = {
        p: ("test" if i < n_holdout else "val" if i < 2 * n_holdout else "train")
        for i, p in enumerate(ranked)
    }
    sets = {
        name: {p for p in patients if groups[p] == name}
        for name in ("train", "val", "test")
    }
    assert not (
        sets["train"] & sets["val"]
        or sets["train"] & sets["test"]
        or sets["val"] & sets["test"]
    )
    assert set.union(*sets.values()) == set(patients)
    split_rows = [{"patient_id": p, "split": groups[p]} for p in patients]
    ssl_rows = [
        {
            **{field: row[field] for field in KEEP},
            "split": groups[row["patient_id"]],
            "pretraining_allowed_by_split": str(
                int(groups[row["patient_id"]] == "train")
            ),
            "train_ready": "0",
        }
        for row in rows
    ]
    assert len({row["archive_relpath"] for row in ssl_rows}) == len(rows)
    assert all(row["split"] == groups[row["patient_id"]] for row in ssl_rows)
    split_path = ROOT / "data/manifests/patient_split_v1.csv"
    ssl_path = ROOT / "data/manifests/pa_ssl_candidates_v1.csv"
    write_csv_once(split_path, ["patient_id", "split"], split_rows)
    write_csv_once(
        ssl_path,
        KEEP + ["split", "pretraining_allowed_by_split", "train_ready"],
        ssl_rows,
    )
    report = {
        "method": "sha256-ranked patient IDs; rounded 10% val and 10% test; remainder train",
        "seed_namespace": SEED,
        "source_manifest_sha256": sha256(source),
        "patient_counts": {name: len(values) for name, values in sets.items()},
        "archive_counts": dict(Counter(row["split"] for row in ssl_rows)),
        "modality_counts_by_split": {
            name: dict(
                Counter(
                    row["dicom_modality_first_instance"]
                    for row in ssl_rows
                    if row["split"] == name
                )
            )
            for name in sets
        },
        "patients_disjoint": True,
        "all_visits_share_patient_split": True,
        "no_disease_label_columns": True,
        "disease_labels_used_for_selection": False,
        "pretraining_split": "train",
        "all_archives_train_ready": False,
        "heldout_pretraining_prohibited": ["val", "test"],
        "split_manifest_sha256": sha256(split_path),
        "ssl_manifest_sha256": sha256(ssl_path),
        "remaining": [
            "full pixel/QC",
            "ROI method validation",
            "training intensity policy",
        ],
    }
    (ROOT / "outputs/preprocess/ssl_patient_split_v1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "patient_counts",
                    "archive_counts",
                    "patients_disjoint",
                    "no_disease_label_columns",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
