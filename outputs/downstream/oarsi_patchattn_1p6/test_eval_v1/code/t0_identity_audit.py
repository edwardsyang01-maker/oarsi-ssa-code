#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

SPLIT = 'data/manifests/patient_split_v1.csv'
SPLIT_SHA256 = 'ff675f2bcce9a7e5db374b6868cc934e55e476fd7a1a642860cc799996230fed'
CANDIDATES = 'data/manifests/xray_knee_pa_candidates.csv'
CANDIDATES_SHA256 = 'cceac628de222be4701bec09d507d9d83ffda71da606331f041c58c07c67ad15'
MANIFEST_DIR = 'data/manifests/chen_yolo_usable448_1p6x_clean_v1_20261112'
OLD_RESULTS = 'outputs/downstream/linear_probe_v1_newton'
NAS_ROOT = '/path/to/nas'
ALLOWED = ('patient_id', 'visit', 'modality', 'series_role', 'format', 'archive_relpath', 'archive_server_path',
           'archive_bytes', 'file_nonempty', 'dicom_first_instance_header_verified')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='drai project root')
    parser.add_argument('--out', required=True, help='output file (existing files are not overwritten)')
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    assert not out.exists(), f'output already exists; refusing to overwrite: {out}'
    hashes = {SPLIT: sha256_file(root / SPLIT), CANDIDATES: sha256_file(root / CANDIDATES)}

    split = {}
    with open(root / SPLIT, encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            assert split.setdefault(row['patient_id'].strip(), row['split']) == row['split']
    test = {p for p, s in split.items() if s == 'test'}
    manifest_patients = {}
    for name in ('train_usable', 'train_held', 'val_usable', 'val_held'):
        rel = f'{MANIFEST_DIR}/{name}.jsonl'
        hashes[rel] = sha256_file(root / rel)
        ids = set()
        with open(root / rel, encoding='utf-8') as f:
            for line in f:
                ids.add(str(json.loads(line)['patient_id']).strip())
        manifest_patients[name] = {'patients': len(ids), 'in_fixed_test': len(ids & test),
                                   'not_in_split_table': len(ids - set(split)),
                                   'split_of_patients': dict(Counter(split.get(p, 'missing') for p in ids))}

    archives = []
    with open(root / CANDIDATES, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        assert all(k in reader.fieldnames for k in ALLOWED)
        total_rows = Counter()
        for row in reader:
            pid = row['patient_id'].strip()
            total_rows[split.get(pid, 'missing')] += 1
            if pid in test:
                archives.append({k: row[k] for k in ALLOWED})
    on_disk = Counter()
    disk_bytes, manifest_bytes, prefixes, suffixes = 0, 0, Counter(), Counter()
    for a in archives:
        manifest_bytes += int(float(a['archive_bytes'] or 0))
        candidates = [a['archive_server_path'], os.path.join(NAS_ROOT, a['archive_relpath'])]
        path = next((c for c in candidates if c and os.path.isfile(c)), None)
        if path is None:
            on_disk['missing'] += 1
            continue
        size = os.stat(path).st_size
        disk_bytes += size
        on_disk['present'] += 1
        on_disk['size_matches_manifest' if size == int(float(a['archive_bytes'] or 0)) else 'size_differs_from_manifest'] += 1
        prefixes[os.sep.join(path.split(os.sep)[:8])] += 1
        suffixes[os.path.splitext(path)[1].lower()] += 1
    per_patient = Counter(a['patient_id'].strip() for a in archives)
    counts = np.array(sorted(per_patient.values()))

    old = {}
    old_union = set()
    base = root / OLD_RESULTS
    for path in sorted(base.glob('*/test_predictions.npz')):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if 'patient_ids.npy' not in names:
                old[path.parent.name] = {'members': names, 'patient_ids_member': False}
                continue
            ids = {str(x).strip() for x in np.load(io.BytesIO(z.read('patient_ids.npy')), allow_pickle=False)}
        old_union |= ids
        old[path.parent.name] = {'test_patients': len(ids), 'in_fixed_test': len(ids & test),
                                 'in_fixed_train': len({p for p in ids if split.get(p) == 'train'}),
                                 'in_fixed_val': len({p for p in ids if split.get(p) == 'val'}),
                                 'with_pa_archive_now': len(ids & set(per_patient))}
    with_archive = set(per_patient)
    result = {
        'schema': 'drai_wp6_t0_identity_and_use_audit_v1',
        'observed': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'scope': 'Metadata only: patient split, patient IDs in the lists, non-label columns of the candidate table, file existence and size, and patient_ids members of old result archives.'
                 'No test image was unpacked or decoded, and no test disease label, prediction or feature was read.',
        'input_sha256': hashes,
        'input_sha256_ok': hashes[SPLIT] == SPLIT_SHA256 and hashes[CANDIDATES] == CANDIDATES_SHA256,
        'split_table': dict(Counter(split.values())),
        'trainval_manifests_vs_fixed_test': manifest_patients,
        'candidate_table_rows_by_split': dict(total_rows),
        'test_archives': {
            'archives': len(archives), 'patients_with_archive': len(per_patient),
            'fixed_test_patients_without_archive': len(test - with_archive),
            'archives_per_patient': {'min': int(counts.min()), 'median': float(np.median(counts)), 'max': int(counts.max())},
            'by_visit': dict(sorted(Counter(a['visit'] for a in archives).items())),
            'by_modality': dict(sorted(Counter(a['modality'] for a in archives).items())),
            'by_series_role': dict(Counter(a['series_role'] for a in archives)),
            'by_format': dict(Counter(a['format'] for a in archives)),
            'file_nonempty_flag': dict(Counter(a['file_nonempty'] for a in archives)),
            'header_verified_flag': dict(Counter(a['dicom_first_instance_header_verified'] for a in archives)),
            'manifest_bytes_total': manifest_bytes, 'on_disk': dict(on_disk), 'on_disk_bytes_total': disk_bytes,
            'path_prefixes': dict(prefixes), 'file_suffixes': dict(suffixes)},
        'old_task_use': {
            'source_directory': OLD_RESULTS, 'tasks': old,
            'union_old_test_patients': len(old_union), 'union_in_fixed_test': len(old_union & test),
            'fixed_test_patients_in_any_old_task': len(test & old_union),
            'fixed_test_patients_in_no_listed_old_task': len(test - old_union),
            'test_patients_with_archive_in_any_old_task': len(with_archive & old_union),
            'test_patients_with_archive_in_no_listed_old_task': len(with_archive - old_union),
            'note': 'Covers only the test prediction archives with patient_ids in this one old result directory. Absence here does not mean never used for other tasks.'},
        'not_done': ['no test image unpacked or decoded', 'no test KL, OARSI or other disease label read', 'no prediction, label or feature array of old tasks read',
                     'no patient IDs exported', 'the final number of usable test knees is known only after T2 processes them under the approved protocol'],
    }
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('input_sha256_ok', 'split_table', 'candidate_table_rows_by_split')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
