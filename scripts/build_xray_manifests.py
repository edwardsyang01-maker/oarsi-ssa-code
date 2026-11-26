import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / 'data/manifests'
PA_FAMILIES = {'Bilateral PA Fixed Flexion Knee', 'PA Fixed Flexion Right Knee', 'PA Fixed Flexion Left Knee'}

def main():
    groups = {'xray_all_candidates': [], 'xray_knee_candidates': [], 'xray_knee_pa_candidates': []}
    source = MANIFESTS / 'metadata.csv'
    with source.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        for row in reader:
            if row['modality'] != 'X-Ray' or row['file_nonempty'] != '1':
                continue
            groups['xray_all_candidates'].append(row)
            if row['body'] == 'knee':
                groups['xray_knee_candidates'].append(row)
                if row['sequence_family'] in PA_FAMILIES:
                    groups['xray_knee_pa_candidates'].append(row)
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'source': str(source), 'row_unit': 'archive, not knee or DICOM frame',
              'eligibility': 'X-Ray and file_nonempty=1; not a complete image/clinical QC filter',
              'pa_sequence_families': sorted(PA_FAMILIES), 'groups': {},
              'patient_split_created_in_this_script': False, 'labels_selected': False,
              'disease_labels_required_for_ssl': False,
              'pixel_preprocessing_completed': False, 'nas_source_modified': False}
    for name, rows in groups.items():
        output = MANIFESTS / (name + '.csv')
        temporary = output.with_suffix('.csv.building')
        with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.replace(output)
        with output.open(encoding='utf-8-sig', newline='') as stream:
            verified_rows = list(csv.DictReader(stream))
        assert len(verified_rows) == len(rows)
        assert {r['archive_relpath'] for r in verified_rows} == {r['archive_relpath'] for r in rows}
        report['groups'][name] = {'path': str(output), 'archive_rows': len(rows),
            'patients': len({r['patient_id'] for r in rows}),
            'dicom_modalities': dict(Counter(r['dicom_modality_first_instance'] for r in rows)),
            'sequence_families': dict(Counter(r['sequence_family'] for r in rows)),
            'date_disagreement_rows': sum(r['manifest_dicom_date_disagree'] == '1' for r in rows),
            'sha256': digest, 'bytes': output.stat().st_size}
    path = ROOT / 'outputs/preprocess/xray_candidate_selection.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
