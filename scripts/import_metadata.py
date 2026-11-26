import csv
import gzip
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/path/to/workspace/test/OAI_audit_20261007/results')
DEST = ROOT / 'data/manifests'
REPORT_DIR = ROOT / 'outputs/preprocess'

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    provenance = {}
    names = ['metadata.csv', 'metadata.csv.gz', 'metadata_schema.json', 'metadata_validation.json']
    for name in names:
        src, dst = SOURCE / name, DEST / name
        source_hash = sha256(src)
        if dst.exists():
            if sha256(dst) != source_hash:
                raise RuntimeError(f'Existing destination differs: {dst}')
        else:
            temporary = dst.with_name(dst.name + '.copying')
            with src.open('rb') as source_stream, temporary.open('xb') as target_stream:
                shutil.copyfileobj(source_stream, target_stream, length=8 * 1024 * 1024)
            if sha256(temporary) != source_hash:
                raise RuntimeError(f'Copy checksum mismatch: {src}')
            temporary.rename(dst)
        provenance[name] = {'source': str(src), 'destination': str(dst),
                            'bytes': dst.stat().st_size, 'sha256': source_hash}

    schema = json.loads((DEST / 'metadata_schema.json').read_text())
    validation = json.loads((DEST / 'metadata_validation.json').read_text())
    for name, expected in schema['checksum'].items():
        assert provenance[name]['sha256'] == expected['sha256'], name
        assert provenance[name]['bytes'] == expected['bytes'], name
    digest = hashlib.sha256()
    with gzip.open(DEST / 'metadata.csv.gz', 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == provenance['metadata.csv']['sha256']

    modalities, xray_profile, available, pixel_checked = Counter(), Counter(), Counter(), Counter()
    patients, archive_paths = set(), set()
    rows = 0
    # utf-8-sig handles the BOM; csv.DictReader keeps strings, so patient IDs are not turned into floats.
    # One row is one image archive, not a patient, a knee or a DICOM instance.
    with (DEST / 'metadata.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == schema['columns']
        for row in reader:
            rows += 1
            patients.add(row['patient_id'])
            archive_paths.add(row['archive_relpath'])
            modalities[row['modality']] += 1
            available[(row['modality'], row['file_nonempty'])] += 1
            pixel_checked[row['pixel_integrity_checked']] += 1
            if row['modality'] == 'X-Ray':
                xray_profile[(row['body'], row['sequence_family'], row['description'])] += 1
    assert rows == schema['rows'] == validation['unique_archive_paths']
    assert len(archive_paths) == rows
    assert len(patients) == validation['participants']
    report = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'files': provenance, 'copy_sha256_verified': True, 'gzip_matches_csv': True,
        'rows': rows, 'columns': schema['column_count'], 'unique_patients': len(patients),
        'modality_counts': dict(modalities),
        'availability': [{'modality': k[0], 'file_nonempty': k[1], 'rows': v} for k, v in available.items()],
        'pixel_integrity_checked_counts': dict(pixel_checked),
        'xray_profile': [{'body': k[0], 'sequence_family': k[1], 'description': k[2], 'rows': v}
                         for k, v in xray_profile.most_common()],
        'status': 'Metadata copied and verified; candidate selection and image preprocessing remain separate.',
        'nas_source_modified': False,
    }
    report_path = REPORT_DIR / 'metadata_import.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'files'}, ensure_ascii=False, indent=2))
    print(f'Metadata directory: {DEST}')
    print(f'Import report: {report_path}')

if __name__ == '__main__':
    main()
