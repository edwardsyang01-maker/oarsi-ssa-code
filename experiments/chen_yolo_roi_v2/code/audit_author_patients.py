from pathlib import Path
from collections import Counter
import json
import sys
import zipfile
import requests

import pipeline as p
sys.path.insert(0, str(p.PILOT / 'code'))
from fetch_assets import RangeZip


def main():
    cache = p.OUT / 'public_detector_directory.json'
    if cache.exists():
        inventory = json.loads(cache.read_text())
    else:
        metadata = json.loads((p.PILOT / 'assets/mendeley_metadata.json').read_text())['content_details']
        reader = RangeZip(requests.Session(), metadata['download_url'], metadata['size'])
        with zipfile.ZipFile(reader) as archive:
            names = [x.filename for x in archive.infolist()
                     if '/DetKneeData/H5/' in x.filename and x.filename.endswith('.h5')]
        inventory = {'dataset': '10.17632/56rmx5bjcr.1', 'files': names,
                     'image_members_downloaded': False, 'whole_archive_downloaded': False}
        p.save(cache, inventory)
    groups = {}
    for group in ['train', 'val', 'test']:
        groups[group] = {Path(x).stem for x in inventory['files'] if f'/{group}H5/' in x}
        if not all(x.isdigit() and len(x) == 7 for x in groups[group]):
            raise RuntimeError('Unexpected author patient filename')
    if any(groups[a] & groups[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise RuntimeError('Author patient sets overlap')
    registry = p.read_csv(p.SPLIT)
    cohort = json.loads((p.OUT / 'cohort_private.json').read_text())
    selection = json.loads((p.OUT / 'selection_private.json').read_text())
    private = []
    for row in registry:
        exposure = next((group for group, ids in groups.items() if row['patient_id'] in ids), 'not_in_public_sets')
        private.append({**row, 'author_public_group': exposure})
    p.save(p.OUT / 'patient_exposure_private.json', private)
    exposure_by_patient = {r['patient_id']: r['author_public_group'] for r in private}
    own_sets = {s: {r['patient_id'] for r in registry if r['split'] == s} for s in ['train', 'val', 'test']}
    prepared_sets = {s: {r['patient_id'] for r in cohort if r['split'] == s} for s in ['train', 'val']}
    summary = {
        'author_public_patient_counts': {s: len(ids) for s, ids in groups.items()},
        'registry_overlap': {s: {'patients': len(ids), **{f'author_{a}': len(ids & patients) for a, patients in groups.items()}}
                             for s, ids in own_sets.items()},
        'prepared_cohort_overlap': {s: {'patients': len(ids), **{f'author_{a}': len(ids & patients) for a, patients in groups.items()},
                                        'outside_author_train_and_val': len(ids - groups['train'] - groups['val'])}
                                   for s, ids in prepared_sets.items()},
        'selection_exposure': {s: dict(Counter(exposure_by_patient[r['patient_id']] for r in selection if r['sample_group'] == s))
                               for s in ['regression48', 'random256']},
        'public_inventory_sha256': p.sha(cache), 'patient_registry_sha256': p.sha(p.SPLIT),
        'detector_weights_sha256': p.sha(p.MODEL_PATH),
        'evidence_boundary': 'published detector trainH5/valH5 membership; exact epoch-470 weight training log unavailable',
        'test_labels_or_images_opened': False,
        'interpretation': 'Any evaluation-patient overlap with detector training/selection data must be disclosed; not proof of KL-label leakage. Do not claim full-pipeline patient independence from classifier split alone.',
    }
    p.save(p.OUT / 'author_patient_overlap_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
