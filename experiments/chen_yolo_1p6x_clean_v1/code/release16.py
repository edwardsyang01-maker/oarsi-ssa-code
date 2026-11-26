from collections import Counter, defaultdict
import fcntl
import json
from pathlib import Path
import time

import pipeline16 as p

DEST = p.crop.PROJECT / 'data/manifests/chen_yolo_usable448_1p6x_clean_v1_20261112'
COMMON = p.crop.PROJECT / 'data/manifests/chen_yolo_common448_1p3_1p6_2p0_v1_20261112'


def read_rows(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def validate_review():
    receipt_path = p.OUT / 'final_visual_review.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['approved_for_engineering_manifest']
    assert not receipt['full_manual_review_completed'] and not receipt['clinical_quality_approved']
    assert receipt['automatic_completion_sha256'] == p.sha(p.OUT / 'automatic_completion.json')
    reviewed, edges, holds = set(), set(), set()
    split_counts = Counter()
    for bundle in receipt['preview_bundles']:
        folder = p.ROOT / bundle['stage']
        assert p.sha(folder / (bundle['name'] + '.zip')) == bundle['sha256']
        public = json.loads((folder / bundle['name'] / 'public_index.json').read_text())
        private = json.loads((folder / (bundle['name'] + '_private.json')).read_text())
        assert len(public['records']) == len(private) == bundle['images_viewed']
        assert bundle['pages_actually_viewed'] == list(range(1, len(public['pages']) + 1))
        for page in public['pages']:
            assert p.sha(folder / bundle['name'] / page['file']) == page['sha256']
        mapping = {r['alias']: r for r in private}
        for alias in bundle['hold_aliases']:
            holds.add(mapping[alias]['key'])
        for r in private:
            final_file = p.ROOT / 'edge_finish_v1/cases' / (r['item']['case_id'] + '.json')
            final = json.loads(final_file.read_text())
            e = next(e for e in final['entries'] if e['key'] == r['key'])
            for key in ['gray01_sha256', 'valid_mask_sha256', 'cleanup_status']:
                assert e[key] == r['item'][key]
            if r['key'] not in reviewed:
                split_counts[e['split']] += 1
                reviewed.add(r['key'])
            if e['edge_extra_mask_pixels'] or (e['baseline_cleanup_status'].startswith('accepted') and not e['engineering_input_ready']):
                edges.add(r['key'])
    expected = set()
    for file in (p.ROOT / 'edge_finish_v1/cases').glob('*.json'):
        for e in json.loads(file.read_text())['entries']:
            if e['edge_extra_mask_pixels'] or (e['baseline_cleanup_status'].startswith('accepted') and not e['engineering_input_ready']):
                expected.add(e['key'])
    assert expected <= edges
    return holds, {'unique_final_images_visually_reviewed': len(reviewed),
                   'visually_reviewed_split_counts': dict(split_counts),
                   'all_edge_supplement_images_visually_reviewed': True,
                   'edge_supplement_images_visually_reviewed': len(expected),
                   'final_visual_review_sha256': p.sha(receipt_path)}


def publish():
    report = json.loads((p.OUT / 'automatic_completion.json').read_text())
    assert report['all_images_automatically_audited'] and report['all_modified_pixels_verified']
    assert p.identity()['identity'] == report['identity']
    holds, visual = validate_review()
    assert not DEST.exists() and not COMMON.exists()
    stage = DEST.with_name(DEST.name + '.staging')
    stage.mkdir(parents=True, exist_ok=False)
    counts, patients, labels = defaultdict(Counter), defaultdict(set), defaultdict(Counter)
    handles = {(s, yes): (stage / (s + ('_usable.jsonl' if yes else '_held.jsonl'))).open('w')
               for s in ['train', 'val'] for yes in [True, False]}
    found, all_keys = set(), set()
    try:
        for name, expected in report['candidate_manifest_sha256'].items():
            path = p.ROOT / 'candidate' / name
            assert p.sha(path) == expected
            for e in read_rows(path):
                assert e['key'] not in all_keys
                all_keys.add(e['key'])
                if e['key'] in holds:
                    found.add(e['key'])
                    if e['engineering_input_ready']:
                        e['engineering_input_ready'] = False
                        e['cleanup_status'] = 'held_final_visual_review'
                        e['cleanup_review_reasons'] += ['final_visual_uncertainty']
                s, ready = e['split'], e['engineering_input_ready']
                counts[s]['total'] += 1
                counts[s]['usable' if ready else 'held'] += 1
                counts[s][e['cleanup_status']] += 1
                if ready:
                    patients[s].add(e['patient_id'])
                    labels[s][str(e['kl'])] += 1
                e['eligibility'] = 'usable_technical_qa' if ready else 'held_excluded_from_training'
                e['final_visual_review_sha256'] = visual['final_visual_review_sha256']
                handles[(s, ready)].write(json.dumps(e, ensure_ascii=False) + '\n')
    finally:
        for f in handles.values():
            f.close()
    assert holds <= found and len(all_keys) == 38520
    assert {s: counts[s]['total'] for s in counts} == p.crop.EXPECTED
    assert not patients['train'] & patients['val']
    final = {**report, **visual, 'phase': 'completed', 'counts': dict(counts),
             'usable_patient_counts': {s: len(v) for s, v in patients.items()}, 'usable_kl_counts': dict(labels),
             'manifest_directory': str(DEST), 'manifest_sha256': {f.name: p.sha(f) for f in sorted(stage.glob('*.jsonl'))},
             'final_visual_additional_hold_images': len(holds), 'final_manifests_published': True,
             'completed_unix': time.time()}
    p.save(stage / 'summary.json', final)
    stage.rename(DEST)
    p.save(p.OUT / 'completion.json', final)
    return final


def common_cohort(final):
    folders = {'v1p3': p.crop.PARENT, 'v1p6': DEST, 'v2p0': p.PRIOR_MANIFEST}
    inputs, rows, by_split = {}, {}, {}
    for version, directory in folders.items():
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['phase'] == 'completed'
        rows[version] = {}
        inputs[version] = {}
        by_split[version] = Counter()
        for split in ['train', 'val']:
            path = directory / (split + '_usable.jsonl')
            assert p.sha(path) == summary['manifest_sha256'][path.name]
            inputs[version][path.name] = p.sha(path)
            for e in read_rows(path):
                assert e['split'] == split and e['key'] not in rows[version]
                rows[version][e['key']] = e
                by_split[version][split] += 1
    common = set.intersection(*(set(r) for r in rows.values()))
    assert common
    stage = COMMON.with_name(COMMON.name + '.staging')
    stage.mkdir(parents=True, exist_ok=False)
    manifests = {}
    counts, patients = Counter(), defaultdict(set)
    for key in sorted(common):
        base = rows['v1p3'][key]
        for version in ['v1p6', 'v2p0']:
            other = rows[version][key]
            for field in ['key', 'patient_id', 'clinical_side', 'image_position', 'split', 'kl', 'variant']:
                assert base[field] == other[field]
        counts[base['split']] += 1
        patients[base['split']].add(base['patient_id'])
    assert not patients['train'] & patients['val']
    for version, mapping in rows.items():
        directory = stage / version
        directory.mkdir()
        for split in ['train', 'val']:
            path = directory / (split + '_usable.jsonl')
            with path.open('w') as f:
                for key in sorted(common):
                    e = mapping[key]
                    if e['split'] == split:
                        f.write(json.dumps(e, ensure_ascii=False) + '\n')
            manifests[version + '/' + path.name] = p.sha(path)
    report = {'phase': 'completed', 'versions': [1.3, 1.6, 2.0], 'output_shape': [1, 448, 448],
              'version_usable_counts': dict(by_split), 'common_counts': dict(counts),
              'common_patient_counts': {s: len(v) for s, v in patients.items()},
              'patient_split_disjoint': True, 'sample_side_label_alignment_verified': True,
              'source_manifest_sha256': inputs, 'manifest_sha256': manifests,
              'manifest_directory': str(COMMON), 'clinical_quality_approved': False, 'training_started': False,
              'note': 'Three crop versions on identical samples; existing image processing and masking differences remain documented.'}
    p.save(stage / 'summary.json', report)
    stage.rename(COMMON)
    p.save(p.OUT / 'three_version_comparison.json', report)
    return report


def main():
    lock = (p.OUT / 'release.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    final = publish()
    common = common_cohort(final)
    p.save(p.OUT / 'status.json', {'phase': 'completed', 'counts': final['counts'], 'common_counts': common['common_counts']})
    print(json.dumps({'phase': 'completed', 'counts': final['counts'], 'common_counts': common['common_counts']}), flush=True)


if __name__ == '__main__':
    main()
