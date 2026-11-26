from collections import Counter
import json
from pathlib import Path
import pipeline16 as p
from release16 import DEST, COMMON

FIELDS = ['patient_id', 'clinical_side', 'image_position', 'split', 'kl', 'variant',
          'gray01_npy', 'gray01_sha256', 'valid_mask_npy', 'valid_mask_sha256']


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def main():
    final = json.loads((DEST / 'summary.json').read_text())
    common = json.loads((COMMON / 'summary.json').read_text())
    assert p.sha(DEST / 'summary.json') == p.sha(p.OUT / 'completion.json')
    assert p.sha(COMMON / 'summary.json') == p.sha(p.OUT / 'three_version_comparison.json')
    assert final['phase'] == common['phase'] == 'completed'
    assert final['unique_final_images_visually_reviewed'] == 382
    assert final['edge_supplement_images_visually_reviewed'] == 5
    assert final['final_visual_additional_hold_images'] == 1
    assert not final['clinical_quality_approved'] and not final['full_manual_review_completed']
    all_keys, ready_keys, patient_sets, final_counts = set(), set(), {}, {}
    visual_held = 0
    for split in ['train', 'val']:
        count = Counter()
        patient_sets[split] = set()
        for ready in [True, False]:
            name = split + ('_usable.jsonl' if ready else '_held.jsonl')
            assert p.sha(DEST / name) == final['manifest_sha256'][name]
            for e in rows(DEST / name):
                assert e['key'] not in all_keys and e['split'] == split
                all_keys.add(e['key'])
                assert bool(e['engineering_input_ready']) == ready
                count['total'] += 1
                count['usable' if ready else 'held'] += 1
                if ready:
                    ready_keys.add(e['key'])
                    patient_sets[split].add(e['patient_id'])
                if e['cleanup_status'] == 'held_final_visual_review':
                    assert not ready and 'final_visual_uncertainty' in e['cleanup_review_reasons']
                    visual_held += 1
        assert all(count[k] == final['counts'][split][k] for k in ['total', 'usable', 'held'])
        final_counts[split] = dict(count)
    assert len(all_keys) == 38520 and visual_held == 1
    assert not patient_sets['train'] & patient_sets['val']
    private = json.loads((p.ROOT / 'edge_finish_v1/train_stageb_private.json').read_text())
    held_key = next(r['key'] for r in private if r['alias'] == 'TRAIN_STAGEB-027')
    assert held_key in all_keys and held_key not in ready_keys
    folders = {'v1p3': p.crop.PARENT, 'v1p6': DEST, 'v2p0': p.PRIOR_MANIFEST}
    source = {}
    for version, folder in folders.items():
        source[version] = {}
        for split in ['train', 'val']:
            name = split + '_usable.jsonl'
            assert p.sha(folder / name) == common['source_manifest_sha256'][version][name]
            for e in rows(folder / name):
                assert e['key'] not in source[version]
                source[version][e['key']] = tuple(e[k] for k in FIELDS)
    expected = set(source['v1p3']) & set(source['v1p6']) & set(source['v2p0'])
    counts_by_version = {}
    for version in folders:
        found, counts = set(), Counter()
        for split in ['train', 'val']:
            name = version + '/' + split + '_usable.jsonl'
            assert p.sha(COMMON / name) == common['manifest_sha256'][name]
            for e in rows(COMMON / name):
                assert e['key'] in expected and e['key'] not in found
                assert tuple(e[k] for k in FIELDS) == source[version][e['key']]
                assert e['split'] == split
                found.add(e['key'])
                counts[split] += 1
        assert found == expected and dict(counts) == common['common_counts']
        counts_by_version[version] = dict(counts)
    for key in expected:
        assert source['v1p3'][key][:6] == source['v1p6'][key][:6] == source['v2p0'][key][:6]
    report = {'phase': 'passed', 'final_counts': final_counts, 'common_counts': common['common_counts'],
              'same_common_counts_all_versions': counts_by_version,
              'exact_usable_intersection_verified': True, 'patient_split_disjoint': True,
              'metadata_and_array_references_match_source': True, 'visual_hold_excluded': True,
              'published_manifest_hashes_verified': True, 'all_image_pixel_checks_bound_to_automatic_report': True,
              'completion_sha256': p.sha(p.OUT / 'completion.json'),
              'comparison_sha256': p.sha(p.OUT / 'three_version_comparison.json'),
              'automatic_completion_sha256': p.sha(p.OUT / 'automatic_completion.json'),
              'clinical_quality_approved': False, 'training_started': False}
    p.save(p.OUT / 'release_verification.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
