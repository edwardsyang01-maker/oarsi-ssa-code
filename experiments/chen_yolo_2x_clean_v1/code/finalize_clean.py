import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np

PROJECT = Path('/path/to/drai')
ROOT = PROJECT / 'experiments/chen_yolo_2x_clean_v1'
OUT = ROOT / 'release_v1'
CANDIDATE = OUT / 'candidate'
DEST = PROJECT / 'data/manifests/chen_yolo_usable448_2x_clean_v1_20261112'
EXPECTED = {'train': 34258, 'val': 4262}
IMMUTABLE = [
    'key', 'case_id', 'patient_id', 'film_id', 'visit', 'kl', 'split',
    'clinical_side', 'image_position', 'side_policy', 'bbox', 'transform',
    'detection_geometry', 'parent_1p3_bbox', 'gray_mapping', 'variant',
    'candidate_family', 'shape', 'output_resolution', 'detector_expansion',
    'archive_sha256', 'native_pixel_sha256', 'parent448_gray01_sha256',
    'parent448_valid_mask_sha256', 'source_is_native_pixels',
    'resized_from_224', 'parent448_exact_reproduction',
]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def load_array(path, digest, dtype):
    assert sha(path) == digest, 'array_hash_mismatch'
    arr = np.load(path, allow_pickle=False)
    assert arr.shape == (1, 448, 448) and arr.dtype == dtype, 'array_shape_or_dtype'
    if dtype == np.float32:
        assert np.isfinite(arr).all() and arr.min() >= 0 and arr.max() <= 1
    return arr


def original_roi_mask(item):
    t = item['transform']
    b = item['bbox']
    x = b[0] + (np.arange(448) - t['offset_x'] + .5) / t['scale_x'] - .5
    y = b[1] + (np.arange(448) - t['offset_y'] + .5) / t['scale_y'] - .5
    old = item['detection_geometry']['unclipped_1p3_bbox']
    return ((x >= old[0]) & (x <= old[2]))[None, :] & ((y >= old[1]) & (y <= old[3]))[:, None]


def geometry_check(item, native_hw):
    assert item['shape'] == [1, 448, 448] and item['output_resolution'] == 448
    assert item['detector_expansion'] == 2.0
    assert item['source_is_native_pixels'] and not item['resized_from_224']
    assert item['parent448_exact_reproduction']
    assert item['clinical_side'] == {'image_left': 'R', 'image_right': 'L'}[item['image_position']]
    assert 'no spatial flip' in item['side_policy']
    old = np.asarray(item['detection_geometry']['unclipped_1p3_bbox'], dtype=float)
    center = (old[:2] + old[2:]) / 2
    half = (old[2:] - old[:2]) / 1.3
    requested = np.r_[center - half, center + half]
    assert np.allclose(requested, item['detection_geometry']['requested_2p0_bbox'], atol=1e-7, rtol=0)
    b = item['bbox']
    prev = item['parent_1p3_bbox']
    h, w = native_hw
    assert 0 <= b[0] <= prev[0] < prev[2] <= b[2] <= w
    assert 0 <= b[1] <= prev[1] < prev[3] <= b[3] <= h
    t = item['transform']
    assert t['source_bbox_xyxy_exclusive'] == b
    assert t['source_crop_size_wh'] == [b[2] - b[0], b[3] - b[1]]
    assert 0 <= t['offset_x'] and t['offset_x'] + t['resized_width'] <= 448
    assert 0 <= t['offset_y'] and t['offset_y'] + t['resized_height'] <= 448


def summary_of(records):
    counts = defaultdict(Counter)
    patients = defaultdict(set)
    reasons = defaultdict(Counter)
    labels = defaultdict(Counter)
    for e in records:
        split = e['split']
        counts[split]['total'] += 1
        counts[split][e['cleanup_status']] += 1
        use = e['engineering_input_ready']
        counts[split]['usable' if use else 'held'] += 1
        if use:
            patients[split].add(e['patient_id'])
            labels[split][str(e['kl'])] += 1
            counts[split]['ruler_removed_usable'] += bool(e['ruler_removal_applied'])
        else:
            reasons[split].update(e['cleanup_review_reasons'])
    return {
        'counts': {s: dict(v) for s, v in counts.items()},
        'usable_patient_counts': {s: len(v) for s, v in patients.items()},
        'usable_kl_counts': {s: dict(v) for s, v in labels.items()},
        'held_reason_counts_nonexclusive': {s: dict(v) for s, v in reasons.items()},
    }


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def verify():
    assert not (OUT / 'verification.json').exists(), 'already_verified'
    requirements = [
        'outputs/raw_completion.json', 'audit_v2/completion.json',
        'proposals_v1/completion.json', 'clean_v1/completion.json',
        'edge_finish_v1/completion.json',
    ]
    started = time.time()
    while not all((ROOT / x).exists() for x in requirements):
        save(OUT / 'status.json', {'phase': 'waiting_for_all_stages', 'updated_unix': time.time()})
        time.sleep(20)
    inputs = {x: sha(ROOT / x) for x in requirements}
    raw_report = json.loads((ROOT / requirements[0]).read_text())
    edge_report = json.loads((ROOT / requirements[-1]).read_text())
    assert raw_report['counts'] == EXPECTED and edge_report['knees'] == sum(EXPECTED.values())
    raw = {}
    split_patients = defaultdict(set)
    for split in EXPECTED:
        file = ROOT / 'raw_manifests' / (split + '_raw2x.jsonl')
        assert sha(file) == raw_report['manifest_sha256'][split]
        inputs[str(file.relative_to(ROOT))] = sha(file)
        count = 0
        for e in read_jsonl(file):
            assert e['key'] not in raw and e['split'] == split
            raw[e['key']] = {k: e[k] for k in IMMUTABLE + [
                'gray01_npy', 'gray01_sha256', 'valid_mask_npy', 'valid_mask_sha256',
                'source_result', 'source_result_sha256', 'crop_generator_identity',
                'old_window_geometry_repair',
            ]}
            split_patients[split].add(e['patient_id'])
            count += 1
        assert count == EXPECTED[split]
    assert not split_patients['train'] & split_patients['val']
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = OUT / 'verified_private.jsonl'
    assert not scratch.exists(), 'partial_verification_requires_explicit_audit'
    seen = set()
    case_digests = {}
    cross_hashes = defaultdict(lambda: defaultdict(list))
    modified = 0
    extra = 0
    files = sorted((ROOT / 'edge_finish_v1/cases').glob('*.json'))
    assert len(files) == 19264
    with scratch.open('w') as output:
        for i, file in enumerate(files, 1):
            case = json.loads(file.read_text())
            assert case['identity'] == edge_report['identity']['identity']
            digest = sha(file)
            case_digests[file.name] = digest
            for stem in ['source_result', 'proposal_result', 'baseline_cleanup_result']:
                assert sha(case[stem + '_path']) == case[stem + '_sha256']
            for e in case['entries']:
                key = e['key']
                assert key in raw and key not in seen
                seen.add(key)
                r = raw[key]
                assert all(e[k] == r[k] for k in IMMUTABLE), 'identity_or_label_changed'
                assert case['source_result_path'] == r['source_result']
                assert case['source_result_sha256'] == r['source_result_sha256']
                geometry_check(e, case['native_hw'])
                for stem, raw_stem in [('gray01', 'gray01'), ('valid_mask', 'valid_mask')]:
                    assert e['raw_' + stem + '_npy'] == r[raw_stem + '_npy']
                    assert e['raw_' + stem + '_sha256'] == r[raw_stem + '_sha256']
                g = load_array(e['gray01_npy'], e['gray01_sha256'], np.float32)
                v = load_array(e['valid_mask_npy'], e['valid_mask_sha256'], np.bool_)
                assert (g[~v] == 0).all()
                assert e['edge_audit_completed'] and e['automated_full_image_audit']
                assert not e['clinical_quality_approved']
                if e['ruler_removal_applied']:
                    original = load_array(r['gray01_npy'], r['gray01_sha256'], np.float32)
                    original_valid = load_array(r['valid_mask_npy'], r['valid_mask_sha256'], np.bool_)
                    mask = load_array(e['cleanup_mask_npy'], e['cleanup_mask_sha256'], np.bool_)
                    assert int(mask.sum()) == e['proposed_mask_pixels']
                    assert not (mask[0] & original_roi_mask(e)).any(), 'original_joint_roi_was_modified'
                    assert not (mask & ~original_valid).any()
                    assert (g[mask] == 0).all() and np.array_equal(g[~mask], original[~mask])
                    assert np.array_equal(v, original_valid & ~mask)
                    modified += 1
                else:
                    assert e['gray01_sha256'] == r['gray01_sha256']
                    assert e['valid_mask_sha256'] == r['valid_mask_sha256']
                    assert e['cleanup_mask_npy'] is None
                use = e['engineering_input_ready']
                assert use == e['cleanup_status'].startswith('accepted')
                if use:
                    assert not e['cleanup_review_reasons'] and not e['postclean_high_confidence_rulers']
                    assert not e['edge_single_fragments']
                    cross_hashes[e['gray01_sha256']][e['split']].append(key)
                extra += e['edge_extra_mask_pixels'] > 0
                record = {
                    **e, 'source_result': r['source_result'],
                    'source_result_sha256': r['source_result_sha256'],
                    'crop_generator_identity': r['crop_generator_identity'],
                    'old_window_geometry_repair': r['old_window_geometry_repair'],
                    'final_cleanup_result': str(file), 'final_cleanup_result_sha256': digest,
                    'final_verifier_code_sha256': sha(__file__),
                    'raw_source_preserved': True,
                    'mask_outside_original_1p3_roi_verified': True,
                    'unmasked_pixels_bit_exact_verified': True,
                    'full_manual_review_completed': False,
                }
                output.write(json.dumps(record, ensure_ascii=False) + '\n')
            if i % 200 == 0:
                save(OUT / 'status.json', {
                    'phase': 'independent_pixel_verification', 'cases': i,
                    'target_cases': len(files), 'knees': len(seen), 'updated_unix': time.time(),
                })
    assert seen == set(raw)
    duplicates = set()
    for groups in cross_hashes.values():
        if len(groups) > 1:
            for keys in groups.values():
                duplicates.update(keys)
    CANDIDATE.mkdir(exist_ok=False)
    handles = {(s, use): (CANDIDATE / (s + ('_usable.jsonl' if use else '_held.jsonl'))).open('w')
               for s in EXPECTED for use in [True, False]}
    records = []
    try:
        for e in read_jsonl(scratch):
            if e['key'] in duplicates:
                e['engineering_input_ready'] = False
                e['cleanup_status'] = 'held_cross_split_duplicate'
                e['cleanup_review_reasons'] += ['postclean_cross_split_identical_array']
            handles[(e['split'], e['engineering_input_ready'])].write(json.dumps(e, ensure_ascii=False) + '\n')
            records.append({k: e[k] for k in [
                'split', 'patient_id', 'kl', 'cleanup_status', 'engineering_input_ready',
                'ruler_removal_applied', 'cleanup_review_reasons',
            ]})
    finally:
        for f in handles.values():
            f.close()
    aggregate = summary_of(records)
    assert {s: v['total'] for s, v in aggregate['counts'].items()} == EXPECTED
    report = {
        'phase': 'verified_pending_final_visual_review', **aggregate,
        'raw_counts': EXPECTED, 'all_images_automatically_audited': True,
        'all_arrays_and_source_hashes_verified': True, 'all_modified_pixels_verified': True,
        'patient_split_disjoint': True,
        'patient_labels_sides_and_native_crop_geometry_preserved': True,
        'modified_arrays_including_held': modified,
        'edge_extra_mask_images_including_held': extra,
        'postclean_cross_split_duplicate_images_held': len(duplicates),
        'accepted_images_with_postclean_confirmed_ruler_candidates': 0,
        'full_manual_review_completed': False, 'clinical_quality_approved': False,
        'training_started': False, 'test_accessed': False,
        'output_shape': [1, 448, 448], 'detector_expansion': 2.0,
        'side_policy': 'image_left=R,image_right=L; no flip',
        'mask_policy': 'zero fill only outside original 1.3 ROI; raw and masks retained',
        'verification_code_sha256': sha(__file__), 'input_sha256': inputs,
        'case_result_hash_digest': hashlib.sha256(json.dumps(case_digests, sort_keys=True).encode()).hexdigest(),
        'candidate_manifest_sha256': {f.name: sha(f) for f in sorted(CANDIDATE.glob('*.jsonl'))},
        'elapsed_seconds': time.time() - started,
    }
    save(OUT / 'verification.json', report)
    save(OUT / 'status.json', {'phase': report['phase'], 'counts': report['counts']})
    print(json.dumps({'phase': report['phase'], 'counts': report['counts']}), flush=True)


def publish():
    report = json.loads((OUT / 'verification.json').read_text())
    review_path = OUT / 'final_visual_review.json'
    review = json.loads(review_path.read_text())
    assert review['approved_for_engineering_manifest']
    assert review['verification_sha256'] == sha(OUT / 'verification.json')
    assert review['all_exported_pages_actually_viewed']
    assert not review['clinical_quality_approved']
    for name, digest in report['candidate_manifest_sha256'].items():
        assert sha(CANDIDATE / name) == digest
    for name, digest in report['input_sha256'].items():
        assert sha(ROOT / name) == digest
    holds = set()
    for bundle in review['preview_bundles']:
        root = ROOT / bundle['relative_stage']
        assert sha(root / (bundle['name'] + '.zip')) == bundle['sha256']
        private = json.loads((root / (bundle['name'] + '_private.json')).read_text())
        mapping = {e['alias']: e for e in private}
        for alias in bundle.get('hold_aliases', []):
            holds.add(mapping[alias]['key'])
    assert not DEST.exists()
    staging = DEST.with_name(DEST.name + '.staging')
    staging.mkdir(parents=True, exist_ok=False)
    handles = {(s, use): (staging / (s + ('_usable.jsonl' if use else '_held.jsonl'))).open('w')
               for s in EXPECTED for use in [True, False]}
    stats = []
    found = set()
    try:
        for source in sorted(CANDIDATE.glob('*.jsonl')):
            for e in read_jsonl(source):
                if e['key'] in holds:
                    found.add(e['key'])
                    if e['engineering_input_ready']:
                        e['engineering_input_ready'] = False
                        e['cleanup_status'] = 'held_final_visual_review'
                        e['cleanup_review_reasons'] += ['final_visual_uncertainty_or_ruler_residual']
                e['final_visual_review_sha256'] = sha(review_path)
                e['eligibility'] = 'usable_technical_qa' if e['engineering_input_ready'] else 'held_excluded_from_training'
                handles[(e['split'], e['engineering_input_ready'])].write(json.dumps(e, ensure_ascii=False) + '\n')
                stats.append({k: e[k] for k in [
                    'split', 'patient_id', 'kl', 'cleanup_status', 'engineering_input_ready',
                    'ruler_removal_applied', 'cleanup_review_reasons',
                ]})
    finally:
        for f in handles.values():
            f.close()
    assert holds <= found
    summary = {
        **report, **summary_of(stats), 'phase': 'completed',
        'verification_sha256': sha(OUT / 'verification.json'),
        'final_visual_review_sha256': sha(review_path),
        'final_visual_additional_hold_images': len(holds),
        'manifest_sha256': {f.name: sha(f) for f in sorted(staging.glob('*.jsonl'))},
        'manifest_directory': str(DEST), 'final_manifests_published': True,
        'completed_unix': time.time(),
    }
    save(staging / 'summary.json', summary)
    staging.rename(DEST)
    save(OUT / 'completion.json', summary)
    save(OUT / 'status.json', {'phase': 'completed', 'counts': summary['counts']})
    print(json.dumps({'phase': 'completed', 'counts': summary['counts']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['verify', 'publish'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / (args.mode + '.lock')).open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (verify if args.mode == 'verify' else publish)()


if __name__ == '__main__':
    main()
