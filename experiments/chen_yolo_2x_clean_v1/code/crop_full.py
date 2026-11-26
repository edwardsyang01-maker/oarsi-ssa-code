from collections import Counter, defaultdict
from pathlib import Path
import gc
import hashlib
import json
import sys
import time
import traceback

import numpy as np

PROJECT = Path('/path/to/drai')
sys.path.insert(0, str(PROJECT / 'experiments/chen_yolo_expand2x_v1/code'))
import preview_direct2x as pilot

e, base = pilot.e, pilot.base
ROOT = PROJECT / 'experiments/chen_yolo_2x_clean_v1'
OUT = ROOT / 'outputs'
PARENT = PROJECT / 'data/manifests/chen_yolo_usable448_v1_20261111'
DATA = PROJECT / 'data/processed/chen_yolo_2x_clean_v1/raw'
RAW_MANIFEST = ROOT / 'raw_manifests'
EXPECTED = {'train': 34258, 'val': 4262}


def save(path, value):
    e.p.save(path, value)


def identity():
    assert e.identity() == json.loads((e.OUT / 'identity.json').read_text())
    summary = json.loads((PARENT / 'summary.json').read_text())
    assert summary['phase'] == 'completed' and summary['output_shape'] == [1, 448, 448]
    assert e.p.sha(PARENT / 'summary.json') == '2bed6726a57313ebc0d39f4bc93ae57e037505b64abb6c4d13a9ed8ab00072d6'
    for name, digest in summary['manifest_sha256'].items():
        assert e.p.sha(PARENT / name) == digest
    deps = [Path(__file__), Path(pilot.__file__), Path(base.__file__), Path(e.__file__)]
    value = {'parent_summary_sha256': e.p.sha(PARENT / 'summary.json'),
             'parent_manifests': summary['manifest_sha256'],
             'dependencies': {str(f): e.p.sha(f) for f in deps},
             'policy': {'detector_expansion': 2.0, 'output_size': 448, 'source': 'native DICOM',
                        'gray': 'freeze the gray mapping of the original 1.3 version, including the local normalization thresholds',
                        'side': 'image_left=R,image_right=L', 'flip': False,
                        'patient_split_and_keys_unchanged': True, 'scope': ['train', 'val'],
                        'ruler_cleaning': 'separate later review stage; this stage only saves the complete native 2.0 crops'},
             'numpy': np.__version__}
    value['identity'] = e.p.digest(value)
    return value


def tasks():
    source = {x['case_id']: x for x in json.loads((e.p.OUT / 'selection_private.json').read_text())}
    grouped, counts, patients, keys = defaultdict(list), Counter(), defaultdict(set), set()
    for split in ['train', 'val']:
        for line in (PARENT / (split + '_usable.jsonl')).open():
            item = json.loads(line)
            assert item['split'] == split and item['key'] not in keys
            assert item['shape'] == [1, 448, 448] and item['output_resolution'] == 448
            row = source[item['case_id']]
            assert row['split'] == split and row['patient_id'] == item['patient_id']
            assert item['clinical_side'] == e.p.SIDES[item['image_position']]
            assert item['kl'] == row['labels'][item['clinical_side']]
            grouped[item['case_id']].append(item)
            counts[split] += 1
            patients[split].add(item['patient_id'])
            keys.add(item['key'])
    assert counts == EXPECTED and len(grouped) == 19264 and not patients['train'] & patients['val']
    return [{'row': source[case], 'items': items} for case, items in sorted(grouped.items())]


def pilot_cases(all_tasks):
    chosen, strata = set(), defaultdict(list)
    for task in all_tasks:
        for item in task['items']:
            strata[(item['split'], item['candidate_family'], item['variant'])].append(task['row']['case_id'])
            for flag in item['selected_flags']:
                strata[(item['split'], 'flag', flag)].append(task['row']['case_id'])
    for cases in strata.values():
        chosen.update(sorted(set(cases), key=lambda s: hashlib.sha256(s.encode()).hexdigest())[:2])
    previous = json.loads((pilot.OUT / 'selection_private.json').read_text())
    chosen.update(x['parent448']['case_id'] for x in previous)
    for split in ['train', 'val']:
        candidates = [t['row']['case_id'] for t in all_tasks if t['row']['split'] == split]
        chosen.update(sorted(candidates, key=lambda s: hashlib.sha256(('pilot2x'+s).encode()).hexdigest())[:12])
    return sorted(chosen)


def initialize():
    e.initialize()


def process(task):
    started = time.monotonic()
    row, items, key = task['row'], task['items'], task['identity']
    directory = DATA / row['case_id']
    directory.mkdir(parents=True, exist_ok=True)
    result = {'case_id': row['case_id'], 'split': row['split'], 'identity': key,
              'parent_items_digest': e.p.digest(items), 'entries': [], 'training': False, 'test_accessed': False}
    try:
        ds, pixels, receipt = e.read_pa_archive(e.p.NAS / row['input_archive_relpath'],
                                               expected_bytes=int(row['archive_bytes']), source_root=e.p.NAS)
        assert receipt['archive_sha256'] == row['archive_sha256']
        assert receipt['native_pixel_sha256'] == row['native_pixel_sha256']
        values, valid, pres = e.presentation_values(pixels, ds)
        variants = {x['variant'] for x in items}
        original = e.normalize_roi(values, valid)[0] if variants & {'original', 'full_original', 'lower_original'} else None
        del values
        raw = raw_values = inverted = None
        if variants - {'original', 'full_original', 'lower_original'}:
            raw, raw_values, _ = e.c.modality_gray(pixels, ds, valid, pres)
            if variants & {'inv_center_raw', 'inv_full_raw', 'lower_inv_raw'}:
                inverted = np.where(valid, 1 - raw, 0).astype(np.float32)
            if 'modality_roi_samebox' not in variants:
                raw_values = None
        del pixels
        result.update(native_hw=list(valid.shape), source_receipt=receipt)
        for item in items:
            for field in ['gray01', 'valid_mask']:
                assert e.p.sha(item[field + '_npy']) == item[field + '_sha256']
            variant = item['variant']
            mapping = {'variant': variant, 'frozen_parent_mapping': True}
            if variant in ['original', 'full_original', 'lower_original']:
                gray = original
            elif variant == 'modality_roi_samebox':
                x0, y0, x1, y1 = item['bbox']
                _, m = e.normalize_roi(raw_values[y0:y1, x0:x1], valid[y0:y1, x0:x1])
                gray = np.clip((raw_values - m['low']) / (m['high'] - m['low']), 0, 1).astype(np.float32)
                gray[~valid] = 0
                mapping.update(low=m['low'], high=m['high'])
            elif variant in ['modality_samebox', 'center_modality', 'full_modality', 'lower_raw']:
                gray = raw
            elif variant in ['inv_center_raw', 'inv_full_raw', 'lower_inv_raw']:
                gray = inverted
            else:
                raise ValueError('unsupported frozen gray variant')
            old = base.crop(gray, valid, item['bbox'], item)
            assert np.array_equal(old['image'], np.load(item['gray01_npy'], allow_pickle=False))
            assert np.array_equal(old['valid_mask'], np.load(item['valid_mask_npy'], allow_pickle=False))
            bbox, clipped, geometry = pilot.direct_box(item, valid.shape)
            new = base.crop(gray, valid, bbox, item)
            image_path = directory / (item['clinical_side'] + '_gray01.npy')
            mask_path = directory / (item['clinical_side'] + '_valid.npy')
            e.p.npy_save(image_path, new['image'])
            e.p.npy_save(mask_path, new['valid_mask'])
            flags = []
            if clipped:
                flags.append('native_source_clipped')
            if bbox[0] < valid.shape[1] / 2 < bbox[2]:
                flags.append('crosses_image_midline')
            entry = {**item, 'parent_1p3_bbox': item['bbox'], 'bbox': bbox,
                     'parent448_gray01_npy': item['gray01_npy'], 'parent448_gray01_sha256': item['gray01_sha256'],
                     'parent448_valid_mask_npy': item['valid_mask_npy'], 'parent448_valid_mask_sha256': item['valid_mask_sha256'],
                     'parent448_source_result': item['source_result'], 'parent448_source_result_sha256': item['source_result_sha256'],
                     'gray01_npy': str(image_path), 'gray01_sha256': e.p.sha(image_path),
                     'valid_mask_npy': str(mask_path), 'valid_mask_sha256': e.p.sha(mask_path),
                     'transform': new['transform'], 'detection_geometry': geometry, 'gray_mapping': mapping,
                     'shape': [1, 448, 448], 'detector_expansion': 2.0, 'parent448_exact_reproduction': True,
                     'parent_selected_flags': item['selected_flags'], 'selected_flags': flags,
                     'engineering_input_ready': False, 'clinical_quality_approved': False,
                     'anatomical_coverage_review': 'pending', 'ruler_review': 'pending', 'ruler_removal_applied': False}
            result['entries'].append(entry)
        result['status'] = 'completed'
    except Exception as exc:
        result.update(status='failed', error=str(exc), traceback=traceback.format_exc())
    result['seconds'] = time.monotonic() - started
    save(directory / 'result.json', result)
    gc.collect()
    return {'case_id': row['case_id'], 'status': result['status'], 'seconds': result['seconds']}


def verify_result(task, key, arrays=True):
    result = json.loads((DATA / task['row']['case_id'] / 'result.json').read_text())
    assert result['status'] == 'completed' and result['identity'] == key
    assert result['parent_items_digest'] == e.p.digest(task['items'])
    old = {x['key']: x for x in task['items']}
    assert {x['key'] for x in result['entries']} == set(old)
    for item in result['entries']:
        for name in ['key', 'split', 'patient_id', 'kl', 'clinical_side', 'image_position', 'variant']:
            assert item[name] == old[item['key']][name]
        assert item['parent_1p3_bbox'] == old[item['key']]['bbox'] and item['parent448_exact_reproduction']
        assert item['detector_expansion'] == 2.0 and not item['ruler_removal_applied']
        for name in ['gray01', 'valid_mask']:
            assert e.p.sha(item[name + '_npy']) == item[name + '_sha256']
        if arrays:
            e.check_array(np.load(item['gray01_npy'], allow_pickle=False), np.load(item['valid_mask_npy'], allow_pickle=False), 448)
        assert item['transform']['image_size'] == 448
    return result


def finalize(all_tasks, key):
    RAW_MANIFEST.mkdir(parents=True, exist_ok=False)
    counts, flags, hashes, result_digests = Counter(), defaultdict(Counter), defaultdict(set), {}
    handles = {s: (RAW_MANIFEST / (s + '_raw2x.jsonl')).open('w') for s in ['train', 'val']}
    try:
        for number, task in enumerate(all_tasks, start=1):
            r = verify_result(task, key)
            path = DATA / r['case_id'] / 'result.json'
            digest = e.p.sha(path)
            result_digests[r['case_id']] = digest
            for item in r['entries']:
                record = {**item, 'source_result': str(path), 'source_result_sha256': digest}
                handles[item['split']].write(json.dumps(record, ensure_ascii=False) + '\n')
                counts[item['split']] += 1
                flags[item['split']].update(item['selected_flags'])
                hashes[item['gray01_sha256']].add(item['split'])
            if number % 200 == 0:
                save(OUT / 'raw_verification_status.json', {'phase': 'verifying_raw2x', 'verified_cases': number, 'target_cases': len(all_tasks)})
    finally:
        for handle in handles.values():
            handle.close()
    assert counts == EXPECTED and not any(len(x) > 1 for x in hashes.values())
    report = {'phase': 'raw_cropping_completed_review_pending', 'identity': key, 'cases': len(all_tasks),
              'counts': dict(counts), 'output_shape': [1, 448, 448], 'detector_expansion': 2.0,
              'flags': {s: dict(v) for s, v in flags.items()}, 'parent448_exact_reproduction_knees': sum(counts.values()),
              'all_raw_arrays_verified': True, 'patient_split_disjoint': True, 'cross_split_roi_duplicates': 0,
              'manifest_sha256': {s: e.p.sha(RAW_MANIFEST / (s + '_raw2x.jsonl')) for s in ['train', 'val']},
              'result_hashes_digest': e.p.digest(result_digests), 'full_anatomical_review_completed': False,
              'ruler_cleaning_completed': False, 'training_started': False, 'test_accessed': False}
    save(OUT / 'raw_completion.json', report)
    return report
