from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PILOT = PROJECT / 'experiments/chen_yolo_crop_compare_v1'
OUT = ROOT / 'outputs'
DATA = PROJECT / 'data/processed/chen_yolo_roi_v2'
NAS = Path('/path/to/nas')
PREP = PROJECT / 'outputs/downstream/oa_kl_wholefilm_v2/preparation_20261025'
SPLIT = PROJECT / 'data/manifests/patient_split_v1.csv'
MODEL_PATH = PILOT / 'assets/chen_yolov2_1wd.pth'
MODEL_SHA = 'daa658b52e41ea7ad1493fc97ed34480d5dfb152e47c0bd296b187af2ceae95a'
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src'), str(PILOT / 'code')]
from compare_crops import author_input, backmap, tensor_from_u8, preview_crop

SIDES = {'image_left': 'R', 'image_right': 'L'}
CONFIG = {
    'schema': 'chen-yolo-roi-v2', 'seed': 'chen-yolo-expand-20261111-v2',
    'new_train_patients': 256, 'score_threshold': .12, 'nms_threshold': .3,
    'expansion': 1.3, 'output_size': 224, 'flip': False,
    'detector_float_range': [0, 255], 'detector_mean': .431, 'detector_std': .0872,
    'gray_policy': 'frozen presentation_values + wholefilm percentile 0.5/99.5',
    'side_policy': 'PA display image_left=R; image_right=L; no geometric flip',
    'qc': {'high_saturation_fraction': .10, 'low_saturation_fraction': .25,
           'min_valid_fraction': .95, 'min_p95_p05': .15},
    'qc_threshold_status': 'engineering review triggers; not validated clinical acceptance thresholds',
    'fallback_policy': 'geometry only for missing YOLO side; saved separately; never silently accepted',
    'training_ready': False, 'clinical_quality_approved': False,
}
MODEL = None
RUN_ID = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def write_csv(path, items, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(items)
    os.replace(temp, path)


def npy_save(path, array):
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:
        np.save(f, array, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def sources():
    paths = [Path(__file__), PILOT / 'code/compare_crops.py',
             PILOT / 'code/chen_yolo_adapter.py', MODEL_PATH,
             PROJECT / 'src/drai/data/dicom_gray.py',
             PROJECT / 'downstream/oa_kl_unified_v1/geometry.py', SPLIT,
             PREP / 'wholefilm_train.csv', PREP / 'wholefilm_val.csv',
             PILOT / 'outputs/selection_private.json']
    return {str(p): sha(p) for p in paths}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    identity_path = OUT / 'identity.json'
    if identity_path.exists():
        verify_identity()
        print('Existing immutable preparation verified', flush=True)
        return
    patients = {r['patient_id']: r['split'] for r in read_csv(SPLIT)}
    cohort = []
    keys = set()
    by_split = defaultdict(set)
    for split in ['train', 'val']:
        for r in read_csv(PREP / f'wholefilm_{split}.csv'):
            if patients.get(r['patient_id']) != split or r['split'] != split:
                raise RuntimeError('Patient split differs from frozen registry')
            if r['sample_id'] in keys:
                raise RuntimeError('Duplicate film identity')
            keys.add(r['sample_id'])
            by_split[split].add(r['patient_id'])
            labels = {'R': int(r['kl_right']), 'L': int(r['kl_left'])}
            if any(v not in range(5) for v in labels.values()):
                raise RuntimeError('Invalid ipsilateral KL label')
            cohort.append({
                'film_id': r['sample_id'], 'patient_id': r['patient_id'], 'split': split,
                'visit': r['visit'], 'modality': r['dicom_modality_first_instance'],
                'labels': labels, 'side_policy': CONFIG['side_policy'],
                **{k: r[k] for k in ['input_archive_relpath', 'archive_bytes',
                                    'archive_sha256', 'native_pixel_sha256']},
            })
    if by_split['train'] & by_split['val']:
        raise RuntimeError('Patient leakage between train and val')
    pilot = json.loads((PILOT / 'outputs/selection_private.json').read_text())['cases']
    pilot_patients = {r['patient_id'] for r in pilot}
    film_map = {r['film_id']: r for r in cohort}
    selection = []
    for r in pilot:
        item = dict(film_map[r['film_id']])
        if item['split'] != 'train':
            raise RuntimeError('Pilot includes held-out patient')
        item.update(case_id=r['case_id'], sample_group='regression48')
        selection.append(item)
    by_patient = defaultdict(list)
    for r in cohort:
        if r['split'] == 'train' and r['patient_id'] not in pilot_patients:
            by_patient[r['patient_id']].append(r)
    order = sorted(by_patient, key=lambda p: digest([CONFIG['seed'], 'patient', p]))
    for i, patient in enumerate(order[:CONFIG['new_train_patients']]):
        item = dict(min(by_patient[patient], key=lambda r: digest([CONFIG['seed'], 'film', r['film_id']])))
        item.update(case_id=f'E{i + 1:03d}', sample_group='random256')
        selection.append(item)
    if len(selection) != 304 or len({r['patient_id'] for r in selection}) != 304:
        raise RuntimeError('Expected 304 different train patients')
    save(OUT / 'cohort_private.json', cohort)
    save(OUT / 'selection_private.json', selection)
    save(OUT / 'config.json', CONFIG)
    summary = {
        'cohort_films': dict(Counter(r['split'] for r in cohort)),
        'cohort_patients': {k: len(v) for k, v in by_split.items()},
        'potential_knees_before_detection_qc': dict(Counter(r['split'] for r in cohort for _ in range(2))),
        'patient_split_disjoint': True, 'selection_films': len(selection),
        'selection_patients': len({r['patient_id'] for r in selection}),
        'selection_modality': {group: dict(Counter(r['modality'] for r in selection if r['sample_group'] == group))
                               for group in ['regression48', 'random256']},
        'selection_visits': dict(Counter(r['visit'] for r in selection if r['sample_group'] == 'random256')),
        'sampling': 'fixed-seed random 256 train patients excluding pilot48, then one random eligible film per patient; not film-weighted',
        'test_labels_opened': False, 'trainval_images_inferred': False,
        'full_cohort_inference_started': False, 'training_ready': False,
    }
    save(OUT / 'preparation_summary.json', summary)
    identity = {'config': CONFIG, 'sources': sources(),
                'cohort_sha256': sha(OUT / 'cohort_private.json'),
                'selection_sha256': sha(OUT / 'selection_private.json')}
    if identity['sources'][str(MODEL_PATH)] != MODEL_SHA:
        raise RuntimeError('Detector fingerprint mismatch')
    identity['run_id'] = digest(identity)
    save(identity_path, identity)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def verify_identity():
    identity = json.loads((OUT / 'identity.json').read_text())
    check = dict(identity)
    expected = check.pop('run_id')
    if digest(check) != expected or identity['config'] != CONFIG:
        raise RuntimeError('Invalid or changed run identity')
    if identity['sources'] != sources():
        raise RuntimeError('A frozen source changed; create a new experiment version')
    for name, key in [('cohort_private.json', 'cohort_sha256'), ('selection_private.json', 'selection_sha256')]:
        if sha(OUT / name) != identity[key]:
            raise RuntimeError(f'Changed {name}')
    if json.loads((OUT / 'config.json').read_text()) != CONFIG:
        raise RuntimeError('Changed config artifact')
    return identity['run_id']


def initializer(run_id):
    global MODEL, RUN_ID
    import torch
    from chen_yolo_adapter import load_detector
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    MODEL, _ = load_detector(MODEL_PATH)
    RUN_ID = run_id


def gray_metrics(gray, valid):
    a = gray[valid]
    if not len(a):
        raise ValueError('No valid pixels')
    p05, p95 = map(float, np.percentile(a, [5, 95]))
    return {'valid_fraction': float(valid.mean()), 'std': float(a.std()),
            'high_saturation_fraction': float((a >= 1 - 1e-6).mean()),
            'low_saturation_fraction': float((a <= 1e-6).mean()),
            'p05': p05, 'p95': p95, 'p95_p05': p95 - p05}


def metric_flags(metrics):
    limits = CONFIG['qc']
    result = []
    for key in ['high_saturation_fraction', 'low_saturation_fraction']:
        if metrics[key] >= limits[key]:
            result.append(key)
    if metrics['valid_fraction'] < limits['min_valid_fraction']:
        result.append('low_valid_fraction')
    if metrics['p95_p05'] < limits['min_p95_p05']:
        result.append('low_gray_dynamic_range')
    return result


def overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def write_roi(gray, valid, item, directory, method, labels):
    from downstream.oa_kl_unified_v1.geometry import extract_regions
    pos = item['image_position']
    x0, y0, x1, y1 = item['bbox']
    region = {'image_position': pos, 'bbox_xyxy_exclusive': item['bbox'],
              'quality_status': 'candidate', 'reject_reasons': []}
    result = dict(item)
    result.update(method=method, clinical_side=SIDES[pos], kl=labels[SIDES[pos]],
                  training_ready=False, clinical_quality_approved=False)
    flags = list(item.get('flags', []))
    result['flags'] = sorted(set(flags))
    try:
        metrics = gray_metrics(gray[y0:y1, x0:x1], valid[y0:y1, x0:x1])
        flags += metric_flags(metrics)
        result['gray_metrics'] = metrics
        result['flags'] = sorted(set(flags))
        crop = extract_regions(gray, valid, [region], image_size=CONFIG['output_size'])[0]
        name = method + '_' + pos
        image_path = directory / f'{name}_gray01.npy'
        mask_path = directory / f'{name}_valid.npy'
        npy_save(image_path, crop['image'])
        npy_save(mask_path, crop['valid_mask'])
        result.update(status='saved', gray01_npy=str(image_path), gray01_sha256=sha(image_path),
                      valid_mask_npy=str(mask_path), valid_mask_sha256=sha(mask_path),
                      shape=list(crop['image'].shape), transform=crop['transform'],
                      technical_qc='review_required' if flags or method != 'yolo' else 'no_automatic_flags')
    except Exception as e:
        result.update(status='extract_error', error=f'{type(e).__name__}: {e}', technical_qc='failed')
    return result


def save_review_panel(gray, yolo, old, case_id):
    from PIL import Image, ImageDraw
    h, w = gray.shape
    source = np.rint(np.clip(gray, 0, 1) * 255).astype('uint8')
    source[:int(.20*h)] = 0
    source[int(.88*h):] = 0
    source[:, :int(.05*w)] = 0
    source[:, int(.95*w):] = 0
    overview = Image.fromarray(source).convert('RGB')
    overview.thumbnail((480, 320), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (1400, 360), '#101820')
    canvas.paste(overview, (8, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f'{case_id} | old: orange | YOLO: cyan | display perimeter masked', fill='white')
    sx, sy = overview.width/w, overview.height/h
    for items, color in [(old, '#ffa500'), (yolo, '#00e5ff')]:
        for item in items:
            x0, y0, x1, y1 = item['bbox']
            draw.rectangle((8+x0*sx, 32+y0*sy, 8+x1*sx, 32+y1*sy), outline=color, width=2)
    for i, (method, pos) in enumerate([(m, p) for p in SIDES for m in ['old', 'yolo']]):
        x = 492 + i*226
        draw.text((x, 54), method + ' ' + SIDES[pos], fill='white')
        item = next((v for v in (old if method == 'old' else yolo) if v['image_position'] == pos), None)
        if item:
            canvas.paste(preview_crop(gray, item['bbox']), (x, 80))
        else:
            draw.text((x+20, 160), 'MISSING', fill='#ffad66')
    folder = OUT / 'review_server_only'
    folder.mkdir(parents=True, exist_ok=True)
    canvas.save(folder / f'{case_id}.png')


def process_case(row):
    import torch
    from drai.data.dicom_gray import read_pa_archive, presentation_values, normalize_roi
    from downstream.oa_kl_unified_v1.geometry import locate_knees, extract_regions
    from chen_yolo_adapter import decode
    if row['split'] != 'train':
        raise RuntimeError('This engineering stage must not process validation/test images')
    directory = DATA / row['case_id']
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = {**row, 'run_id': RUN_ID, 'training_ready': False,
              'clinical_quality_approved': False, 'yolo': [], 'fallback': [], 'old': []}
    try:
        ds, pixels, receipt = read_pa_archive(NAS / row['input_archive_relpath'],
                                             expected_bytes=int(row['archive_bytes']), source_root=NAS)
        if receipt['archive_sha256'] != row['archive_sha256'] or receipt['native_pixel_sha256'] != row['native_pixel_sha256']:
            raise RuntimeError('Original archive/pixel fingerprint mismatch')
        values, valid, presentation = presentation_values(pixels, ds)
        gray, normalization = normalize_roi(values, valid)
        del values, pixels
        small, transform = author_input(gray, ds)
        film_metrics = gray_metrics(gray, valid)
        with torch.inference_mode():
            raw = decode(MODEL(tensor_from_u8(small)), threshold=CONFIG['score_threshold'],
                         nms_threshold=CONFIG['nms_threshold'])
        mapped = []
        h, w = gray.shape
        for detection in raw:
            box, unbounded = backmap(detection['bbox'], transform, CONFIG['expansion'])
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            pos = 'image_left' if (box[0] + box[2]) / 2 < w / 2 else 'image_right'
            flags = []
            if any(abs(a - b) > 1.01 for a, b in zip(box, unbounded)):
                flags.append('expanded_box_clipped')
            if box[0] < w / 2 < box[2]:
                flags.append('crosses_image_midline')
            if transform['padding_needed']:
                flags.append('detector_center_padding')
            if any(v <= 0 for v in detection['bbox'][:2]) or detection['bbox'][2] >= 319 or detection['bbox'][3] >= 255:
                flags.append('detector_frame_edge')
            mapped.append({'image_position': pos, 'bbox': box, 'unclipped_bbox': unbounded,
                           'detector_bbox': detection['bbox'], 'score': detection['score'], 'flags': flags})
        selected = []
        for pos in SIDES:
            items = [x for x in mapped if x['image_position'] == pos]
            if items:
                item = dict(max(items, key=lambda x: x['score']))
                item['flags'] = list(item['flags'])
                if len(items) > 1:
                    item['flags'].append('multiple_detections_same_side')
                selected.append(item)
        if len(selected) == 2 and overlap(selected[0]['bbox'], selected[1]['bbox']):
            for item in selected:
                item['flags'].append('bilateral_boxes_overlap')
        old = []
        old_rejected = []
        for region in locate_knees(gray, valid, layout='bilateral')['regions']:
            if region['quality_status'] != 'candidate':
                old_rejected.append(region)
                continue
            try:
                old_crop = extract_regions(gray, valid, [region], image_size=448)[0]
                old.append({'image_position': old_crop['image_position'],
                            'bbox': old_crop['bbox_xyxy_exclusive']})
            except Exception as e:
                old_rejected.append({'image_position': region['image_position'], 'error': str(e)})
        yolo = [write_roi(gray, valid, item, directory, 'yolo', row['labels']) for item in selected]
        missing = [p for p in SIDES if p not in {x['image_position'] for x in selected}]
        fallback = [write_roi(gray, valid, {**item, 'flags': ['yolo_missed_geometry_review_only']},
                              directory, 'geometry_fallback', row['labels'])
                    for item in old if item['image_position'] in missing]
        save_review_panel(gray, yolo, old, row['case_id'])
        result.update(status='completed', source_verified=True, source_receipt=receipt,
                      native_hw=list(gray.shape), presentation=presentation, normalization=normalization,
                      detector_transform=transform, film_gray_metrics=film_metrics,
                      film_flags=metric_flags(film_metrics), raw_detections=raw,
                      raw_detection_count=len(raw), mapped_detection_count=len(mapped),
                      duplicate_count=len(mapped) - len(selected), missing_positions=missing,
                      yolo=yolo, fallback=fallback, old=old, old_rejected=old_rejected)
    except Exception as e:
        result.update(status='case_error', error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())
    result['seconds'] = time.monotonic() - started
    save(directory / 'result.json', result)
    gc.collect()
    return {'case_id': row['case_id'], 'status': result['status'],
            'yolo': len(result['yolo']), 'seconds': result['seconds']}


def check_checkpoint(row, run_id):
    path = DATA / row['case_id'] / 'result.json'
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    if result['run_id'] != run_id or result['film_id'] != row['film_id']:
        raise RuntimeError('Checkpoint identity mismatch')
    for roi in result['yolo'] + result['fallback']:
        if roi['status'] == 'saved':
            for field in ['gray01', 'valid_mask']:
                if sha(roi[field + '_npy']) != roi[field + '_sha256']:
                    raise RuntimeError('Checkpoint output hash mismatch')
    return True


def run(resume, limit, workers):
    run_id = verify_identity()
    DATA.mkdir(parents=True, exist_ok=True)
    with (OUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = json.loads((OUT / 'selection_private.json').read_text())
        completed = [r for r in selection if check_checkpoint(r, run_id)]
        if completed and not resume:
            raise RuntimeError('Existing results require explicit --resume')
        done_ids = {r['case_id'] for r in completed}
        pending = [r for r in selection if r['case_id'] not in done_ids]
        if limit:
            pending = pending[:limit]
        save(OUT / 'status.json', {'phase': 'running', 'completed': len(completed), 'total': len(selection),
                                  'run_id': run_id, 'resume': resume, 'workers': workers, 'pid': os.getpid()})
        if pending:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                     initializer=initializer, initargs=(run_id,)) as pool:
                futures = [pool.submit(process_case, row) for row in pending]
                for future in as_completed(futures):
                    record = future.result()
                    completed.append(record)
                    save(OUT / 'status.json', {'phase': 'running', 'completed': len(completed),
                                              'total': len(selection), 'run_id': run_id, 'last': record})
                    print(json.dumps(record), flush=True)
        save(OUT / 'status.json', {'phase': 'completed' if len(completed) == len(selection) else 'checkpointed',
                                  'completed': len(completed), 'total': len(selection), 'run_id': run_id})
        report()


def report():
    selection = json.loads((OUT / 'selection_private.json').read_text())
    records = [json.loads((DATA / row['case_id'] / 'result.json').read_text())
               for row in selection if (DATA / row['case_id'] / 'result.json').exists()]
    group_stats = {}
    review_rows = []
    manifest = []
    for group in ['regression48', 'random256']:
        group_rows = [r for r in records if r['sample_group'] == group]
        rois = [x for r in group_rows for x in r['yolo']]
        saved = [x for x in rois if x['status'] == 'saved']
        group_stats[group] = {
            'films': len(group_rows), 'technical_case_errors': sum(r['status'] != 'completed' for r in group_rows),
            'yolo_boxes': len(rois), 'yolo_saved_float_rois': len(saved),
            'yolo_two_box_films': sum(len(r['yolo']) == 2 for r in group_rows),
            'old_valid_rois': sum(len(r['old']) for r in group_rows),
            'old_two_roi_films': sum(len(r['old']) == 2 for r in group_rows),
            'yolo_rois_without_automatic_flags': sum(x['technical_qc'] == 'no_automatic_flags' for x in saved),
            'yolo_rois_for_review': sum(x['technical_qc'] == 'review_required' for x in saved),
            'yolo_extract_errors': sum(x['status'] != 'saved' for x in rois),
            'geometry_fallback_candidates': sum(len(r['fallback']) for r in group_rows),
            'missing_yolo_sides': sum(len(r.get('missing_positions', [])) for r in group_rows),
            'roi_flag_counts': dict(Counter(f for x in rois for f in x['flags'])),
            'film_flag_counts': dict(Counter(f for r in group_rows for f in r.get('film_flags', []))),
            'modality': dict(Counter(r['modality'] for r in group_rows)),
        }
    regression_mismatch = []
    for r in records:
        if r['sample_group'] == 'regression48' and r['status'] == 'completed':
            before = json.loads((PILOT / 'outputs/cases' / r['case_id'] / 'result.json').read_text())
            for method in ['yolo', 'old']:
                a = {x['image_position']: x['bbox'] for x in before[method]}
                b = {x['image_position']: x['bbox'] for x in r[method]}
                if a != b:
                    regression_mismatch.append({'case_id': r['case_id'], 'method': method})
        for pos, side in SIDES.items():
            item = next((x for x in r['yolo'] if x['image_position'] == pos), None)
            flags = [] if item is None else item['flags']
            review_rows.append({
                'case_id': r['case_id'], 'sample_group': r['sample_group'], 'modality': r['modality'],
                'visit': r['visit'], 'image_position': pos, 'clinical_side': side,
                'yolo_status': item['status'] if item else 'missing', 'flags': ';'.join(flags),
                'film_flags': ';'.join(r.get('film_flags', [])),
                'priority': 'flagged' if item is None or flags or item['status'] != 'saved' else 'routine',
                'gray01_npy': item.get('gray01_npy', '') if item else '',
                'review_panel': str(OUT / 'review_server_only' / (r['case_id'] + '.png')),
                'joint_space_visible': '', 'medial_lateral_margins_covered': '',
                'femur_tibia_covered': '', 'gray_detail_acceptable': '',
                'side_verified': '', 'decision': '', 'reviewer': '', 'review_time': '', 'notes': '',
            })
        for item in r['yolo']:
            if item['status'] == 'saved':
                manifest.append({**{k: r[k] for k in ['case_id', 'film_id', 'patient_id', 'split', 'visit', 'modality']},
                                 **item, 'source_archive_sha256': r['archive_sha256'],
                                 'case_evidence_path': str(DATA / r['case_id'] / 'result.json'),
                                 'side_policy': CONFIG['side_policy']})
    summary = {
        'total_completed_records': len(records), 'target_films': len(selection), 'groups': group_stats,
        'regression48_bbox_mismatches': regression_mismatch,
        'clinical_quality_approved': False, 'training_ready': False, 'human_reviews_completed': 0,
        'test_images_or_labels_opened': False, 'new_classification_training': False,
        'quality_note': 'automatic flags are review triggers; zero flags is not clinical approval',
        'interpretation': 'random256 samples train patients, not independent detector test subjects; no ground-truth IoU or KL improvement measured',
    }
    save(OUT / 'summary.json', summary)
    save(OUT / 'roi_candidates_private.json', manifest)
    if review_rows:
        write_csv(OUT / 'human_review_template.csv', review_rows, list(review_rows[0]))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'report'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'run':
        run(args.resume, args.limit, args.workers)
    else:
        report()


if __name__ == '__main__':
    main()
