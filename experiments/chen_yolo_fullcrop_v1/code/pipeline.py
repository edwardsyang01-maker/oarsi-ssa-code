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
DATA = PROJECT / 'data/processed/chen_yolo_fullcrop_v1'
PRIOR = PROJECT / 'experiments/chen_yolo_roi_v2'
NAS = Path('/path/to/nas')
PREP = PROJECT / 'outputs/downstream/oa_kl_wholefilm_v2/preparation_20261025'
SPLIT = PROJECT / 'data/manifests/patient_split_v1.csv'
MODEL_PATH = PILOT / 'assets/chen_yolov2_1wd.pth'
MODEL_SHA = 'daa658b52e41ea7ad1493fc97ed34480d5dfb152e47c0bd296b187af2ceae95a'
sys.path[:0] = [str(PROJECT), str(PROJECT / 'src'), str(PILOT / 'code')]
from compare_crops import author_input, backmap, tensor_from_u8, preview_crop

SIDES = {'image_left': 'R', 'image_right': 'L'}
CONFIG = {
    'schema': 'chen-yolo-fullcrop-v1', 'seed': 'chen-yolo-fullcrop-20261111-v1',
    'scope': ['train', 'val'], 'detector_retraining': False, 'score_threshold': .12, 'nms_threshold': .3,
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
             PILOT / 'outputs/selection_private.json',
             PRIOR / 'outputs/identity.json', PRIOR / 'outputs/cohort_private.json',
             PRIOR / 'outputs/selection_private.json', PRIOR / 'code/pipeline.py',
             ROOT / 'code/verify_outputs.py', ROOT / 'code/make_review.py',
             ROOT / 'code/run_full.py']
    return {str(p): sha(p) for p in paths}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'identity.json').exists():
        verify_identity()
        print('Existing immutable full-cohort preparation verified', flush=True)
        return
    patients = {r['patient_id']: r['split'] for r in read_csv(SPLIT)}
    cohort = json.loads((PRIOR / 'outputs/cohort_private.json').read_text())
    frozen = json.loads((PRIOR / 'outputs/identity.json').read_text())
    if sha(PRIOR / 'outputs/cohort_private.json') != frozen['cohort_sha256']:
        raise RuntimeError('Prior frozen cohort changed')
    current_sources = sources()
    for path, expected in frozen['sources'].items():
        if sha(Path(path)) != expected:
            raise RuntimeError('Prior frozen dependency changed: ' + path)
    if current_sources[str(MODEL_PATH)] != MODEL_SHA:
        raise RuntimeError('Author detector fingerprint mismatch')
    by_split = defaultdict(set)
    keys = set()
    prior_rows = json.loads((PRIOR / 'outputs/selection_private.json').read_text())
    prior_by_film = {r['film_id']: r for r in prior_rows}
    selection = []
    for split, prefix in [('train', 'T'), ('val', 'V')]:
        rows = sorted((r for r in cohort if r['split'] == split), key=lambda r: r['film_id'])
        for index, row in enumerate(rows, 1):
            if patients.get(row['patient_id']) != split:
                raise RuntimeError('Patient boundary differs from frozen registry')
            if row['film_id'] in keys or any(v not in range(5) for v in row['labels'].values()):
                raise RuntimeError('Duplicate film or invalid ipsilateral label')
            keys.add(row['film_id'])
            by_split[split].add(row['patient_id'])
            item = dict(row)
            item.update(case_id=f'{prefix}{index:06d}', sample_group=split)
            if row['film_id'] in prior_by_film:
                item['prior_case_id'] = prior_by_film[row['film_id']]['case_id']
            selection.append(item)
    if len(selection) != len(cohort) or by_split['train'] & by_split['val']:
        raise RuntimeError('Unsupported split or overlapping patients')
    if dict(Counter(r['split'] for r in selection)) != {'train': 17133, 'val': 2131}:
        raise RuntimeError('Expected frozen cohort counts changed')
    smoke_ids = set()
    for old_id in ['C05', 'C10', 'C30', 'C31', 'C38', 'C39']:
        smoke_ids.update(r['case_id'] for r in selection if r.get('prior_case_id') == old_id)
    for modality in sorted({r['modality'] for r in selection}):
        known = [r for r in selection if r.get('prior_case_id') and r['modality'] == modality]
        held = [r for r in selection if r['split'] == 'val' and r['modality'] == modality]
        for rows in [known, held]:
            smoke_ids.update(r['case_id'] for r in sorted(rows, key=lambda r: digest([CONFIG['seed'], r['film_id']]))[:4])
    save(OUT / 'cohort_private.json', cohort)
    save(OUT / 'selection_private.json', selection)
    save(OUT / 'smoke_selection_private.json', sorted(smoke_ids))
    save(OUT / 'config.json', CONFIG)
    identity = {
        'config': CONFIG, 'sources': current_sources,
        'cohort_sha256': sha(OUT / 'cohort_private.json'),
        'selection_sha256': sha(OUT / 'selection_private.json'),
        'smoke_selection_sha256': sha(OUT / 'smoke_selection_private.json'),
    }
    identity['run_id'] = digest(identity)
    save(OUT / 'identity.json', identity)
    summary = {
        'cohort_films': dict(Counter(r['split'] for r in selection)),
        'cohort_patients': {k: len(v) for k, v in by_split.items()},
        'potential_knees_before_detection_qc': dict(Counter(r['split'] for r in selection for _ in range(2))),
        'patient_split_disjoint': True, 'prior_regression_films': len(prior_by_film),
        'smoke_films': len(smoke_ids), 'model_sha256': MODEL_SHA,
        'clinical_quality_approved': False, 'training_ready': False,
        'detector_retraining': False, 'new_classification_training': False,
        'test_images_or_labels_opened': False,
        'side_assignment': 'PA image-position assumption; individual review remains pending',
        'overlap_limitation': 'Known author public detector patient overlap; not a patient-independent end-to-end evaluation',
    }
    save(OUT / 'preparation_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def verify_identity():
    identity = json.loads((OUT / 'identity.json').read_text())
    check = dict(identity)
    expected = check.pop('run_id')
    if digest(check) != expected or identity['config'] != CONFIG:
        raise RuntimeError('Invalid or changed run identity')
    if identity['sources'] != sources():
        raise RuntimeError('A frozen source changed; create a new experiment version')
    for name, key in [('cohort_private.json', 'cohort_sha256'), ('selection_private.json', 'selection_sha256'),
                      ('smoke_selection_private.json', 'smoke_selection_sha256')]:
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
    if row['split'] not in CONFIG['scope']:
        raise RuntimeError('This frozen cropping stage excludes test images')
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
    if result['run_id'] != run_id or any(result[k] != row[k] for k in
            ['case_id', 'film_id', 'patient_id', 'split', 'labels', 'archive_sha256', 'native_pixel_sha256']):
        raise RuntimeError('Checkpoint identity mismatch')
    for roi in result['yolo'] + result['fallback']:
        if roi['status'] == 'saved':
            for field in ['gray01', 'valid_mask']:
                if sha(roi[field + '_npy']) != roi[field + '_sha256']:
                    raise RuntimeError('Checkpoint output hash mismatch')
    return True


def run(resume, limit, workers, smoke=False):
    run_id = verify_identity()
    authorization = json.loads((OUT / 'cropping_authorization.json').read_text())
    if authorization.get('approved') is not True or authorization.get('run_id') != run_id:
        raise PermissionError('Cropping authorization does not match this frozen run')
    DATA.mkdir(parents=True, exist_ok=True)
    with (OUT / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selection = json.loads((OUT / 'selection_private.json').read_text())
        completed = [r for r in selection if check_checkpoint(r, run_id)]
        if completed and not resume:
            raise RuntimeError('Existing results require explicit --resume')
        done_ids = {r['case_id'] for r in completed}
        pending = [r for r in selection if r['case_id'] not in done_ids]
        if smoke:
            smoke_ids = set(json.loads((OUT / 'smoke_selection_private.json').read_text()))
            pending = [r for r in pending if r['case_id'] in smoke_ids]
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
                                              'total': len(selection), 'run_id': run_id, 'last': record,
                                              'workers': workers, 'pid': os.getpid(), 'updated_unix': time.time()})
                    print(json.dumps(record), flush=True)
        save(OUT / 'status.json', {'phase': 'completed' if len(completed) == len(selection) else 'checkpointed',
                                  'completed': len(completed), 'total': len(selection), 'run_id': run_id})
        report()


def report():
    run_id = verify_identity()
    selection = json.loads((OUT / 'selection_private.json').read_text())
    records = []
    for row in selection:
        path = DATA / row['case_id'] / 'result.json'
        if path.exists():
            record = json.loads(path.read_text())
            if record['run_id'] != run_id or record['film_id'] != row['film_id']:
                raise RuntimeError('Report encountered an unrelated checkpoint')
            records.append(record)
    stats, manifest, fallback_manifest, review_rows, issues = {}, [], [], [], []
    regression_checked, regression_mismatches = 0, []
    all_fields = ['sample_id', 'case_id', 'film_id', 'patient_id', 'split', 'visit', 'modality',
                  'image_position', 'clinical_side', 'kl', 'method', 'gray01_npy', 'gray01_sha256',
                  'valid_mask_npy', 'valid_mask_sha256', 'bbox_json', 'transform_json', 'score',
                  'technical_qc', 'roi_flags', 'film_flags', 'source_archive_sha256',
                  'case_evidence_path', 'side_policy', 'side_assignment_status',
                  'training_ready', 'clinical_quality_approved']
    for split in CONFIG['scope']:
        group = [r for r in records if r['split'] == split]
        rois = [x for r in group for x in r['yolo']]
        saved = [x for x in rois if x['status'] == 'saved']
        missing = sum(2-len({x['image_position'] for x in r['yolo'] if x['status']=='saved'}) for r in group)
        stats[split] = {
            'target_films': sum(r['split'] == split for r in selection),
            'processed_films': len(group), 'processed_patients': len({r['patient_id'] for r in group}),
            'successful_films': sum(r['status'] == 'completed' for r in group),
            'technical_case_errors': sum(r['status'] != 'completed' for r in group),
            'yolo_boxes': len(rois), 'yolo_saved_float_rois': len(saved),
            'yolo_two_saved_roi_films': sum(sum(x['status']=='saved' for x in r['yolo'])==2 for r in group),
            'yolo_one_saved_roi_films': sum(sum(x['status']=='saved' for x in r['yolo'])==1 for r in group),
            'yolo_zero_saved_roi_films': sum(not any(x['status']=='saved' for x in r['yolo']) for r in group),
            'patients_with_yolo_roi': len({r['patient_id'] for r in group if any(x['status']=='saved' for x in r['yolo'])}),
            'yolo_rois_without_automatic_flags': sum(x['technical_qc']=='no_automatic_flags' for x in saved),
            'yolo_rois_for_review': sum(x['technical_qc']=='review_required' for x in saved),
            'yolo_extract_errors': sum(x['status']!='saved' for x in rois),
            'missing_or_failed_yolo_sides': missing,
            'old_valid_rois': sum(len(r['old']) for r in group),
            'geometry_fallback_candidates': sum(sum(x['status']=='saved' for x in r['fallback']) for r in group),
            'roi_flag_counts': dict(Counter(f for x in rois for f in x['flags'])),
            'film_flag_counts': dict(Counter(f for r in group for f in r.get('film_flags',[]))),
            'modality': dict(Counter(r['modality'] for r in group)),
        }
    for r in records:
        if r.get('prior_case_id'):
            before_path = PROJECT/'data/processed/chen_yolo_roi_v2'/r['prior_case_id']/'result.json'
            before = json.loads(before_path.read_text())
            regression_checked += 1
            for method in ['yolo', 'old', 'fallback']:
                fields = ['bbox'] if method == 'old' else ['bbox', 'status', 'flags', 'gray01_sha256', 'valid_mask_sha256']
                a = {x['image_position']: {k:x.get(k) for k in fields} for x in before[method]}
                b = {x['image_position']: {k:x.get(k) for k in fields} for x in r[method]}
                if a != b:
                    regression_mismatches.append({'case_id':r['case_id'],'method':method})
        for pos, side in SIDES.items():
            item = next((x for x in r['yolo'] if x['image_position']==pos),None)
            flags = [] if item is None else item['flags']
            priority = 'flagged' if item is None or flags or item['status']!='saved' else 'routine'
            review_rows.append({
                'case_id':r['case_id'], 'sample_group':r['split'], 'modality':r['modality'], 'visit':r['visit'],
                'image_position':pos, 'clinical_side':side, 'yolo_status':item['status'] if item else 'missing',
                'flags':';'.join(flags), 'film_flags':';'.join(r.get('film_flags',[])), 'priority':priority,
                'gray01_npy':item.get('gray01_npy','') if item else '',
                'review_panel':str(OUT/'review_server_only'/(r['case_id']+'.png')),
                'joint_space_visible':'', 'medial_lateral_margins_covered':'', 'femur_tibia_covered':'',
                'gray_detail_acceptable':'', 'side_verified':'', 'decision':'', 'reviewer':'',
                'review_time':'', 'notes':'',
            })
            if priority == 'flagged':
                issues.append({'case_id':r['case_id'],'split':r['split'],'image_position':pos,
                               'case_status':r['status'],'yolo_status':item['status'] if item else 'missing',
                               'flags':';'.join(flags),'error':r.get('error',item.get('error','') if item else '')})
        for method, target in [('yolo',manifest),('fallback',fallback_manifest)]:
            for item in r[method]:
                if item['status']!='saved':
                    continue
                target.append({
                    **{k:r[k] for k in ['case_id','film_id','patient_id','split','visit','modality']},
                    **{k:item[k] for k in ['image_position','clinical_side','kl','method','gray01_npy',
                         'gray01_sha256','valid_mask_npy','valid_mask_sha256','technical_qc']},
                    'sample_id':r['film_id']+'_'+item['clinical_side'],
                    'bbox_json':json.dumps(item['bbox']), 'transform_json':json.dumps(item['transform']),
                    'score':item.get('score',''), 'roi_flags':';'.join(item['flags']),
                    'film_flags':';'.join(r.get('film_flags',[])), 'source_archive_sha256':r['archive_sha256'],
                    'case_evidence_path':str(DATA/r['case_id']/'result.json'), 'side_policy':CONFIG['side_policy'],
                    'side_assignment_status':'position_assumption_pending_individual_review',
                    'training_ready':0, 'clinical_quality_approved':0,
                })
    for split in CONFIG['scope']:
        rows = [x for x in manifest if x['split']==split]
        write_csv(OUT/f'yolo_candidates_{split}.csv',rows,all_fields)
        write_csv(OUT/f'yolo_no_auto_flags_{split}.csv',[x for x in rows if not x['roi_flags']],all_fields)
        write_csv(OUT/f'geometry_fallback_{split}.csv',[x for x in fallback_manifest if x['split']==split],all_fields)
    if review_rows:
        write_csv(OUT/'human_review_template.csv',review_rows,list(review_rows[0]))
    if issues:
        write_csv(OUT/'issues_private.csv',issues,list(issues[0]))
    summary = {
        'run_id':run_id, 'target_films':len(selection), 'processed_films':len(records),
        'all_candidates_complete':len(records)==len(selection), 'groups':stats,
        'prior304_regression_checked':regression_checked, 'prior304_regression_mismatches':regression_mismatches,
        'human_review_rows':len(review_rows), 'human_reviews_completed':0,
        'clinical_quality_approved':False, 'training_ready':False,
        'test_images_or_labels_opened':False, 'new_classification_training':False, 'detector_retraining':False,
        'quality_note':'Saved/no-auto-flag ROI counts are technical candidates, not clinically approved usable counts',
        'overlap_limitation':'Known author public detector patient overlap; not independent end-to-end performance evidence',
    }
    save(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'run', 'report'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, choices=[1, 2, 4, 8, 12, 16], default=2)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'run':
        run(args.resume, args.limit, args.workers, args.smoke)
    else:
        report()


if __name__ == '__main__':
    main()

