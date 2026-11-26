#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

PROJECT = Path('/path/to/drai')
NAS = Path('/path/to/nas')
B = PROJECT / 'outputs/downstream/oarsi_patchattn_1p6'
EXP = PROJECT / 'experiments'
PROTOCOL = B / 'test_eval_v1/T1/test_protocol_draft_v2.json'
PROTOCOL_SHA256 = 'aa09e55834ba3cc043f93ddcfd243d49ffc1573400437c42ba8d6728a56070ea'
SPLIT = PROJECT / 'data/manifests/patient_split_v1.csv'
CANDIDATES = PROJECT / 'data/manifests/xray_knee_pa_candidates.csv'
FROZEN_SELECTION = EXP / 'chen_yolo_fullcrop_v1/outputs/selection_private.json'
BILATERAL = 'Bilateral PA Fixed Flexion Knee'
# Take only these non-label columns from the candidate table; KL, MOAKS and other disease columns are never read.
META = ('patient_id', 'visit', 'exam_date_iso', 'modality', 'dicom_modality_first_instance', 'description',
        'archive_relpath', 'archive_bytes', 'dicom_first_instance_rows', 'dicom_first_instance_columns')
CODE = {
    'crop13': [EXP / 'chen_yolo_fullcrop_v1/code', EXP / 'chen_yolo_repair_v1/code'],
    'exp448': [EXP / 'chen_yolo_448_v1/code', EXP / 'chen_yolo_repair_v1/code',
               EXP / 'chen_yolo_fullcrop_v1/code'],
    'clean2x': [EXP / 'chen_yolo_2x_clean_v1/code', EXP / 'chen_yolo_expand2x_v1/code',
                EXP / 'chen_yolo_expand_v1/code', EXP / 'chen_yolo_448_v1/code',
                EXP / 'chen_yolo_repair_v1/code', EXP / 'chen_yolo_fullcrop_v1/code'],
    'crop16': [EXP / 'chen_yolo_1p6x_clean_v1/code', EXP / 'chen_yolo_expand2x_v1/code',
               EXP / 'chen_yolo_expand_v1/code', EXP / 'chen_yolo_448_v1/code',
               EXP / 'chen_yolo_repair_v1/code', EXP / 'chen_yolo_fullcrop_v1/code'],
}
STAGES = ('admit', 'crop13', 'exp448', 'clean2x', 'crop16')
STATE = {}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
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
        json.dump(value, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def dirs(cohort):
    assert cohort in ('val_replay', 'test')
    return (B / 'test_eval_v1/prep' / cohort,
            PROJECT / 'data/processed/oarsi_test_eval_v1_20261117' / cohort)


def identity(stage):
    return digest({'protocol_sha256': PROTOCOL_SHA256, 'stage': stage, 'prep_code_sha256': sha(__file__)})


def films(cohort):
    out, _ = dirs(cohort)
    target = out / 'films_private.json'
    assert not target.exists(), 'film list exists; refusing to overwrite'
    split = {}
    with open(SPLIT, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            split[r['patient_id'].strip()] = r['split']
    meta = {}
    with open(CANDIDATES, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            pid = r['patient_id'].strip()
            if split.get(pid) == ('test' if cohort == 'test' else 'val'):
                meta[r['archive_relpath']] = {k: r[k] for k in META}
    rows = []
    if cohort == 'test':
        chosen = sorted(m for m in meta.values() if m['description'] == BILATERAL)
        chosen = sorted(chosen, key=lambda m: m['archive_relpath'])
        for i, m in enumerate(chosen, 1):
            rows.append({'case_id': f'X{i:06d}', 'film_id': hashlib.sha256(m['archive_relpath'].encode()).hexdigest()[:20],
                         'patient_id': m['patient_id'].strip(), 'split': 'test', 'visit': m['visit'],
                         'modality': m['dicom_modality_first_instance'], 'input_archive_relpath': m['archive_relpath'],
                         'archive_bytes': m['archive_bytes'], 'header_source': m})
    else:
        frozen = [r for r in json.load(open(FROZEN_SELECTION)) if r['split'] == 'val']
        for r in sorted(frozen, key=lambda r: r['case_id']):
            m = meta[r['input_archive_relpath']]
            rows.append({'case_id': r['case_id'], 'film_id': r['film_id'], 'patient_id': r['patient_id'], 'split': 'val',
                         'visit': r['visit'], 'modality': m['dicom_modality_first_instance'],
                         'input_archive_relpath': r['input_archive_relpath'], 'archive_bytes': m['archive_bytes'],
                         'header_source': m,
                         'frozen_reference': {k: r[k] for k in ('archive_sha256', 'native_pixel_sha256', 'modality', 'archive_bytes')}})
    save(target, rows)
    counts = {'films': len(rows), 'patients': len({r['patient_id'] for r in rows}), 'by_visit': dict(Counter(r['visit'] for r in rows))}
    save(out / 'films_summary.json', {**counts, 'films_private_sha256': sha(target), 'cohort': cohort})
    print(json.dumps(counts, ensure_ascii=False))


def init(stage, cohort):
    out, data = dirs(cohort)
    sys.path[:0] = [str(PROJECT), str(PROJECT / 'src')] + [str(p) for p in CODE.get(stage, [])]
    import torch
    if stage != 'crop13':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    STATE.update(stage=stage, cohort=cohort, out=out, data=data, ident=identity(stage))
    if stage == 'crop13':
        import pipeline as p
        p.DATA = data / 'crop13'
        p.OUT = out / 'crop13_frozen_out_unused'
        p.CONFIG['scope'] = ['train', 'val', 'test']
        p.save_review_panel = lambda *a, **k: None
        p.initializer(STATE['ident'])
        import run_stream_v2
        import run_supplement_v2
        STATE.update(p=p, eligible=run_stream_v2.eligible, reasons=run_supplement_v2.reasons)
        from verify_candidates import roi_check
        STATE['roi_check'] = roi_check
    elif stage == 'exp448':
        import export_448 as e
        e.DATA = data / 'exp448'
        STATE['e'] = e
    elif stage == 'clean2x':
        import crop_full as crop
        import audit_rulers as a
        import run_audit
        import proposal_v1
        import clean_stage
        import edge_finish
        root = out / 'clean2x'
        crop.DATA = data / 'clean2x/raw'
        a.ROOT = root
        run_audit.AUDIT = root / 'audit_v2'
        proposal_v1.OUT = root / 'proposals_v1'
        clean_stage.DATA = data / 'clean2x'
        STATE.update(crop=crop, a=a, run_audit=run_audit, proposal=proposal_v1, clean=clean_stage, edge=edge_finish, root=root)
    elif stage == 'crop16':
        import crop_full as crop
        import audit_rulers as a
        import clean_stage
        import pipeline16
        root = out / 'crop16'
        crop.DATA = data / 'crop16/raw'
        a.ROOT = root
        clean_stage.DATA = data / 'crop16'
        pipeline16.ROOT = root
        STATE.update(crop=crop, pipe=pipeline16, root=root)


def admit_one(row):
    from drai.data.dicom_gray import read_pa_archive, presentation_values, normalize_roi
    from drai.data.full_pa import letterbox
    from downstream.preprocess.prepare_baseline_qc_v1 import header_checks
    import numpy as np
    result = {'case_id': row['case_id'], 'status': 'failed_technical', 'error': None}
    try:
        ds, pixels, se = read_pa_archive(NAS / row['input_archive_relpath'], row['archive_bytes'], source_root=NAS)
        header, warnings = header_checks(ds, row['header_source'], row['patient_id'])
        values, valid, display = presentation_values(pixels, ds)
        gray, statistics = normalize_roi(values, valid)
        cached, mask, box = letterbox(gray, valid, 512)
        if not mask.any() or not np.isfinite(cached).all() or cached[mask].std() < 0.01:
            raise ValueError('gray degenerate after resizing the full canvas')
        result.update(status='passed', archive_sha256=se['archive_sha256'], native_pixel_sha256=se['native_pixel_sha256'],
                      dicom_modality=str(ds.Modality), warnings=warnings)
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
    return result


def crop13_one(row):
    p = STATE['p']
    p.process_case(row)
    path = p.DATA / row['case_id'] / 'result.json'
    result = json.loads(path.read_text())
    record = {'case_id': row['case_id'], 'status': result['status'], 'result_path': str(path), 'result_sha256': sha(path)}
    if result['status'] != 'completed':
        record.update(automatic_pass=False, triggers=['technical_failure'])
        return record
    triggers = []
    if STATE['eligible'](result):
        triggers.append('eligible_missing_or_flagged')
    triggers += STATE['reasons'](result)
    saved = [x for x in result['yolo'] if x['status'] == 'saved']
    record.update(automatic_pass=not triggers, triggers=triggers,
                  flags={x['image_position']: x['flags'] for x in saved})
    if triggers:
        return record
    items = []
    for pos, side in p.SIDES.items():
        roi = next(x for x in saved if x['image_position'] == pos)
        assert not (set(roi['flags']) - {'detector_center_padding'})
        assert STATE['roi_check'](roi, result, row)
        items.append({'key': row['case_id'] + '_' + side, 'film_id': row['film_id'], 'patient_id': row['patient_id'],
                      'split': row['split'], 'visit': row['visit'], 'modality': row['modality'], 'case_id': row['case_id'],
                      'archive_sha256': row['archive_sha256'], 'native_pixel_sha256': row['native_pixel_sha256'],
                      'image_position': pos, 'clinical_side': side, 'kl': None,
                      'side_policy': 'image_left=R,image_right=L; no spatial flip or metadata side inference',
                      'source_result': str(path), 'source_result_sha256': record['result_sha256'],
                      'candidate_family': 'base', 'variant': 'original', 'bbox': roi['bbox'],
                      'gray01_npy': roi['gray01_npy'], 'gray01_sha256': roi['gray01_sha256'],
                      'valid_mask_npy': roi['valid_mask_npy'], 'valid_mask_sha256': roi['valid_mask_sha256'],
                      'shape': [1, 224, 224], 'original_flags': roi['flags'], 'selected_flags': roi['flags'],
                      'review': {'kind': 'automatic_only_test_protocol_v2', 'reason': 'no repair trigger'},
                      'engineering_input_ready': True, 'clinical_quality_approved': False,
                      'individual_anatomical_side_verified': False})
    record['items'] = items
    return record


def exp448_one(task):
    e = STATE['e']
    e.process({**task, 'identity': STATE['ident']})
    path = e.DATA / task['row']['case_id'] / 'result.json'
    r = json.loads(path.read_text())
    record = {'case_id': task['row']['case_id'], 'status': r['status'], 'result_path': str(path), 'result_sha256': sha(path)}
    if r['status'] == 'completed':
        record['items'] = [{**x, 'source_result': str(path), 'source_result_sha256': record['result_sha256']} for x in r['entries']]
    else:
        record['error'] = r.get('error')
    return record


def clean2x_one(task):
    S = STATE
    case = task['row']['case_id']
    record = {'case_id': case, 'status': 'failed'}
    try:
        r = S['crop'].process({**task, 'identity': S['ident']})
        raw = S['crop'].DATA / case / 'result.json'
        assert r['status'] == 'completed', json.loads(raw.read_text()).get('error')
        S['run_audit'].worker(str(raw))
        S['proposal'].worker(str(raw))
        proposal = S['proposal'].OUT / 'cases' / case / 'result.json'
        S['clean'].worker((str(proposal), 'clean_v1', S['ident'], {}))
        S['edge'].worker((str(S['root'] / 'clean_v1/cases' / (case + '.json')), 'edge_finish_v1', S['ident']))
        final = S['root'] / 'edge_finish_v1/cases' / (case + '.json')
        entries = json.loads(final.read_text())['entries']
        record.update(status='completed', final_path=str(final), final_sha256=sha(final),
                      prior_final={x['key']: {k: x[k] for k in ('key', 'engineering_input_ready', 'cleanup_status', 'cleanup_review_reasons')}
                                   for x in entries})
    except Exception as exc:
        record.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-2000:])
    return record


def crop16_one(task):
    S = STATE
    case = task['row']['case_id']
    record = {'case_id': case, 'status': 'failed'}
    try:
        S['pipe'].process({**task, 'identity': S['ident'] + ':crop', 'pipeline_identity': S['ident']})
        raw = S['crop'].DATA / case / 'result.json'
        final = S['root'] / 'edge_finish_v1/cases' / (case + '.json')
        record.update(status='completed', raw_path=str(raw), raw_sha256=sha(raw), final_path=str(final), final_sha256=sha(final))
    except Exception as exc:
        record.update(error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()[-2000:])
    return record


WORKERS = {'admit': admit_one, 'crop13': crop13_one, 'exp448': exp448_one, 'clean2x': clean2x_one, 'crop16': crop16_one}


def call(stage, task):
    try:
        return WORKERS[stage](task)
    except Exception as exc:
        case = task['case_id'] if 'case_id' in task else task['row']['case_id']
        return {'case_id': case, 'status': 'worker_error', 'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc()[-2000:]}


def load_receipts(out, stage):
    folder = out / stage
    return {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob('*.json'))} if folder.exists() else {}


def tasks_for(stage, cohort):
    out, _ = dirs(cohort)
    rows = {r['case_id']: r for r in json.load(open(out / 'films_private.json'))}
    if stage == 'admit':
        return list(rows.values())
    admit = load_receipts(out, 'admit')
    passed = [r for r in admit.values() if r['status'] == 'passed']
    groups = defaultdict(list)
    for r in passed:
        groups[r['native_pixel_sha256']].append(r['case_id'])
    duplicated = {c for g in groups.values() if len(g) > 1 for c in g}
    base_rows = {}
    for r in passed:
        if r['case_id'] in duplicated:
            continue
        row = {k: v for k, v in rows[r['case_id']].items() if k not in ('header_source', 'frozen_reference')}
        row.update(archive_sha256=r['archive_sha256'], native_pixel_sha256=r['native_pixel_sha256'],
                   labels={'R': None, 'L': None})
        base_rows[r['case_id']] = row
    if stage == 'crop13':
        return list(base_rows.values())
    c13 = load_receipts(out, 'crop13')
    ok = {c: r for c, r in c13.items() if r.get('automatic_pass')}
    if stage == 'exp448':
        return [{'row': base_rows[c], 'items': r['items']} for c, r in sorted(ok.items())]
    e448 = {c: r for c, r in load_receipts(out, 'exp448').items() if r['status'] == 'completed'}
    if stage == 'clean2x':
        return [{'row': base_rows[c], 'items': r['items']} for c, r in sorted(e448.items())]
    c2 = {c: r for c, r in load_receipts(out, 'clean2x').items() if r['status'] == 'completed'}
    return [{'row': base_rows[c], 'items': e448[c]['items'], 'prior_case_path': r['final_path'],
             'prior_case_sha256': r['final_sha256'], 'prior_final': r['prior_final']} for c, r in sorted(c2.items())]


def run(stage, cohort, workers, limit, only):
    assert sha(PROTOCOL) == PROTOCOL_SHA256, 'protocol file changed'
    assert json.load(open(PROTOCOL.parent / 'authorization_T1.json'))['approved_file_sha256'] == PROTOCOL_SHA256
    out, data = dirs(cohort)
    folder = out / stage
    folder.mkdir(parents=True, exist_ok=True)
    done = set(load_receipts(out, stage))
    todo = [t for t in tasks_for(stage, cohort) if (t.get('case_id') or t['row']['case_id']) not in done]
    if only:
        wanted = set(only.split(','))
        todo = [t for t in todo if (t.get('case_id') or t['row']['case_id']) in wanted]
    if limit:
        todo = todo[:limit]
    save(out / f'{stage}_status.json', {'phase': 'running', 'todo': len(todo), 'already_done': len(done),
                                        'identity': identity(stage), 'started': time.strftime('%F %T %Z')})
    counts, started = Counter(), time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'), initializer=init,
                             initargs=(stage, cohort), max_tasks_per_child=25) as pool:
        pending = iter(todo)
        inflight = {pool.submit(call, stage, t) for t in [next(pending, None) for _ in range(workers)] if t is not None}
        n = 0
        while inflight:
            ready, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for future in ready:
                r = future.result()
                save(folder / (r['case_id'] + '.json'), r)
                counts[r['status']] += 1
                n += 1
                t = next(pending, None)
                if t is not None:
                    inflight.add(pool.submit(call, stage, t))
                if n % 20 == 0 or n == len(todo):
                    save(out / f'{stage}_status.json', {'phase': 'running', 'done_this_run': n, 'todo': len(todo),
                                                        'counts': dict(counts), 'seconds': time.time() - started})
    final = {'phase': 'finished', 'done_this_run': n, 'todo': len(todo), 'counts': dict(counts),
             'seconds': time.time() - started, 'identity': identity(stage), 'finished': time.strftime('%F %T %Z')}
    save(out / f'{stage}_status.json', final)
    print(json.dumps(final, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['films', 'run'])
    ap.add_argument('--cohort', required=True, choices=['val_replay', 'test'])
    ap.add_argument('--stage', choices=STAGES)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    if a.cohort == 'test':
        replay = B / 'test_eval_v1/prep/val_replay/gate.json'
        assert replay.exists() and json.load(open(replay))['passed'] is True, 'validation replay gate failed; test not touched'
    if a.command == 'films':
        films(a.cohort)
    else:
        run(a.stage, a.cohort, a.workers, a.limit, a.only)


if __name__ == '__main__':
    main()
