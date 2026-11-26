#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path('/path/to/drai')
B = PROJECT / 'outputs/downstream/oarsi_patchattn_1p6'
OUT = B / 'test_eval_v1/prep/test'
MANIFEST = PROJECT / 'data/manifests/oarsi_test_eval_v1_20261117'
LOCK = OUT / 'test_images_locked_private.jsonl'
TRAINVAL = PROJECT / 'data/manifests/chen_yolo_usable448_1p6x_clean_v1_20261112'
OARSI_CODE = PROJECT / 'experiments/oarsi_aux_labels_v1/code'
VISITS = ('V00', 'V01', 'V03', 'V05', 'V06', 'V08', 'V10')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    assert not path.exists(), f'refusing to overwrite: {path}'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n')


def receipts(stage):
    return {p.stem: json.loads(p.read_text()) for p in sorted((OUT / stage).glob('*.json'))}


def lock():
    assert not LOCK.exists(), 'row set is locked; refusing to overwrite'
    films = {r['case_id']: r for r in json.load(open(OUT / 'films_private.json'))}
    stages = {s: receipts(s) for s in ('admit', 'crop13', 'exp448', 'clean2x', 'crop16')}
    for s in ('admit', 'crop13', 'exp448', 'clean2x', 'crop16'):
        status = json.load(open(OUT / f'{s}_status.json'))
        assert status['phase'] == 'finished', f'{s} did not finish'
    fate = {}
    groups = defaultdict(list)
    for c, r in stages['admit'].items():
        if r['status'] == 'passed':
            groups[r['native_pixel_sha256']].append(c)
    dup = {c for g in groups.values() if len(g) > 1 for c in g}
    for c in films:
        a = stages['admit'].get(c)
        if a is None:
            fate[c] = 'admit_missing'
        elif a['status'] != 'passed':
            fate[c] = 'technical_failure'
        elif c in dup:
            fate[c] = 'native_pixel_duplicate'
        elif stages['crop13'].get(c, {}).get('status') != 'completed':
            fate[c] = 'crop13_failure'
        elif not stages['crop13'][c].get('automatic_pass'):
            fate[c] = 'repair_trigger_not_automatic'
        elif stages['exp448'].get(c, {}).get('status') != 'completed':
            fate[c] = 'export448_failure'
        elif stages['clean2x'].get(c, {}).get('status') != 'completed':
            fate[c] = 'clean2x_failure'
        elif stages['crop16'].get(c, {}).get('status') != 'completed':
            fate[c] = 'crop16_failure'
        else:
            fate[c] = 'image_steps_completed'
    tv_native, tv_gray = set(), set()
    for name in ('train_usable', 'train_held', 'val_usable', 'val_held'):
        for line in open(TRAINVAL / f'{name}.jsonl'):
            r = json.loads(line)
            tv_native.add(r['native_pixel_sha256'])
            tv_gray.add(r['gray01_sha256'])
    rows, cross = [], Counter()
    for c in sorted(c for c, f in fate.items() if f == 'image_steps_completed'):
        rec = stages['crop16'][c]
        raw_path, final_path = Path(rec['raw_path']), Path(rec['final_path'])
        assert sha(raw_path) == rec['raw_sha256'] and sha(final_path) == rec['final_sha256']
        for e in json.load(open(final_path))['entries']:
            assert e['kl'] is None and e['split'] == 'test'
            for f in ('gray01', 'valid_mask'):
                assert sha(e[f + '_npy']) == e[f + '_sha256']
            if e['native_pixel_sha256'] in tv_native:
                cross['native_pixel_in_trainval'] += 1
            if e['gray01_sha256'] in tv_gray:
                cross['final_array_in_trainval'] += 1
                e = {**e, 'engineering_input_ready': False, 'cleanup_status': 'held_cross_split_duplicate'}
            rows.append({**e, 'source_result': str(raw_path), 'source_result_sha256': rec['raw_sha256'],
                         'final_cleanup_result': str(final_path), 'final_cleanup_result_sha256': rec['final_sha256']})
    LOCK.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    summary = {
        'films': len(films), 'film_fate': dict(Counter(fate.values())),
        'knees_locked': len(rows), 'knees_usable': sum(r['engineering_input_ready'] for r in rows),
        'knees_held': sum(not r['engineering_input_ready'] for r in rows),
        'cleanup_status': dict(Counter(r['cleanup_status'] for r in rows)),
        'patients_locked': len({r['patient_id'] for r in rows}),
        'cross_split_checks': dict(cross),
        'repair_trigger_counts': dict(Counter(t for r in stages['crop13'].values() for t in r.get('triggers', []))),
        'locked_rows_sha256': sha(LOCK), 'labels_read': False,
    }
    write_json(OUT / 'image_lock.json', summary)
    write_json(OUT / 'film_fate_private.json', fate)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def labels():
    info = json.load(open(OUT / 'image_lock.json'))
    assert sha(LOCK) == info['locked_rows_sha256'], 'locked row set changed'
    assert not MANIFEST.exists(), 'list directory exists; refusing to overwrite'
    rows = [json.loads(l) for l in open(LOCK)]
    films = {r['case_id']: r for r in json.load(open(OUT / 'films_private.json'))}
    sys.path[:0] = [str(PROJECT), str(PROJECT / 'src')]
    from downstream.oa_kl_wholefilm_v2 import labels as L
    import pandas as pd
    splits = L.read_splits(PROJECT / 'data/manifests/patient_split_v1.csv')
    test = {p for p, s in splits.items() if s == 'test'}
    images = pd.read_csv(PROJECT / 'data/manifests/xray_knee_pa_candidates.csv', encoding='utf-8-sig', dtype=str,
                         keep_default_na=False, usecols=L.META_COLUMNS)
    images = images[(images.sequence_family == 'Bilateral PA Fixed Flexion Knee') & (images.file_nonempty == '1')]
    as_val = {p: 'val' for p in test}  # Only to call the frozen build_visit; the split is set back to test afterwards.
    kl_by_relpath, reasons = {}, Counter()
    source_sha = {}
    for visit in VISITS:
        table = L.TABLE_ROOT / ('KXR_SQ_BU' + visit[1:] + '.txt')
        source_sha[table.name] = L.sha256(table)
        frame = images[(images.visit == visit) & images.patient_id.isin(test)]
        records, _ = L.read_table(table, visit, test)
        out, counts = L.build_visit(frame.to_dict('records'), records, as_val, visit)
        reasons.update(counts['val']['exclusion_reasons'])
        for r in out:
            kl_by_relpath[r['input_archive_relpath']] = (int(r['kl_right']), int(r['kl_left']), r['read_project_right'])
    usable, held, excluded = [], [], []
    for r in rows:
        relpath = films[r['case_id']]['input_archive_relpath']
        if relpath not in kl_by_relpath:
            excluded.append({'key': r['key'], 'case_id': r['case_id'], 'reason': 'kl_link_rule_excluded_film'})
            continue
        right, left, project = kl_by_relpath[relpath]
        e = {**r, 'kl': right if r['clinical_side'] == 'R' else left, 'read_project': project}
        (usable if e['engineering_input_ready'] else held).append(e)
    MANIFEST.mkdir(parents=True)
    for name, items in (('test_usable.jsonl', usable), ('test_held.jsonl', held)):
        (MANIFEST / name).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in items))
    write_json(MANIFEST / 'test_excluded_private.json', excluded)
    sys.path.insert(0, str(OARSI_CODE))
    import build_oarsi_labels_v2 as O
    O.SOURCE = MANIFEST
    tables = {v: O.load_visit(v) for v in O.PROJECT_BY_VISIT}
    oarsi, ocounts = O.build_split('test', tables)
    (MANIFEST / 'test_oarsi.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in oarsi))
    kl = Counter(e['kl'] for e in usable)
    held_kl = Counter(e['kl'] for e in held)
    summary = {
        'image_lock_sha256': info['locked_rows_sha256'], 'image_lock_summary': info,
        'kl_link_exclusion_reasons_films': dict(reasons),
        'knees': {'usable': len(usable), 'held': len(held), 'excluded_by_kl_link': len(excluded)},
        'patients_usable': len({e['patient_id'] for e in usable}),
        'films_usable_any_knee': len({e['case_id'] for e in usable}),
        'usable_kl_counts': {str(k): kl.get(k, 0) for k in range(5)},
        'held_kl_counts': {str(k): held_kl.get(k, 0) for k in range(5)},
        'held_fraction_by_kl': {str(k): held_kl.get(k, 0) / max(1, held_kl.get(k, 0) + kl.get(k, 0)) for k in range(5)},
        'oarsi_counts': dict(sorted(ocounts.items())),
        'kl_source_sha256': source_sha,
        'oarsi_code_sha256': sha(OARSI_CODE / 'build_oarsi_labels_v2.py'),
        'manifest_sha256': {p.name: sha(p) for p in sorted(MANIFEST.glob('*.json*'))},
        'protocol_sha256': 'aa09e55834ba3cc043f93ddcfd243d49ffc1573400437c42ba8d6728a56070ea',
        'note': 'The non-inclusion fraction (automatic failures) must be reported by KL, which needs KL linked to the excluded films separately; see excluded_fate_kl.',
    }
    fate = json.load(open(OUT / 'film_fate_private.json'))
    not_in = Counter()
    for c, f in fate.items():
        if f == 'image_steps_completed':
            continue
        link = kl_by_relpath.get(films[c]['input_archive_relpath'])
        if link:
            not_in[(f, link[0])] += 1
            not_in[(f, link[1])] += 1
    summary['excluded_fate_kl'] = {f'{f}|kl{k}': n for (f, k), n in sorted(not_in.items())}
    write_json(MANIFEST / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('knees', 'patients_usable', 'usable_kl_counts', 'held_kl_counts', 'kl_link_exclusion_reasons_films')},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['lock', 'labels'])
    a = ap.parse_args()
    {'lock': lock, 'labels': labels}[a.command]()
