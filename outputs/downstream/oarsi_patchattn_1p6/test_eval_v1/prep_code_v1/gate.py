#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from collections import Counter
from pathlib import Path

PROJECT = Path('/path/to/drai')
B = PROJECT / 'outputs/downstream/oarsi_patchattn_1p6'
OUT = B / 'test_eval_v1/prep/val_replay'
FROZEN = PROJECT / 'data/manifests/chen_yolo_usable448_1p6x_clean_v1_20261112'
FROZEN_SHA = {'val_usable.jsonl': '4327b06965bb4041de161f19aa8974b452061549364009431b1dd3c123c53258',
              'val_held.jsonl': 'c1a812dde407783890abefe5bd15b7198f36aeedb0be2f2d5a8ca490ef5d49fc'}
THRESHOLD = 0.99


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def receipts(stage):
    return {p.stem: json.loads(p.read_text()) for p in sorted((OUT / stage).glob('*.json'))}


def main():
    assert not (OUT / 'gate.json').exists(), 'gate.json exists; refusing to overwrite'
    for name, digest in FROZEN_SHA.items():
        assert sha(FROZEN / name) == digest
    frozen = {}
    for name in FROZEN_SHA:
        for line in open(FROZEN / name):
            r = json.loads(line)
            frozen[r['key']] = r
    base = {k: r for k, r in frozen.items() if r['candidate_family'] == 'base'}
    base_cases = {r['case_id'] for r in base.values()}
    films = {r['case_id']: r for r in json.load(open(OUT / 'films_private.json'))}

    admit = receipts('admit')
    a = Counter()
    for c, r in admit.items():
        ref = films[c]['frozen_reference']
        a[r['status']] += 1
        if r['status'] == 'passed':
            a['archive_sha_equal'] += r['archive_sha256'] == ref['archive_sha256']
            a['native_sha_equal'] += r['native_pixel_sha256'] == ref['native_pixel_sha256']
            a['modality_equal'] += r['dicom_modality'] == ref['modality']

    c13 = receipts('crop13')
    mine = {c for c, r in c13.items() if r.get('automatic_pass')}
    trig = Counter(t for r in c13.values() for t in r.get('triggers', []))
    parent = Counter()
    for k, r in base.items():
        rec = c13.get(r['case_id'])
        if rec and rec.get('automatic_pass'):
            item = next(x for x in rec['items'] if x['key'] == k)
            parent['compared'] += 1
            parent['equal'] += item['gray01_sha256'] == r['parent_224_gray01_sha256']

    c16 = receipts('crop16')
    pre, fin, mism = Counter(), Counter(), {'pre': [], 'final': []}
    status_pairs = Counter()
    for k, r in base.items():
        rec = c16.get(r['case_id'])
        if not rec or rec['status'] != 'completed':
            pre['missing'] += 1
            fin['missing'] += 1
            mism['pre'].append(k)
            mism['final'].append(k)
            continue
        raw = {e['key']: e for e in json.load(open(rec['raw_path']))['entries']}
        final = {e['key']: e for e in json.load(open(rec['final_path']))['entries']}
        pre['compared'] += 1
        if raw[k]['gray01_sha256'] == r['raw_gray01_sha256']:
            pre['equal'] += 1
        else:
            mism['pre'].append(k)
        f = final[k]
        fin['compared'] += 1
        same = f['engineering_input_ready'] == r['engineering_input_ready'] and f['gray01_sha256'] == r['gray01_sha256']
        fin['equal'] += same
        status_pairs[(r['cleanup_status'], f['cleanup_status'])] += 1
        if not same:
            mism['final'].append(k)
    total = len(base)
    pre_rate = pre['equal'] / total
    fin_rate = fin['equal'] / total
    result = {
        'frozen_base_knees': total, 'frozen_base_films': len(base_cases),
        'admit': {'films': len(films), **dict(a)},
        'automatic_pass': {'mine_films': len(mine), 'frozen_base_films': len(base_cases),
                           'both': len(mine & base_cases), 'only_mine': len(mine - base_cases),
                           'only_frozen': len(base_cases - mine), 'trigger_counts': dict(trig)},
        'parent_224_equal': dict(parent),
        'gate1_preclean_1p6_identical': {**dict(pre), 'rate': pre_rate, 'required': 1.0, 'pass': pre['equal'] == total},
        'gate2_final_disposition_and_array': {**dict(fin), 'rate': fin_rate, 'required': THRESHOLD, 'pass': fin_rate >= THRESHOLD},
        'final_status_pairs_frozen_to_replay': {f'{a} -> {b}': n for (a, b), n in sorted(status_pairs.items()) if a != b},
    }
    result['passed'] = bool(result['gate1_preclean_1p6_identical']['pass'] and result['gate2_final_disposition_and_array']['pass'])
    with open(OUT / 'gate_mismatches_private.json', 'w') as f:
        json.dump({'pre': mism['pre'], 'final': mism['final'], 'only_mine': sorted(mine - base_cases),
                   'only_frozen': sorted(base_cases - mine)}, f, indent=1)
    with open(OUT / 'gate.json', 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
