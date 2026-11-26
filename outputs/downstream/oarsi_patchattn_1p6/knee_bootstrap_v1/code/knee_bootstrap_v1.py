#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/path/to/drai/outputs/downstream/oarsi_patchattn_1p6/pvalues_v1/code')
import pvalues_v1 as pv  # noqa: E402
from wp1_primary import MANIFEST_REL, interval, load_manifest, qwk_from_confusion  # noqa: E402
from wp2_secondary import read_jsonl  # noqa: E402

C, R = pv.C, pv.N_BOOT


def weights(n):
    return pv.boot_weights(n)


def conf(idx, n, y, pred):
    return pv.patient_conf(idx, n, y, pred, C)


def diff_summary(pt, bt):
    dist = np.mean([bt[s] for s in pv.SEEDS], axis=0)
    return {'mean_diff': float(np.mean([pt[s] for s in pv.SEEDS])), 'ci95': interval(dist), **pv.boot_p(dist)}


def run(units, y, preds_by_key, keys_pairs):
    idx, n = units
    w = weights(n)
    out = {}
    for name, preds in preds_by_key.items():
        pt, bt = {}, {}
        for s in pv.SEEDS:
            q = {}
            for g in ('P0', 'P3'):
                pts, bts = [], []
                for p in preds[(s, g)]:
                    c = conf(idx, n, y, p)
                    pts.append(float(qwk_from_confusion(c.sum(0).reshape(C, C))))
                    bts.append(qwk_from_confusion((w @ c).reshape(R, C, C)))
                q[g] = (float(np.mean(pts)), np.mean(bts, axis=0))
            pt[s], bt[s] = q['P3'][0] - q['P0'][0], q['P3'][1] - q['P0'][1]
        out[name] = diff_summary(pt, bt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f'output directory not empty: {out}'

    vpidx, vy, vP = load_manifest(pv.ROOT / MANIFEST_REL)
    vN = len(vy)
    vpred = {'val_selected': {}, 'val_plateau_11_20': {}}
    for s in pv.SEEDS:
        for g in ('P0', 'P3'):
            key = (s, g)
            be = pv.best_epoch(key)
            z = np.load(pv.B / pv.RUNS[key] / f'val_epoch{be:02d}.npz')
            assert np.array_equal(z['labels'].astype(int), vy)
            vpred['val_selected'][key] = [z['probabilities'].argmax(1)]
            vpred['val_plateau_11_20'][key] = [np.load(pv.B / pv.RUNS[key] / f'val_epoch{e:02d}.npz')['probabilities'].argmax(1)
                                               for e in pv.PLATEAU]

    trows = read_jsonl(pv.TEST_MANIFEST / 'test_usable.jsonl', ('key', 'patient_id', 'kl'))
    ty = np.array([r['kl'] for r in trows])
    tnames = sorted({r['patient_id'] for r in trows})
    look = {p: i for i, p in enumerate(tnames)}
    tpidx = np.array([look[r['patient_id']] for r in trows])
    tP, tN = len(tnames), len(ty)
    tpred = {'test_selected': {}, 'test_epoch20_end': {}}
    for role, name in (('primary_selected', 'test_selected'), ('sensitivity_epoch20_end', 'test_epoch20_end')):
        for s in pv.SEEDS:
            for g in ('P0', 'P3'):
                z = np.load(pv.T4 / f'{role}_seed{s}_{g}.npz')
                assert np.array_equal(z['labels'], ty)
                tpred[name][(s, g)] = [z['probabilities'].argmax(1)]

    res = {
        'patient': {**run((vpidx, vP), vy, vpred, None), **run((tpidx, tP), ty, tpred, None)},
        'knee': {**run((np.arange(vN), vN), vy, vpred, None), **run((np.arange(tN), tN), ty, tpred, None)},
    }
    ref = json.load(open(pv.B / 'pvalues_v1/outputs/pvalues_v1.json'))
    refmap = {'val_selected': ref['A_kl_validation_P3_minus_P0']['selected'],
              'val_plateau_11_20': ref['A_kl_validation_P3_minus_P0']['plateau_epochs_11_20'],
              'test_selected': ref['A_kl_test_P3_minus_P0']['primary_selected'],
              'test_epoch20_end': ref['A_kl_test_P3_minus_P0']['sensitivity_epoch20_end']}
    d = max(max(abs(res['patient'][k]['mean_diff'] - v['mean_diff']), *[abs(a - b) for a, b in zip(res['patient'][k]['ci95'], v['ci95'])],
                abs(res['patient'][k]['p_two_sided'] - v['p_two_sided'])) for k, v in refmap.items())
    result = {'schema': 'drai_knee_vs_patient_bootstrap_v1',
              'note': 'The knee-level bootstrap treats the knee-visits of one participant as independent units; for comparison only, not for formal inference.',
              'n': {'val_knees': int(vN), 'val_patients': int(vP), 'test_knees': int(tN), 'test_patients': int(tP)},
              'selfcheck_patient_vs_pvalues_v1': {'max_abs_diff': d, 'pass': d <= 1e-12},
              **res}
    (out / 'knee_bootstrap_v1.json').write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if d > 1e-12:
        sys.exit(2)


if __name__ == '__main__':
    main()
