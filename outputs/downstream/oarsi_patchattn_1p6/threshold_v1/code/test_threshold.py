#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
B = HERE.parent.parent
ROOT = Path('/path/to/drai')
T = B / 'test_eval_v1'
MANIFEST = ROOT / 'data/manifests/oarsi_test_eval_v1_20261117'
sys.path.insert(0, str(B / 'analysis_v1/code'))
sys.path.insert(0, str(HERE))
from wp1_primary import MANIFEST_REL, RUNS, interval, load_manifest, qwk_from_confusion, sha256_file as sha  # noqa: E402
from wp2_secondary import read_jsonl, write_json  # noqa: E402
from threshold_analysis import fit_bias, fit_threshold, predict_bias, predict_threshold  # noqa: E402

SEEDS, GROUPS, C = (17, 23, 29), ('P0', 'P3'), 5
N_BOOT, BOOT_SEED = 10000, 20261115
ROLES = {'selected': 'primary_selected', 'epoch20': 'sensitivity_epoch20_end'}
VARIANTS = ('argmax', 'round', 'bias', 'threshold')
ROUND_T = np.array([0.5, 1.5, 2.5, 3.5])


def decide(variant, prob, params):
    if variant == 'argmax':
        return prob.argmax(1)
    if variant == 'round':
        return predict_threshold((prob * np.arange(C)).sum(1), ROUND_T)
    if variant == 'bias':
        return predict_bias(np.log(np.clip(prob, 1e-12, None)), params)
    return predict_threshold((prob * np.arange(C)).sum(1), params)


def qwk(y, p):
    o = np.bincount(y * C + p, minlength=C * C).reshape(C, C)
    return float(qwk_from_confusion(o))


def main():
    out = B / 'threshold_v1/test_outputs'
    out.mkdir(exist_ok=True)
    assert not any(out.iterdir()), 'output directory not empty'
    started = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    hashes = {}

    _, kl_val, _ = load_manifest(ROOT / MANIFEST_REL)
    stored = json.load(open(B / 'threshold_v1/outputs/threshold_results.json'))['per_run']
    params, check = {}, {}
    for s in SEEDS:
        for g in GROUPS:
            run = B / RUNS[(s, g)]
            sel = json.load(open(run / 'completed.json'))['best_epoch']
            for window, epoch in (('selected', sel), ('epoch20', 20)):
                f = run / f'val_epoch{epoch:02d}.npz'
                hashes[str(f.relative_to(ROOT))] = sha(f)
                z = np.load(f)
                assert np.array_equal(z['labels'].astype(int), kl_val)
                prob = z['probabilities']
                pb, qb = fit_bias(np.log(np.clip(prob, 1e-12, None)), kl_val)
                pt, qt = fit_threshold((prob * np.arange(C)).sum(1), kl_val)
                params[(window, s, g)] = {'bias': pb, 'threshold': pt, 'val_qwk': {'argmax': qwk(kl_val, prob.argmax(1)),
                                          'bias': qb, 'threshold': qt}}
                if window == 'selected':
                    st = stored[f'seed{s}_{g}']['selected']
                    check[f'seed{s}_{g}'] = bool(np.array_equal(pb, np.array(st['bias']['oracle_params']))
                                                 and np.array_equal(pt, np.array(st['threshold']['oracle_params'])))

    receipt = json.load(open(T / 'T4/test_inference_receipt.json'))
    rows = read_jsonl(MANIFEST / 'test_usable.jsonl', ('key', 'patient_id', 'kl'))
    y = np.array([r['kl'] for r in rows])
    names = sorted({r['patient_id'] for r in rows})
    lookup = {p: i for i, p in enumerate(names)}
    pidx = np.array([lookup[r['patient_id']] for r in rows])
    P, N = len(names), len(rows)
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, P, size=(N_BOOT, P))
    weights = np.zeros((N_BOOT, P)); np.add.at(weights, (np.arange(N_BOOT)[:, None], draws), 1)

    point, boot, recalls = {}, {}, {}
    for window, role in ROLES.items():
        for s in SEEDS:
            for g in GROUPS:
                f = T / 'T4/test' / f'{role}_seed{s}_{g}.npz'
                assert sha(f) == receipt['outputs'][f.name]
                hashes[str(f.relative_to(ROOT))] = sha(f)
                z = np.load(f)
                assert np.array_equal(z['ids'], np.arange(N)) and np.array_equal(z['labels'], y)
                prob = z['probabilities']
                for v in VARIANTS:
                    pred = decide(v, prob, params[(window, s, g)].get(v))
                    cbp = np.zeros((P, C * C)); np.add.at(cbp, (pidx, y * C + pred), 1)
                    point[(window, v, s, g)] = qwk(y, pred)
                    boot[(window, v, s, g)] = qwk_from_confusion((weights @ cbp).reshape(N_BOOT, C, C))
                    recalls[(window, v, s, g)] = [float((pred[y == k] == k).mean()) for k in range(C)]

    t5 = json.load(open(T / 'T5/t5_results.json'))
    raw = [point[('selected', 'argmax', s, 'P3')] - point[('selected', 'argmax', s, 'P0')] for s in SEEDS]
    dist = np.mean([boot[('selected', 'argmax', s, 'P3')] - boot[('selected', 'argmax', s, 'P0')] for s in SEEDS], axis=0)
    rep = max(abs(np.mean(raw) - t5['primary']['mean']), *[abs(a - b) for a, b in zip(interval(dist), t5['primary']['ci95'])])
    raw20 = np.mean([point[('epoch20', 'argmax', s, 'P3')] - point[('epoch20', 'argmax', s, 'P0')] for s in SEEDS])
    rep20 = abs(raw20 - t5['sensitivity_P3_minus_P0_epoch20_end']['qwk']['mean'])
    selfcheck = {'val_params_equal_stored_oracle': check, 'patients': P, 'knees': N,
                 't5_primary_reproduced_max_abs_diff': rep, 't5_epoch20_mean_reproduced_abs_diff': rep20}
    selfcheck['pass'] = bool(all(check.values()) and rep <= 1e-12 and rep20 <= 1e-12)
    write_json(out / 'selfcheck_test_threshold.json', selfcheck)
    if not selfcheck['pass']:
        print('self-check failed', file=sys.stderr); sys.exit(2)

    specs = {
        'raw_argmax_P3_minus_P0': (('argmax', 'P3'), ('argmax', 'P0')),
        'raw_round_P3_minus_P0': (('round', 'P3'), ('round', 'P0')),
        'bias_tuned_P3_minus_bias_tuned_P0': (('bias', 'P3'), ('bias', 'P0')),
        'threshold_tuned_P3_minus_threshold_tuned_P0': (('threshold', 'P3'), ('threshold', 'P0')),
        'raw_argmax_P3_minus_bias_tuned_P0': (('argmax', 'P3'), ('bias', 'P0')),
        'raw_argmax_P3_minus_threshold_tuned_P0': (('argmax', 'P3'), ('threshold', 'P0')),
        'P0_gain_bias_minus_argmax': (('bias', 'P0'), ('argmax', 'P0')),
        'P3_gain_bias_minus_argmax': (('bias', 'P3'), ('argmax', 'P3')),
        'P0_gain_threshold_minus_argmax': (('threshold', 'P0'), ('argmax', 'P0')),
        'P3_gain_threshold_minus_argmax': (('threshold', 'P3'), ('argmax', 'P3')),
        'P0_gain_round_minus_argmax': (('round', 'P0'), ('argmax', 'P0')),
        'P3_gain_round_minus_argmax': (('round', 'P3'), ('argmax', 'P3')),
    }
    comparisons = {}
    for window in ROLES:
        comparisons[window] = {}
        for nm, (a, b) in specs.items():
            per = {s: point[(window, a[0], s, a[1])] - point[(window, b[0], s, b[1])] for s in SEEDS}
            d = np.mean([boot[(window, a[0], s, a[1])] - boot[(window, b[0], s, b[1])] for s in SEEDS], axis=0)
            comparisons[window][nm] = {'per_seed': per, 'mean': float(np.mean(list(per.values()))), 'ci95': interval(d),
                                       'undefined_resamples': int(np.isnan(d).sum())}
    per_model = {f'{w}_seed{s}_{g}': {'test_qwk': {v: point[(w, v, s, g)] for v in VARIANTS},
                                      'val_qwk_in_sample': params[(w, s, g)]['val_qwk'],
                                      'params_fitted_on_validation': {v: params[(w, s, g)][v] for v in ('bias', 'threshold')},
                                      'test_recall': {v: recalls[(w, v, s, g)] for v in VARIANTS}}
                 for w in ROLES for s in SEEDS for g in GROUPS}
    recall_mean = {w: {g: {v: np.mean([recalls[(w, v, s, g)] for s in SEEDS], axis=0) for v in VARIANTS} for g in GROUPS} for w in ROLES}
    write_json(out / 'test_threshold_results.json', {'note': __doc__.strip(), 'comparisons': comparisons, 'per_model': per_model,
                                                    'test_recall_three_seed_mean': recall_mean})
    code = Path(__file__).resolve()
    lines = ['# Threshold test check output (post hoc, exploratory)', '', f'- Start: {started}', f'- End: {time.strftime("%Y-%m-%d %H:%M:%S %Z")}',
             f'- Python {platform.python_version()}, numpy {np.__version__}', f'- Code: `{code.name}`, SHA256 `{sha(code)}`; '
             f'depends on `threshold_analysis.py` SHA256 `{sha(HERE / "threshold_analysis.py")}`',
             '- Parameters are fitted on validation only; test uses only the stored WP6 T4 probabilities, no new inference; outputs hold aggregate statistics only.', '', '## Input file SHA256', '']
    lines += [f'- `{n}` `{d}`' for n, d in sorted(hashes.items())]
    (out / 'README_test_threshold.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({w: {k: round(v['mean'], 4) for k, v in comparisons[w].items()} for w in ROLES}, ensure_ascii=False))


if __name__ == '__main__':
    main()
