#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'analysis_v1/code'))
from wp1_primary import (B_REL, BOOT_SEED, CLASSES, MANIFEST_REL, MANIFEST_SHA256, N_BOOT, N_KNEES,  # noqa: E402
                         N_PATIENTS, PLATEAU, RUNS, SEEDS, TOL, interval, load_manifest, qwk_from_confusion,
                         sha256_file)
from wp2_secondary import write_json  # noqa: E402

GROUPS = ('P0', 'P3')
FOLDS, REPEATS, FOLD_SEED = 5, 5, 20261120
BIAS_GRID = np.round(np.arange(-4, 4.0001, 0.05), 2)
TH_STEP = 0.01
MAX_ROUNDS = 20
METHODS = ('bias', 'threshold')


def qwk_batch(y, preds):
    g = preds.shape[0]
    idx = (np.arange(g)[:, None] * 25 + y[None, :] * CLASSES + preds).ravel()
    o = np.bincount(idx, minlength=g * 25).reshape(g, CLASSES, CLASSES)
    return qwk_from_confusion(o)


def pick(scores, grid, current):
    best = scores.max()
    cand = grid[scores == best]
    return float(cand[np.argmin(np.abs(cand - current))]), float(best)


def predict_bias(logp, b):
    return (logp + b).argmax(1)


def fit_bias(logp, y):
    b = np.zeros(CLASSES)
    best = float(qwk_batch(y, predict_bias(logp, b)[None])[0])
    for _ in range(MAX_ROUNDS):
        improved = False
        for j in range(1, CLASSES):
            base = logp + b
            others = base.copy(); others[:, j] = -np.inf
            ao, mo = others.argmax(1), others.max(1)
            sj = logp[:, j][None, :] + BIAS_GRID[:, None]
            pred = np.where(sj > mo, j, np.where(sj < mo, ao, np.minimum(j, ao)))
            v, q = pick(qwk_batch(y, pred), BIAS_GRID, b[j])
            if q > best + 1e-12:
                b[j], best, improved = v, q, True
        if not improved:
            break
    return b, best


def predict_threshold(s, t):
    return (s[:, None] >= t[None, :]).sum(1)


def fit_threshold(s, y):
    t = np.array([0.5, 1.5, 2.5, 3.5])
    best = float(qwk_batch(y, predict_threshold(s, t)[None])[0])
    for _ in range(MAX_ROUNDS):
        improved = False
        for j in range(4):
            lo = t[j - 1] if j > 0 else 0.0
            hi = t[j + 1] if j < 3 else 4.0
            grid = np.round(np.arange(lo + TH_STEP, hi - TH_STEP / 2, TH_STEP), 2)
            if grid.size == 0:
                continue
            rest = np.delete(t, j)
            base = (s[:, None] >= rest[None, :]).sum(1)
            pred = base[None, :] + (s[None, :] >= grid[:, None])
            v, q = pick(qwk_batch(y, pred), grid, t[j])
            if q > best + 1e-12:
                t[j], best, improved = v, q, True
        if not improved:
            break
    return t, best


def rule(method, prob):
    if method == 'bias':
        return np.log(np.clip(prob, 1e-12, None)), fit_bias, predict_bias, np.zeros(CLASSES)
    return (prob * np.arange(CLASSES)).sum(1), fit_threshold, predict_threshold, np.array([0.5, 1.5, 2.5, 3.5])


def crossfit(method, prob, y, knee_fold):
    x, fit, predict, _ = rule(method, prob)
    out = np.empty(len(y), dtype=int)
    for f in range(FOLDS):
        test = knee_fold == f
        params, _ = fit(x[~test], y[~test])
        out[test] = predict(x[test], params)
    return out


def recall(y, pred):
    return [float((pred[y == k] == k).mean()) for k in range(CLASSES)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--wp1-out', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root, out, wp1_out = Path(args.root), Path(args.out), Path(args.wp1_out)
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f'output directory not empty: {out}'
    started = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    base = root / B_REL
    hashes = {MANIFEST_REL: sha256_file(root / MANIFEST_REL)}
    patient_index, kl, n_patients = load_manifest(root / MANIFEST_REL)
    assert hashes[MANIFEST_REL] == MANIFEST_SHA256 and n_patients == N_PATIENTS and len(kl) == N_KNEES

    folds = []
    for r in range(REPEATS):
        perm = np.random.default_rng(FOLD_SEED + r).permutation(N_PATIENTS)
        fold_of_patient = np.empty(N_PATIENTS, dtype=int); fold_of_patient[perm] = np.arange(N_PATIENTS) % FOLDS
        folds.append(fold_of_patient[patient_index])

    keys = [(s, g) for s in SEEDS for g in GROUPS]
    plateau = list(PLATEAU)
    variants = ('argmax', 'round', 'bias', 'threshold')
    conf = {v: np.zeros((len(keys), 1 + len(plateau), N_PATIENTS, 25)) for v in variants}
    per_run, selfcheck = {}, {'runs': {}}
    for m, key in enumerate(keys):
        run = base / RUNS[key]
        completed = json.load(open(run / 'completed.json'))
        history = json.load(open(run / 'epoch_history.json'))
        sel = completed['best_epoch']
        epochs = [sel] + plateau
        res = {'selected_epoch': sel, 'selected': {}, 'plateau': {}}
        cache = {}
        max_diff = 0.0
        for slot, e in enumerate(epochs):
            name = f'val_epoch{e:02d}.npz'
            hashes[f'{B_REL}/{RUNS[key]}/{name}'] = sha256_file(run / name)
            z = np.load(run / name)
            assert np.array_equal(z['ids'], np.arange(N_KNEES)) and np.array_equal(z['labels'].astype(int), kl)
            prob = z['probabilities']
            preds = {'argmax': prob.argmax(1), 'round': predict_threshold((prob * np.arange(CLASSES)).sum(1), np.array([.5, 1.5, 2.5, 3.5]))}
            q_raw = float(qwk_batch(kl, preds['argmax'][None])[0])
            max_diff = max(max_diff, abs(q_raw - history[e - 1]['validation']['qwk']))
            entry = {'argmax_raw': q_raw, 'round_raw': float(qwk_batch(kl, preds['round'][None])[0])}
            for method in METHODS:
                x, fit, predict, _ = rule(method, prob)
                params, q_oracle = fit(x, kl)
                cf = [crossfit(method, prob, kl, folds[r]) for r in range(REPEATS)]
                qs = qwk_batch(kl, np.stack(cf))
                entry[method] = {'oracle': q_oracle, 'oracle_params': params, 'crossfit_mean': float(qs.mean()),
                                 'crossfit_min': float(qs.min()), 'crossfit_max': float(qs.max()), 'crossfit_r0': float(qs[0])}
                preds[method] = cf[0]
            for v in variants:
                np.add.at(conf[v][m, slot], (patient_index, kl * CLASSES + preds[v]), 1)
            cache[slot] = (entry, preds)
        res['selected'] = cache[0][0]
        res['recall_selected'] = {v: recall(kl, cache[0][1][v]) for v in variants}
        pl = [cache[i][0] for i in range(1, len(epochs))]
        res['plateau'] = {'argmax_raw': float(np.mean([p['argmax_raw'] for p in pl])),
                          'round_raw': float(np.mean([p['round_raw'] for p in pl])),
                          **{method: {k: float(np.mean([p[method][k] for p in pl])) for k in ('oracle', 'crossfit_mean', 'crossfit_r0')}
                             for method in METHODS}}
        per_run[f'seed{key[0]}_{key[1]}'] = res
        selfcheck['runs'][f'seed{key[0]}_{key[1]}'] = {'raw_qwk_max_abs_diff_vs_history': max_diff, 'pass': max_diff <= TOL}

    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, N_PATIENTS, size=(N_BOOT, N_PATIENTS))
    weights = np.zeros((N_BOOT, N_PATIENTS)); np.add.at(weights, (np.arange(N_BOOT)[:, None], draws), 1)
    boot = {}
    for v in variants:
        mat = conf[v].transpose(2, 0, 1, 3).reshape(N_PATIENTS, -1)
        qsel, qplat = np.zeros((N_BOOT, len(keys))), np.zeros((N_BOOT, len(keys)))
        for start in range(0, N_BOOT, 500):
            w = weights[start:start + 500]
            o = (w @ mat).reshape(len(w), len(keys), 1 + len(plateau), CLASSES, CLASSES)
            q = qwk_from_confusion(o)
            qsel[start:start + 500], qplat[start:start + 500] = q[:, :, 0], q[:, :, 1:].mean(2)
        boot[v] = {'sel': qsel, 'plat': qplat}
    ix = {key: i for i, key in enumerate(keys)}

    def dist(window, a, b):
        return np.mean([boot[a[0]][window][:, ix[(s, a[1])]] - boot[b[0]][window][:, ix[(s, b[1])]] for s in SEEDS], axis=0)

    wp1 = json.load(open(wp1_out / 'primary_p3_vs_p0.json'))
    rep = max(abs(x - y) for w, mname in (('sel', 'q_sel'), ('plat', 'q_plat'))
              for x, y in zip(interval(dist(w, ('argmax', 'P3'), ('argmax', 'P0'))), wp1['ci95_mean_diff'][mname]))
    selfcheck['wp1_interval_reproduced_max_abs_diff'] = rep
    selfcheck['pass'] = bool(rep <= 1e-12 and all(r['pass'] for r in selfcheck['runs'].values()))
    write_json(out / 'selfcheck_threshold.json', selfcheck)
    if not selfcheck['pass']:
        print('self-check failed', file=sys.stderr); sys.exit(2)

    def value(run, window, variant, field='crossfit_mean'):
        e = per_run[run]['selected' if window == 'sel' else 'plateau']
        return e[f'{variant}_raw'] if variant in ('argmax', 'round') else e[variant][field]

    comparisons = {}
    specs = {
        'raw_argmax_P3_minus_P0': (('argmax', 'P3'), ('argmax', 'P0')),
        'raw_round_P3_minus_P0': (('round', 'P3'), ('round', 'P0')),
        'bias_tuned_P3_minus_bias_tuned_P0': (('bias', 'P3'), ('bias', 'P0')),
        'raw_argmax_P3_minus_bias_tuned_P0': (('argmax', 'P3'), ('bias', 'P0')),
        'threshold_tuned_P3_minus_threshold_tuned_P0': (('threshold', 'P3'), ('threshold', 'P0')),
        'raw_argmax_P3_minus_threshold_tuned_P0': (('argmax', 'P3'), ('threshold', 'P0')),
        'P0_gain_bias_tuned_minus_raw': (('bias', 'P0'), ('argmax', 'P0')),
        'P3_gain_bias_tuned_minus_raw': (('bias', 'P3'), ('argmax', 'P3')),
        'P0_gain_threshold_tuned_minus_raw': (('threshold', 'P0'), ('argmax', 'P0')),
        'P3_gain_threshold_tuned_minus_raw': (('threshold', 'P3'), ('argmax', 'P3')),
    }
    for window in ('sel', 'plat'):
        comparisons[window] = {}
        for name, (a, b) in specs.items():
            per_seed = {s: value(f'seed{s}_{a[1]}', window, a[0]) - value(f'seed{s}_{b[1]}', window, b[0]) for s in SEEDS}
            per_seed_r0 = {s: value(f'seed{s}_{a[1]}', window, a[0], 'crossfit_r0') - value(f'seed{s}_{b[1]}', window, b[0], 'crossfit_r0') for s in SEEDS}
            comparisons[window][name] = {'per_seed_repeat_mean': per_seed, 'mean_repeat_mean': float(np.mean(list(per_seed.values()))),
                                         'mean_repeat0': float(np.mean(list(per_seed_r0.values()))),
                                         'ci95_repeat0': interval(dist(window, a, b))}
        for method in METHODS:
            per_seed = {s: per_run[f'seed{s}_P3']['selected' if window == 'sel' else 'plateau'][method]['oracle']
                        - per_run[f'seed{s}_P0']['selected' if window == 'sel' else 'plateau'][method]['oracle'] for s in SEEDS}
            comparisons[window][f'oracle_{method}_P3_minus_P0'] = {'per_seed': per_seed, 'mean': float(np.mean(list(per_seed.values())))}
    raw = {w: comparisons[w]['raw_argmax_P3_minus_P0']['mean_repeat_mean'] for w in ('sel', 'plat')}
    explained = {w: {m: comparisons[w][f'P0_gain_{m}_tuned_minus_raw']['mean_repeat_mean'] / raw[w] for m in METHODS} for w in ('sel', 'plat')}
    recall_mean = {g: {v: np.mean([per_run[f'seed{s}_{g}']['recall_selected'][v] for s in SEEDS], axis=0) for v in variants} for g in GROUPS}

    write_json(out / 'threshold_results.json', {
        'note': __doc__.strip(), 'per_run': per_run, 'comparisons': comparisons,
        'P0_tuning_gain_as_fraction_of_raw_gap': explained,
        'recall_selected_epoch_three_seed_mean': recall_mean,
        'windows': {'sel': 'selected epoch (each run own selected epoch; selection still by the original argmax QWK)', 'plat': 'mean over epochs 11-20'},
    })
    code = Path(__file__).resolve()
    lines = ['# Threshold analysis output (validation, post hoc, exploratory)', '', f'- Start: {started}', f'- End: {time.strftime("%Y-%m-%d %H:%M:%S %Z")}',
             f'- Python {platform.python_version()}, numpy {np.__version__}', f'- Code: `{code.name}`, SHA256 `{sha256_file(code)}`',
             f'- Command: `{" ".join(sys.argv)}`', '- Reads existing validation predictions only, CPU only, test not touched, outputs hold aggregate statistics only.', '', '## Input file SHA256', '']
    lines += [f'- `{n}` `{d}`' for n, d in sorted(hashes.items())]
    (out / 'README_threshold.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'selfcheck': True, 'raw': raw,
                      'tuned_gap': {w: {m: comparisons[w][f'{m}_tuned_P3_minus_{m}_tuned_P0']['mean_repeat_mean'] for m in METHODS} for w in ('sel', 'plat')},
                      'P3raw_minus_P0tuned': {w: {m: comparisons[w][f'raw_argmax_P3_minus_{m}_tuned_P0']['mean_repeat_mean'] for m in METHODS} for w in ('sel', 'plat')},
                      'explained': explained}, ensure_ascii=False))


if __name__ == '__main__':
    main()
