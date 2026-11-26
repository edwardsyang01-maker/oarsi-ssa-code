#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/path/to/drai')
B = ROOT / 'outputs/downstream/oarsi_patchattn_1p6'
sys.path.insert(0, str(B / 'analysis_v1/code'))
sys.path.insert(0, str(B / 'pvalues_v1/code'))
import pvalues_v1 as pv  # noqa: E402
from wp1_primary import MANIFEST_REL, interval  # noqa: E402
from wp2_secondary import OARSI_DIR, TARGETS, clean, conf_metrics, read_jsonl, targets_from  # noqa: E402

SEEDS = (17, 23, 29)
GROUPS = ('P0', 'P1', 'P2', 'P3', 'P4')
R, C, BINS, TOL = 10000, 5, 15, 1e-12
RUNS = dict(pv.RUNS)
RUNS.update({(17, 'P1'): 'attempt02_lambda_v2/train_P1', (23, 'P1'): 'attempt04_x1_seed23/train_P1',
             (29, 'P1'): 'attempt04_x1_seed29/train_P1'})
KL_METRICS = ('qwk', 'accuracy', 'mae', 'macro_f1', 'auc_kl_ge2', 'cross_two_error_rate', 'ece')
HASHES = {}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    try:
        HASHES[str(Path(path).relative_to(ROOT))] = h.hexdigest()
    except ValueError:
        HASHES[str(path)] = h.hexdigest()
    return h.hexdigest()


def load(path):
    sha(path)
    return np.load(path)


def wauc(scores, positive, w, chunk=1000):
    s = np.asarray(scores, float)
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    pos = np.asarray(positive, bool)[order]
    out = np.empty(len(w))
    for a in range(0, len(w), chunk):
        ww = w[a:a + chunk][:, order]
        wp, wn = ww * pos, ww * ~pos
        gp, gn = np.add.reduceat(wp, starts, axis=1), np.add.reduceat(wn, starts, axis=1)
        below = np.cumsum(gn, axis=1) - gn
        out[a:a + chunk] = (gp * (below + 0.5 * gn)).sum(1) / (wp.sum(1) * wn.sum(1))
    return out


def kl_metrics(prob, y, w):
    n = len(y)
    pred = prob.argmax(1)
    cell = np.zeros((n, C * C))
    cell[np.arange(n), y * C + pred] = 1
    pt_m = conf_metrics(cell.sum(0).reshape(C, C))
    bt_m = conf_metrics((w @ cell).reshape(len(w), C, C))
    pt = {k: float(pt_m[k]) for k in ('qwk', 'accuracy', 'mae', 'macro_f1', 'cross_two_error_rate')}
    bt = {k: np.asarray(bt_m[k], float) for k in pt}
    score, positive = prob[:, 2:].sum(1), y >= 2
    pt['auc_kl_ge2'] = float(wauc(score, positive, np.ones((1, n)))[0])
    bt['auc_kl_ge2'] = wauc(score, positive, w)
    conf = prob.max(1)
    bins = np.minimum((conf * BINS).astype(int), BINS - 1)
    d = np.zeros((n, BINS))
    d[np.arange(n), bins] = conf - (pred == y)
    pt['ece'] = float(np.abs(d.sum(0)).sum() / n)
    bt['ece'] = np.abs(w @ d).sum(1) / w.sum(1)
    return pt, bt


def oarsi_metrics(oprob, targets, w):
    pred = oprob.argmax(-1)
    pt, bt = {}, {}
    for j, t in enumerate(TARGETS):
        m = targets[:, j] >= 0
        idx = np.flatnonzero(m)
        cell = np.zeros((len(targets), 16))
        cell[idx, targets[idx, j] * 4 + pred[idx, j]] = 1
        o_pt, o_bt = cell.sum(0).reshape(4, 4), (w @ cell).reshape(len(w), 4, 4)
        pt[t] = {'n': int(m.sum()), 'qwk': float(pv.qwk_c(o_pt, 4)), 'accuracy': float(np.trace(o_pt) / o_pt.sum())}
        bt[t] = {'qwk': pv.qwk_c(o_bt, 4), 'accuracy': np.trace(o_bt, axis1=1, axis2=2) / o_bt.sum((1, 2))}
    return pt, bt


def summary(per_seed_pt, per_seed_bt):
    seeds = list(per_seed_pt)
    dist = np.mean([per_seed_bt[s] for s in seeds], axis=0)
    return {'estimate': float(np.mean([per_seed_pt[s] for s in seeds])),
            'per_seed': {str(s): float(per_seed_pt[s]) for s in seeds}, 'ci95': interval(dist), **pv.boot_p(dist)}


def absolute(per_seed_pt, per_seed_bt):
    seeds = list(per_seed_pt)
    dist = np.mean([per_seed_bt[s] for s in seeds], axis=0)
    return {'mean': float(np.mean([per_seed_pt[s] for s in seeds])), 'per_seed': {str(s): float(per_seed_pt[s]) for s in seeds},
            'sd': float(np.std([per_seed_pt[s] for s in seeds], ddof=1)) if len(seeds) > 1 else None, 'ci95': interval(dist)}


def factorial(v):
    return {'pooling_main': ((v['P1'] - v['P0']) + (v['P3'] - v['P2'])) / 2,
            'auxiliary_main': ((v['P2'] - v['P0']) + (v['P3'] - v['P1'])) / 2,
            'interaction': v['P3'] - v['P2'] - v['P1'] + v['P0']}


def close(a, b):
    return abs(a - b) <= TOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f'output directory is not empty; refusing to overwrite: {out}'
    started = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    checks = {}
    mainline = json.load(open(B / 'knee_bootstrap_v1/outputs_mainline/pvalues_v1.json'))
    sha(B / 'knee_bootstrap_v1/outputs_mainline/pvalues_v1.json')
    wp8 = json.load(open(B / 'wp8_analysis/outputs/oarsi_heads_and_laterality_three_seeds.json'))
    t5 = json.load(open(B / 'test_eval_v1/T5/t5_results.json'))
    sha(B / 'wp8_analysis/outputs/oarsi_heads_and_laterality_three_seeds.json')
    sha(B / 'test_eval_v1/T5/t5_results.json')

    sha(ROOT / MANIFEST_REL)
    sha(ROOT / f'{OARSI_DIR}/val_oarsi.jsonl')
    vrows = read_jsonl(ROOT / MANIFEST_REL, ('key', 'patient_id', 'split', 'visit', 'clinical_side', 'kl'))
    vlabels = read_jsonl(ROOT / f'{OARSI_DIR}/val_oarsi.jsonl', ('key', 'kl', 'clinical_side', 'visit', *TARGETS))
    vt = targets_from(vrows, vlabels)
    vy = np.array([int(r['kl']) for r in vrows])
    vw = pv.boot_weights(len(vy))
    vpt, vbt, vopt, vobt = {}, {}, {}, {}
    for s in SEEDS:
        for g in GROUPS:
            be = pv.best_epoch((s, g)) if (s, g) in pv.RUNS else int(json.load(open(B / RUNS[(s, g)] / 'completed.json'))['best_epoch'])
            z = load(B / RUNS[(s, g)] / f'val_epoch{be:02d}.npz')
            assert np.array_equal(z['labels'].astype(int), vy)
            vpt[(s, g)], vbt[(s, g)] = kl_metrics(z['probabilities'], vy, vw)
            if g in ('P2', 'P3', 'P4'):
                vopt[(s, g)], vobt[(s, g)] = oarsi_metrics(z['oarsi_probabilities'], vt, vw)
    val = {'n_knees': int(len(vy)),
           'per_model': {g: {m: absolute({s: vpt[(s, g)][m] for s in SEEDS}, {s: vbt[(s, g)][m] for s in SEEDS}) for m in KL_METRICS}
                         for g in GROUPS},
           'P3_minus_P0': {m: summary({s: vpt[(s, 'P3')][m] - vpt[(s, 'P0')][m] for s in SEEDS},
                                      {s: vbt[(s, 'P3')][m] - vbt[(s, 'P0')][m] for s in SEEDS}) for m in KL_METRICS},
           'qwk_simple': {f'{a}_minus_{b}': summary({s: vpt[(s, a)]['qwk'] - vpt[(s, b)]['qwk'] for s in SEEDS},
                                                     {s: vbt[(s, a)]['qwk'] - vbt[(s, b)]['qwk'] for s in SEEDS})
                          for a, b in (('P1', 'P0'), ('P2', 'P0'), ('P4', 'P3'), ('P4', 'P0'))},
           'factorial': {}}
    for m in ('qwk', 'auc_kl_ge2', 'cross_two_error_rate'):
        fpt = {s: factorial({g: vpt[(s, g)][m] for g in GROUPS[:4]}) for s in SEEDS}
        fbt = {s: factorial({g: vbt[(s, g)][m] for g in GROUPS[:4]}) for s in SEEDS}
        val['factorial'][m] = {e: summary({s: fpt[s][e] for s in SEEDS}, {s: fbt[s][e] for s in SEEDS}) for e in fpt[17]}
    val['oarsi'] = {}
    for t in TARGETS:
        entry = {'n': vopt[(17, 'P3')][t]['n']}
        for g in ('P2', 'P3', 'P4'):
            entry[g] = {k: absolute({s: vopt[(s, g)][t][k] for s in SEEDS}, {s: vobt[(s, g)][t][k] for s in SEEDS}) for k in ('qwk', 'accuracy')}
        for a, b in (('P3', 'P2'), ('P4', 'P3')):
            entry[f'{a}_minus_{b}'] = {k: summary({s: vopt[(s, a)][t][k] - vopt[(s, b)][t][k] for s in SEEDS},
                                                  {s: vobt[(s, a)][t][k] - vobt[(s, b)][t][k] for s in SEEDS}) for k in ('qwk', 'accuracy')}
        val['oarsi'][t] = entry
    adj = pv.holm({t: val['oarsi'][t]['P3_minus_P2']['qwk']['p_two_sided'] for t in TARGETS})
    for t in TARGETS:
        val['oarsi'][t]['P3_minus_P2']['qwk']['p_holm_6_heads'] = adj[t]

    ml = mainline['A_kl_validation_P3_minus_P0']['selected']
    a = val['P3_minus_P0']['qwk']
    checks['val_kl_vs_mainline'] = close(a['estimate'], ml['mean_diff']) and all(close(x, y) for x, y in zip(a['ci95'], ml['ci95'])) \
        and close(a['p_two_sided'], ml['p_two_sided'])
    ok = True
    for t in TARGETS:
        ml = mainline['B_oarsi_validation_P3_minus_P2_three_seeds'][t]['P3_minus_P2']
        a = val['oarsi'][t]['P3_minus_P2']['qwk']
        ok &= close(a['estimate'], ml['mean_diff']) and all(close(x, y) for x, y in zip(a['ci95'], ml['ci95'])) and close(a['p_two_sided'], ml['p_two_sided'])
        for g in ('P2', 'P3', 'P4'):
            for s in SEEDS:
                ok &= close(vopt[(s, g)][t]['qwk'], wp8['oarsi_heads'][g]['per_seed'][str(s)][t]['qwk'])
    checks['val_oarsi_vs_mainline_and_wp8'] = bool(ok)

    tm = ROOT / 'data/manifests/oarsi_test_eval_v1_20261117'
    sha(tm / 'test_usable.jsonl')
    sha(tm / 'test_oarsi.jsonl')
    trows = read_jsonl(tm / 'test_usable.jsonl', ('key', 'patient_id', 'split', 'visit', 'clinical_side', 'kl', 'cleanup_status'))
    tlabels = read_jsonl(tm / 'test_oarsi.jsonl', ('key', 'kl', 'clinical_side', 'visit', *TARGETS))
    tt = targets_from(trows, tlabels)
    ty = np.array([int(r['kl']) for r in trows])
    keys = [r['key'] for r in trows]
    assert len(set(keys)) == len(keys)
    rank = {k: i for i, k in enumerate(sorted(keys))}
    tidx = np.array([rank[k] for k in keys])
    tw = pv.boot_weights(len(ty))[:, tidx]
    names = {(s, g): f'primary_selected_seed{s}_{g}' for s in SEEDS for g in ('P0', 'P3')}
    names.update({(17, g): f'secondary_selected_seed17_{g}' for g in ('P1', 'P2', 'P4')})
    tpt, tbt, topt, tobt = {}, {}, {}, {}
    for key, name in names.items():
        z = load(B / f'test_eval_v1/T4/test/{name}.npz')
        assert np.array_equal(z['labels'], ty)
        tpt[key], tbt[key] = kl_metrics(z['probabilities'], ty, tw)
        if key[1] in ('P2', 'P3', 'P4'):
            topt[key], tobt[key] = oarsi_metrics(z['oarsi_probabilities'], tt, tw)
        for m in ('qwk', 'accuracy', 'mae', 'macro_f1', 'cross_two_error_rate', 'ece'):
            assert abs(tpt[key][m] - t5['per_model'][name][m]) < 1e-9, (name, m)
        assert abs(tpt[key]['auc_kl_ge2'] - t5['per_model'][name]['auc_kl_ge_2']) < 1e-9, name
    test = {'n_knees': int(len(ty)), 'n_participants': int(len({r['patient_id'] for r in trows})),
            'P3_three_seeds': {m: absolute({s: tpt[(s, 'P3')][m] for s in SEEDS}, {s: tbt[(s, 'P3')][m] for s in SEEDS}) for m in KL_METRICS},
            'P0_three_seeds': {m: absolute({s: tpt[(s, 'P0')][m] for s in SEEDS}, {s: tbt[(s, 'P0')][m] for s in SEEDS}) for m in KL_METRICS},
            'P3_minus_P0': {m: summary({s: tpt[(s, 'P3')][m] - tpt[(s, 'P0')][m] for s in SEEDS},
                                       {s: tbt[(s, 'P3')][m] - tbt[(s, 'P0')][m] for s in SEEDS}) for m in KL_METRICS},
            'seed17_points': {g: {m: tpt[(17, g)][m] for m in KL_METRICS} for g in GROUPS},
            'seed17_factorial_qwk': {}, 'seed17_P4_minus_P3_qwk': None, 'oarsi': {}}
    fpt = factorial({g: tpt[(17, g)]['qwk'] for g in GROUPS[:4]})
    fbt = factorial({g: tbt[(17, g)]['qwk'] for g in GROUPS[:4]})
    test['seed17_factorial_qwk'] = {e: summary({17: fpt[e]}, {17: fbt[e]}) for e in fpt}
    test['seed17_P4_minus_P3_qwk'] = summary({17: tpt[(17, 'P4')]['qwk'] - tpt[(17, 'P3')]['qwk']},
                                             {17: tbt[(17, 'P4')]['qwk'] - tbt[(17, 'P3')]['qwk']})
    for t in TARGETS:
        entry = {'n': topt[(17, 'P3')][t]['n'],
                 'P3_three_seeds': {k: absolute({s: topt[(s, 'P3')][t][k] for s in SEEDS}, {s: tobt[(s, 'P3')][t][k] for s in SEEDS})
                                    for k in ('qwk', 'accuracy')},
                 'seed17': {g: {k: topt[(17, g)][t][k] for k in ('qwk', 'accuracy')} for g in ('P2', 'P3', 'P4')}}
        for a, b in (('P3', 'P2'), ('P4', 'P3')):
            entry[f'seed17_{a}_minus_{b}'] = {k: summary({17: topt[(17, a)][t][k] - topt[(17, b)][t][k]},
                                                         {17: tobt[(17, a)][t][k] - tobt[(17, b)][t][k]}) for k in ('qwk', 'accuracy')}
        test['oarsi'][t] = entry
    ml = mainline['A_kl_test_P3_minus_P0']['primary_selected']
    a = test['P3_minus_P0']['qwk']
    checks['test_kl_vs_mainline'] = close(a['estimate'], ml['mean_diff']) and all(close(x, y) for x, y in zip(a['ci95'], ml['ci95'])) \
        and close(a['p_two_sided'], ml['p_two_sided'])
    ok = True
    for t in TARGETS:
        ml = mainline['B_oarsi_test_seed17_P3_minus_P2'][t]
        a = test['oarsi'][t]['seed17_P3_minus_P2']['qwk']
        ok &= close(a['estimate'], ml['diff']) and all(close(x, y) for x, y in zip(a['ci95'], ml['ci95'])) and close(a['p_two_sided'], ml['p_two_sided'])
        for s in SEEDS:
            ok &= close(topt[(s, 'P3')][t]['qwk'], t5['per_model'][f'primary_selected_seed{s}_P3']['oarsi'][t]['qwk'])
            ok &= close(topt[(s, 'P3')][t]['accuracy'], t5['per_model'][f'primary_selected_seed{s}_P3']['oarsi'][t]['accuracy'])
    checks['test_oarsi_vs_mainline_and_t5'] = bool(ok)

    rr_path = B / 'analysis_v1/outputs/reader_reliability.json'
    sha(rr_path)
    rr = json.load(open(rr_path))['results']['all_visits_pooled']
    retest = {}
    for name, entry in list(rr.items()) + [('KL_0_2_both', rr['KL_range_0_2']['both_readings_kl0_2_three_grades'])]:
        if 'crosstab_rows_first_cols_second' not in entry:
            continue
        o = np.asarray(entry['crosstab_rows_first_cols_second'], float)
        k = o.shape[0]
        first, second = np.nonzero(o)
        reps = o[first, second].astype(int)
        cell = np.repeat(first * k + second, reps)
        n = len(cell)
        onehot = np.zeros((n, k * k))
        onehot[np.arange(n), cell] = 1
        w = pv.boot_weights(n)
        boot = pv.qwk_c((w @ onehot).reshape(R, k, k), k)
        point = float(pv.qwk_c(o, k))
        retest[name] = {'pairs': n, 'qwk': point, 'ci95_knee_pair_bootstrap': interval(boot),
                        'ci95_participant_bootstrap_wp4': entry.get('ci95_kappa_quadratic'),
                        'point_matches_wp4': close(point, entry['kappa_quadratic'])}
    checks['retest_points_vs_wp4'] = all(v['point_matches_wp4'] for v in retest.values())

    passed = all(checks.values())
    result = {'schema': 'drai_results_v16_knee_v1',
              'note': 'Knee-level bootstrap statistics for the v1.6 results section; reads existing predictions only, CPU only, no new inference/training. Knee resampling treats the knee-visits of one participant as independent units, '
                      'so intervals are narrower than the prespecified participant-cluster bootstrap; training randomness and epoch selection are excluded. P values are uncorrected except where Holm is stated.',
              'bootstrap': 'unit is the knee; default_rng(20261115).integers(0, N, size=(10000, N)); validation knees numbered in list order, test knees by ascending key (as in mainline)',
              'selfcheck': {'pass': passed, **checks},
              'validation': val, 'test': test, 'reader_retest': retest,
              'started': started, 'finished': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'input_sha256': HASHES}
    (out / 'results_knee_v16.json').write_text(json.dumps(clean(result), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    code = Path(__file__).resolve()
    lines = ['# results_knee_v16', '', f'- Start: {started}', f'- End: {result["finished"]}',
             f'- Python {platform.python_version()}, numpy {np.__version__}', f'- Code SHA256 `{sha(code)}`',
             f'- Command: `{" ".join(sys.argv)}`', f'- Self-check: {"passed" if passed else "failed"} {json.dumps(checks)}', '', '## Input SHA256', '']
    lines += [f'- `{k}` `{v}`' for k, v in sorted(HASHES.items()) if not k.endswith(code.name)]
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'selfcheck': result['selfcheck']}, ensure_ascii=False))
    if not passed:
        sys.exit(2)


if __name__ == '__main__':
    main()
