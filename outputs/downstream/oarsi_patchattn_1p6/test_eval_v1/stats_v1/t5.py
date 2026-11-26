#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path('/path/to/drai')
B = ROOT / 'outputs/downstream/oarsi_patchattn_1p6'
T = B / 'test_eval_v1'
MANIFEST = ROOT / 'data/manifests/oarsi_test_eval_v1_20261117'
sys.path.insert(0, str(B / 'analysis_v1/code'))
from wp1_primary import WeightedAuc, auc_rank, cross2_from_confusion, interval, qwk_from_confusion, sha256_file as sha  # noqa: E402
from wp2_secondary import TARGETS, clean, conf_metrics, oarsi_metrics, read_jsonl, subset_metrics, targets_from  # noqa: E402

SEEDS = (17, 23, 29)
N_BOOT, BOOT_SEED, C, BINS = 10000, 20261115, 5, 15
GATE_QWK, GATE_AUC_DROP, GATE_X2_RISE = 0.003, 0.005, 0.01


def name(role, seed, group):
    return f'{role}_seed{seed}_{group}'


def ece_value(prob, y):
    conf, correct = prob.max(1), prob.argmax(1) == y
    b = np.minimum((conf * BINS).astype(int), BINS - 1)
    return float(sum(abs(conf[b == k].mean() - correct[b == k].mean()) * (b == k).mean() for k in range(BINS) if np.any(b == k)))


def main():
    out = T / 'T5/t5_results.json'
    assert not out.exists(), 'refusing to overwrite'
    receipt = json.load(open(T / 'T4/test_inference_receipt.json'))
    summary = json.load(open(MANIFEST / 'summary.json'))
    rows = read_jsonl(MANIFEST / 'test_usable.jsonl', ('key', 'patient_id', 'split', 'visit', 'clinical_side', 'kl', 'cleanup_status'))
    labels = read_jsonl(MANIFEST / 'test_oarsi.jsonl', ('key', 'kl', 'clinical_side', 'visit', *TARGETS))
    targets = targets_from(rows, labels)
    y = np.array([r['kl'] for r in rows])
    names = sorted({r['patient_id'] for r in rows})
    lookup = {p: i for i, p in enumerate(names)}
    pidx = np.array([lookup[r['patient_id']] for r in rows])
    P, N = len(names), len(rows)

    pred = {}
    for f in sorted((T / 'T4/test').glob('*.npz')):
        assert sha(f) == receipt['outputs'][f.name]
        z = np.load(f)
        assert np.array_equal(z['ids'], np.arange(N)) and np.array_equal(z['labels'], y)
        pred[f.stem] = {k: z[k] for k in z.files}
    assert len(pred) == 15

    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, P, size=(N_BOOT, P))
    weights = np.zeros((N_BOOT, P))
    np.add.at(weights, (np.arange(N_BOOT)[:, None], draws), 1)
    knee_w = weights[:, pidx]

    def conf_by_patient(p):
        cbp = np.zeros((P, C * C))
        np.add.at(cbp, (pidx, y * C + p.argmax(1)), 1)
        return cbp

    cache = {}

    def boot(model):
        if model in cache:
            return cache[model]
        prob = pred[model]['probabilities']
        o = (weights @ conf_by_patient(prob)).reshape(N_BOOT, C, C)
        m = conf_metrics(o)
        r = {k: m[k] for k in ('qwk', 'accuracy', 'mae', 'macro_f1', 'kl1_precision', 'kl1_recall', 'kl1_f1',
                               'kl2_as_kl0_rate', 'cross_two_error_rate')}
        for k in range(1, C):
            r[f'auc_kl_ge_{k}'] = WeightedAuc(prob[:, k:].sum(1), y >= k)(knee_w)
        cache[model] = r
        return r

    def point(model):
        prob = pred[model]['probabilities']
        o = conf_by_patient(prob).sum(0).reshape(C, C)
        m = conf_metrics(o)
        r = {k: float(m[k]) for k in ('qwk', 'accuracy', 'mae', 'macro_f1', 'kl1_precision', 'kl1_recall', 'kl1_f1',
                                      'kl2_as_kl0_rate', 'cross_two_error_rate')}
        r.update({f'auc_kl_ge_{k}': auc_rank(prob[:, k:].sum(1), y >= k) for k in range(1, C)},
                 kl2_as_kl0_count=int(o[2, 0]), ece=ece_value(prob, y),
                 per_class={str(c): {'precision': float(m['precision'][c]), 'recall': float(m['recall'][c]),
                                     'f1': float(m['f1'][c]), 'support': int(o[c].sum())} for c in range(C)},
                 confusion_matrix_rows_true_cols_pred=o.astype(int).tolist())
        if 'oarsi_probabilities' in pred[model]:
            r['oarsi'] = oarsi_metrics(pred[model]['oarsi_probabilities'], targets)
        return r

    points = {m: point(m) for m in pred}
    scalars = ('qwk', 'accuracy', 'mae', 'macro_f1', 'kl1_precision', 'kl1_recall', 'kl1_f1', 'kl2_as_kl0_rate',
               'cross_two_error_rate', 'auc_kl_ge_1', 'auc_kl_ge_2', 'auc_kl_ge_3', 'auc_kl_ge_4')

    def p3_minus_p0(role):
        res = {}
        for k in scalars:
            per = {s: points[name(role, s, 'P3')][k] - points[name(role, s, 'P0')][k] for s in SEEDS}
            dist = np.mean([boot(name(role, s, 'P3'))[k] - boot(name(role, s, 'P0'))[k] for s in SEEDS], axis=0)
            res[k] = {'per_seed': per, 'mean': float(np.mean(list(per.values()))),
                      'sd_across_seeds_ddof1': float(np.std(list(per.values()), ddof=1)),
                      'ci95': interval(dist), 'undefined_resamples': int(np.isnan(dist).sum()),
                      'per_seed_ci95': {s: interval(boot(name(role, s, 'P3'))[k] - boot(name(role, s, 'P0'))[k]) for s in SEEDS}}
        ece = {s: points[name(role, s, 'P3')]['ece'] - points[name(role, s, 'P0')]['ece'] for s in SEEDS}
        res['ece'] = {'per_seed': ece, 'mean': float(np.mean(list(ece.values()))), 'ci95': None,
                      'note': 'ECE is reported as point values and per-seed differences only'}
        return res

    primary = p3_minus_p0('primary_selected')
    sensitivity = p3_minus_p0('sensitivity_epoch20_end')
    d = primary
    gates = {'gate1_qwk_selected_diff_ge_0.003': {'value': d['qwk']['mean'], 'pass': d['qwk']['mean'] >= GATE_QWK},
             'gate2_plateau_qwk_diff': {'value': None, 'pass': None, 'note': 'other checkpoints of epochs 11-20 were not kept, so this cannot be computed on test'},
             'gate3_auc_kl_ge2_drop_le_0.005': {'value': -d['auc_kl_ge_2']['mean'], 'pass': -d['auc_kl_ge_2']['mean'] <= GATE_AUC_DROP},
             'gate4_cross_two_rise_le_0.01': {'value': d['cross_two_error_rate']['mean'], 'pass': d['cross_two_error_rate']['mean'] <= GATE_X2_RISE}}

    def effects(v):
        return {'pooling_main': ((v['P1'] - v['P0']) + (v['P3'] - v['P2'])) / 2,
                'auxiliary_main': ((v['P2'] - v['P0']) + (v['P3'] - v['P1'])) / 2,
                'interaction': v['P3'] - v['P2'] - v['P1'] + v['P0']}
    fac_names = {'P0': name('primary_selected', 17, 'P0'), 'P3': name('primary_selected', 17, 'P3'),
                 'P1': name('secondary_selected', 17, 'P1'), 'P2': name('secondary_selected', 17, 'P2')}
    est = effects({g: points[m]['qwk'] for g, m in fac_names.items()})
    dist = effects({g: boot(m)['qwk'] for g, m in fac_names.items()})
    factorial = {'values': {g: points[m]['qwk'] for g, m in fac_names.items()},
                 'effects': {k: {'estimate': est[k], 'ci95': interval(dist[k])} for k in est}}
    p4, p3 = name('secondary_selected', 17, 'P4'), name('primary_selected', 17, 'P3')
    p4_vs_p3 = {'qwk': {'P4': points[p4]['qwk'], 'P3': points[p3]['qwk'], 'diff': points[p4]['qwk'] - points[p3]['qwk'],
                        'ci95': interval(boot(p4)['qwk'] - boot(p3)['qwk'])},
                'oarsi_qwk': {t: {'P4': points[p4]['oarsi'][t]['qwk'], 'P3': points[p3]['oarsi'][t]['qwk']} for t in TARGETS}}

    visit = np.array([r['visit'] for r in rows])
    side = np.array([r['clinical_side'] for r in rows])
    cleaned = np.array([r['cleanup_status'] == 'accepted_cleaned' for r in rows])
    complete = (targets[:, :4] >= 0).all(1)
    masks = {**{f'true_kl{k}': y == k for k in range(C)},
             'osteophytes_all_four_read': complete, 'osteophytes_incomplete': ~complete,
             'visit_V00': visit == 'V00', 'visit_followup': visit != 'V00', 'knee_right': side == 'R', 'knee_left': side == 'L',
             'ruler_accepted_cleaned': cleaned, 'ruler_accepted_unchanged': ~cleaned}
    subgroups = {}
    for label, mask in masks.items():
        per = {s: {g: subset_metrics(y, pred[name('primary_selected', s, g)]['probabilities'], mask) for g in ('P0', 'P3')} for s in SEEDS}
        diff = {}
        for metric in ('accuracy', 'mae', 'cross_two_error_rate', 'qwk', 'auc_kl_ge2'):
            if per[17]['P0'][metric] is None:
                continue
            vals = [per[s]['P3'][metric] - per[s]['P0'][metric] for s in SEEDS]
            diff[metric] = {'mean_diff': float(np.mean(vals)), 'per_seed': dict(zip(SEEDS, vals))}
        subgroups[label] = {'knees': int(mask.sum()), 'patients': int(len(np.unique(pidx[mask]))), 'P3_minus_P0': diff, 'per_seed': per}

    vrows = read_jsonl(ROOT / 'data/manifests/chen_yolo_usable448_1p6x_clean_v1_20261112/val_usable.jsonl',
                       ('key', 'patient_id', 'kl', 'candidate_family'))
    vbase = np.array([r['candidate_family'] == 'base' for r in vrows])
    vy = np.array([r['kl'] for r in vrows])
    vnames = sorted({r['patient_id'] for r in vrows})
    vlook = {p: i for i, p in enumerate(vnames)}
    vp = np.array([vlook[r['patient_id']] for r in vrows])
    vrng = np.random.default_rng(BOOT_SEED)
    vdraw = vrng.integers(0, len(vnames), size=(N_BOOT, len(vnames)))
    vw = np.zeros((N_BOOT, len(vnames)))
    np.add.at(vw, (np.arange(N_BOOT)[:, None], vdraw), 1)
    runs = {(17, 'P0'): 'attempt01/train_P0', (17, 'P3'): 'attempt02_lambda_v2/train_P3',
            (23, 'P0'): 'attempt03_seed23/train_P0', (23, 'P3'): 'attempt03_seed23/train_P3',
            (29, 'P0'): 'attempt03_seed29/train_P0', (29, 'P3'): 'attempt03_seed29/train_P3'}
    vq, vb = {}, {}
    for key, run in runs.items():
        epoch = json.load(open(B / run / 'completed.json'))['best_epoch']
        prob = np.load(B / run / f'val_epoch{epoch:02d}.npz')['probabilities']
        cbp = np.zeros((len(vnames), C * C))
        np.add.at(cbp, (vp[vbase], vy[vbase] * C + prob[vbase].argmax(1)), 1)
        vq[key] = float(qwk_from_confusion(cbp.sum(0).reshape(C, C)))
        vb[key] = qwk_from_confusion((vw @ cbp).reshape(N_BOOT, C, C))
    vdiff = {s: vq[(s, 'P3')] - vq[(s, 'P0')] for s in SEEDS}
    val_base = {'knees': int(vbase.sum()), 'patients': int(len(np.unique(vp[vbase]))), 'qwk': {f'seed{s}_{g}': vq[(s, g)] for s, g in vq},
                'P3_minus_P0_per_seed': vdiff, 'P3_minus_P0_mean': float(np.mean(list(vdiff.values()))),
                'ci95': interval(np.mean([vb[(s, 'P3')] - vb[(s, 'P0')] for s in SEEDS], axis=0))}

    result = {
        'schema': 'drai_wp6_T5_results_v1', 'created': time.strftime('%F %T %Z'),
        'protocol_sha256': 'aa09e55834ba3cc043f93ddcfd243d49ffc1573400437c42ba8d6728a56070ea',
        'test_manifest_summary_sha256': sha(MANIFEST / 'summary.json'), 'test_inference_receipt_sha256': sha(T / 'T4/test_inference_receipt.json'),
        'cohort': {'knees': N, 'patients': P, 'kl_counts': {str(k): int((y == k).sum()) for k in range(C)},
                   'name': 'internal held-out set with prior test use'},
        'bootstrap': f'patient-clustered, {P} test patients resampled with replacement, {P} patients, {N_BOOT} times, default_rng({BOOT_SEED}), patients ordered by ascending ID string; per-seed differences first, then the mean; percentile interval',
        'primary': {'estimand': 'mean P3 - P0 test QWK over three seeds (selected epoch)', **primary['qwk'],
                    'direction_consistent_on_test': primary['qwk']['mean'] > 0},
        'gates_on_test': gates,
        'secondary_P3_minus_P0_selected': primary,
        'sensitivity_P3_minus_P0_epoch20_end': sensitivity,
        'per_model': points,
        'factorial_seed17_qwk': factorial,
        'P4_vs_P3_seed17': p4_vs_p3,
        'subgroups_selected': subgroups,
        'val_same_subset_base_family_P3_minus_P0': val_base,
        'not_computed_on_test': ['plateau mean over epochs 11-20', 'flip consistency and N4 subsets', 'attention maps'],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clean(result), ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(clean({'primary': result['primary'], 'gates': gates, 'val_base': {k: val_base[k] for k in ('P3_minus_P0_mean', 'ci95')}}),
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
