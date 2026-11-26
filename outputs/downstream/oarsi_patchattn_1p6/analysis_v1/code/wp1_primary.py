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

SEEDS = (17, 23, 29)
GROUPS = ('P0', 'P3')
RUNS = {
    (17, 'P0'): 'attempt01/train_P0',
    (17, 'P3'): 'attempt02_lambda_v2/train_P3',
    (23, 'P0'): 'attempt03_seed23/train_P0',
    (23, 'P3'): 'attempt03_seed23/train_P3',
    (29, 'P0'): 'attempt03_seed29/train_P0',
    (29, 'P3'): 'attempt03_seed29/train_P3',
}
B_REL = 'outputs/downstream/oarsi_patchattn_1p6'
MANIFEST_REL = 'data/manifests/chen_yolo_usable448_1p6x_clean_v1_20261112/val_usable.jsonl'
MANIFEST_SHA256 = '4327b06965bb4041de161f19aa8974b452061549364009431b1dd3c123c53258'
N_KNEES, N_PATIENTS, EPOCHS, CLASSES = 3962, 435, 20, 5
PLATEAU = tuple(range(11, 21))
N_BOOT, BOOT_SEED = 10000, 20261115
TOL = 1e-9
# The four gates of the frozen plan (section 6.1); not to be changed.
GATE_QWK_SEL, GATE_QWK_PLAT, GATE_AUC_DROP, GATE_X2_RISE = 0.003, 0.003, 0.005, 0.01
HAND_SEED17 = {
    'P0': {'best_epoch': 8, 'q_sel': 0.8548, 'q_plat': 0.8442, 'auc_sel': 0.9445, 'x2_sel': 0.0386},
    'P3': {'best_epoch': 13, 'q_sel': 0.8628, 'q_plat': 0.8542, 'auc_sel': 0.9461, 'x2_sel': 0.0422},
    'diff': {'q_sel': 0.0080, 'q_plat': 0.0100, 'auc_sel': 0.0016, 'x2_sel': 0.0036},
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_manifest(path):
    patients, kl = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            assert row['split'] == 'val', 'the list contains non-val rows'
            patients.append(row['patient_id'])
            kl.append(int(row['kl']))
    names = sorted(set(patients))
    lookup = {name: i for i, name in enumerate(names)}
    return np.array([lookup[p] for p in patients]), np.array(kl), len(names)


def qwk_from_confusion(o):
    o = np.asarray(o, float)
    w = np.subtract.outer(np.arange(CLASSES), np.arange(CLASSES)) ** 2 / (CLASSES - 1) ** 2
    total = o.sum((-1, -2))
    e = o.sum(-1)[..., :, None] * o.sum(-2)[..., None, :] / total[..., None, None]
    denom = (w * e).sum((-1, -2))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom == 0, np.nan, 1 - (w * o).sum((-1, -2)) / denom)


def cross2_from_confusion(o):
    o = np.asarray(o, float)
    far = np.abs(np.subtract.outer(np.arange(CLASSES), np.arange(CLASSES))) >= 2
    return (o * far).sum((-1, -2)) / o.sum((-1, -2))


def auc_rank(scores, positive):
    s, p = np.asarray(scores, float), np.asarray(positive, bool)
    n1, n0 = int(p.sum()), int((~p).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    average = ends - (counts - 1) / 2
    ranks = average[inverse]
    return float((ranks[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class WeightedAuc:
    def __init__(self, scores, positive):
        s = np.asarray(scores, float)
        self.order = np.argsort(s, kind='mergesort')
        sorted_s = s[self.order]
        self.starts = np.flatnonzero(np.r_[True, sorted_s[1:] != sorted_s[:-1]])
        self.pos = np.asarray(positive, bool)[self.order]

    def __call__(self, knee_weights):
        w = knee_weights[:, self.order]
        wp, wn = w * self.pos, w * ~self.pos
        gpos = np.add.reduceat(wp, self.starts, axis=1)
        gneg = np.add.reduceat(wn, self.starts, axis=1)
        below = np.cumsum(gneg, axis=1) - gneg
        num = (gpos * (below + 0.5 * gneg)).sum(1)
        den = wp.sum(1) * wn.sum(1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(den == 0, np.nan, num / den)


def interval(x):
    low, high = np.nanpercentile(x, [2.5, 97.5])
    return [float(low), float(high)]


def write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='drai project root')
    parser.add_argument('--out', required=True, help='output directory (must be empty or absent)')
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f'output directory is not empty; refusing to overwrite: {out}'
    started = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    base = root / B_REL
    input_hashes = {}

    manifest_path = root / MANIFEST_REL
    manifest_sha = sha256_file(manifest_path)
    input_hashes[MANIFEST_REL] = manifest_sha
    patient_index, kl, n_patients = load_manifest(manifest_path)
    selfcheck = {
        'manifest_sha256_ok': manifest_sha == MANIFEST_SHA256,
        'knees_ok': len(kl) == N_KNEES,
        'patients_ok': n_patients == N_PATIENTS,
        'runs': {},
    }

    keys = [(s, g) for s in SEEDS for g in GROUPS]
    qwk_all = np.zeros((len(keys), EPOCHS))
    patient_conf = np.zeros((len(keys), EPOCHS, N_PATIENTS, CLASSES * CLASSES))
    best_epoch, sel_scores = {}, {}
    for m, key in enumerate(keys):
        run = base / RUNS[key]
        for name in ('epoch_history.json', 'completed.json'):
            input_hashes[f'{B_REL}/{RUNS[key]}/{name}'] = sha256_file(run / name)
        history = json.load(open(run / 'epoch_history.json'))
        completed = json.load(open(run / 'completed.json'))
        check = {'history_rows_ok': len(history) == EPOCHS and [r['epoch'] for r in history] == list(range(1, EPOCHS + 1)),
                 'completed_epochs_ok': completed['epochs'] == EPOCHS,
                 'test_accessed_false': completed['test_accessed'] is False,
                 'ids_ok': True, 'labels_ok': True,
                 'max_abs_diff': {'qwk': 0.0, 'auc_kl_ge2': 0.0, 'cross_two_error_rate': 0.0}}
        auc_by_epoch, x2_by_epoch, scores_by_epoch = [], [], []
        for epoch in range(1, EPOCHS + 1):
            name = f'val_epoch{epoch:02d}.npz'
            input_hashes[f'{B_REL}/{RUNS[key]}/{name}'] = sha256_file(run / name)
            z = np.load(run / name)
            ids, labels, prob = z['ids'], z['labels'].astype(int), z['probabilities']
            check['ids_ok'] &= bool(ids.shape == (N_KNEES,) and np.array_equal(ids, np.arange(N_KNEES)))
            check['labels_ok'] &= bool(np.array_equal(labels, kl))
            pred = prob.argmax(1)
            np.add.at(patient_conf[m, epoch - 1], (patient_index, labels * CLASSES + pred), 1)
            o = patient_conf[m, epoch - 1].sum(0).reshape(CLASSES, CLASSES)
            q, x2 = float(qwk_from_confusion(o)), float(cross2_from_confusion(o))
            score = prob[:, 2:].sum(1)
            a = auc_rank(score, labels >= 2)
            qwk_all[m, epoch - 1] = q
            auc_by_epoch.append(a); x2_by_epoch.append(x2); scores_by_epoch.append(score)
            v = history[epoch - 1]['validation']
            for metric, value in (('qwk', q), ('auc_kl_ge2', a), ('cross_two_error_rate', x2)):
                check['max_abs_diff'][metric] = max(check['max_abs_diff'][metric], abs(value - v[metric]))
        # Epoch selection: highest unrounded QWK, earliest epoch on ties (runner.py updates only on a strict increase).
        recomputed_best = int(np.argmax(qwk_all[m])) + 1
        check.update(recomputed_best_epoch=recomputed_best, completed_best_epoch=completed['best_epoch'],
                     best_epoch_ok=recomputed_best == completed['best_epoch'],
                     metrics_ok=all(d <= TOL for d in check['max_abs_diff'].values()))
        check['pass'] = all(check[k] for k in ('history_rows_ok', 'completed_epochs_ok', 'test_accessed_false',
                                               'ids_ok', 'labels_ok', 'best_epoch_ok', 'metrics_ok'))
        selfcheck['runs'][f'seed{key[0]}_{key[1]}'] = check
        best_epoch[key] = recomputed_best
        sel_scores[key] = {'score': scores_by_epoch[recomputed_best - 1],
                           'auc': auc_by_epoch[recomputed_best - 1], 'x2': x2_by_epoch[recomputed_best - 1]}
    selfcheck['pass'] = bool(selfcheck['manifest_sha256_ok'] and selfcheck['knees_ok'] and selfcheck['patients_ok']
                             and all(c['pass'] for c in selfcheck['runs'].values()))
    write_json(out / 'selfcheck.json', selfcheck)
    if not selfcheck['pass']:
        print('self-check failed; stopping. See selfcheck.json', file=sys.stderr)
        sys.exit(2)

    plateau_idx = [e - 1 for e in PLATEAU]
    point = {}
    for m, key in enumerate(keys):
        point[key] = {'best_epoch': best_epoch[key], 'q_sel': float(qwk_all[m, best_epoch[key] - 1]),
                      'q_plat': float(qwk_all[m, plateau_idx].mean()),
                      'auc_sel': sel_scores[key]['auc'], 'x2_sel': sel_scores[key]['x2']}
    metrics = ('q_sel', 'q_plat', 'auc_sel', 'x2_sel')
    diff_seed = {s: {k: point[(s, 'P3')][k] - point[(s, 'P0')][k] for k in metrics} for s in SEEDS}
    diff_mean = {k: float(np.mean([diff_seed[s][k] for s in SEEDS])) for k in metrics}
    diff_sd = {k: float(np.std([diff_seed[s][k] for s in SEEDS], ddof=1)) for k in metrics}

    # ---------- patient-clustered bootstrap ----------
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, N_PATIENTS, size=(N_BOOT, N_PATIENTS))
    weights = np.zeros((N_BOOT, N_PATIENTS))
    np.add.at(weights, (np.arange(N_BOOT)[:, None], draws), 1)
    conf_matrix = patient_conf.transpose(2, 0, 1, 3).reshape(N_PATIENTS, -1)
    weighted_auc = {key: WeightedAuc(sel_scores[key]['score'], kl >= 2) for key in keys}
    sel_idx = np.array([best_epoch[key] - 1 for key in keys])
    boot = {k: np.zeros((N_BOOT, len(keys))) for k in metrics}
    for start in range(0, N_BOOT, 500):
        w = weights[start:start + 500]
        o = (w @ conf_matrix).reshape(len(w), len(keys), EPOCHS, CLASSES, CLASSES)
        q = qwk_from_confusion(o)
        boot['q_sel'][start:start + 500] = q[:, np.arange(len(keys)), sel_idx]
        boot['q_plat'][start:start + 500] = q[:, :, plateau_idx].mean(2)
        o_sel = o[:, np.arange(len(keys)), sel_idx]
        boot['x2_sel'][start:start + 500] = cross2_from_confusion(o_sel)
        knee_w = w[:, patient_index]
        for m, key in enumerate(keys):
            boot['auc_sel'][start:start + 500, m] = weighted_auc[key](knee_w)
    undefined = {k: int(np.isnan(boot[k]).any(1).sum()) for k in metrics}

    repeat = np.repeat(np.arange(N_KNEES), weights[0, patient_index].astype(int))
    verify = {'qwk_sel': 0.0, 'auc_sel': 0.0}
    for m, key in enumerate(keys):
        z = np.load(base / RUNS[key] / f'val_epoch{best_epoch[key]:02d}.npz')
        y, prob = z['labels'].astype(int)[repeat], z['probabilities'][repeat]
        o = np.zeros((CLASSES, CLASSES)); np.add.at(o, (y, prob.argmax(1)), 1)
        verify['qwk_sel'] = max(verify['qwk_sel'], abs(float(qwk_from_confusion(o)) - boot['q_sel'][0, m]))
        verify['auc_sel'] = max(verify['auc_sel'], abs(auc_rank(prob[:, 2:].sum(1), y >= 2) - boot['auc_sel'][0, m]))
    verify['pass'] = all(v <= TOL for v in (verify['qwk_sel'], verify['auc_sel']))
    assert verify['pass'], f'weighted algorithm differs from direct computation on duplicated data: {verify}'

    # Take the difference within each seed first, then average over the three seeds.
    boot_diff_seed = {s: {k: boot[k][:, keys.index((s, 'P3'))] - boot[k][:, keys.index((s, 'P0'))] for k in metrics}
                      for s in SEEDS}
    boot_diff_mean = {k: np.mean([boot_diff_seed[s][k] for s in SEEDS], axis=0) for k in metrics}
    ci_mean = {k: interval(boot_diff_mean[k]) for k in metrics}
    ci_seed = {s: {k: interval(boot_diff_seed[s][k]) for k in metrics} for s in SEEDS}

    def gates(d):
        return {
            'qwk_selected_diff_ge_0.003': {'value': d['q_sel'], 'pass': d['q_sel'] >= GATE_QWK_SEL},
            'qwk_plateau_diff_ge_0.003': {'value': d['q_plat'], 'pass': d['q_plat'] >= GATE_QWK_PLAT},
            'auc_kl_ge2_drop_le_0.005': {'value': -d['auc_sel'], 'pass': -d['auc_sel'] <= GATE_AUC_DROP},
            'cross_two_error_rise_le_0.01': {'value': d['x2_sel'], 'pass': d['x2_sel'] <= GATE_X2_RISE},
        }
    gates_mean = gates(diff_mean)
    all_gates = all(g['pass'] for g in gates_mean.values())
    plat_all_positive = all(diff_seed[s]['q_plat'] > 0 for s in SEEDS)
    sel_all_positive = all(diff_seed[s]['q_sel'] > 0 for s in SEEDS)
    if not (diff_mean['q_sel'] > 0 and diff_mean['q_plat'] > 0):
        grade = 'D'
    elif not all_gates or not (plat_all_positive and sel_all_positive):
        grade = 'C'
    elif ci_mean['q_plat'][0] > 0:
        grade = 'A'
    else:
        grade = 'B'

    def per_seed(s):
        return {'P0': point[(s, 'P0')], 'P3': point[(s, 'P3')], 'diff_P3_minus_P0': diff_seed[s], 'ci95_diff': ci_seed[s]}
    write_json(out / 'primary_p3_vs_p0.json', {
        'definition': {
            'q_sel': 'QWK at the selected epoch', 'q_plat': 'arithmetic mean of QWK over epochs 11-20',
            'auc_sel': 'KL>=2 AUC at the selected epoch', 'x2_sel': 'cross-two-grade error rate at the selected epoch',
            'estimand': 'mean of the P3-P0 differences over three seeds',
            'bootstrap': f'patient-clustered, {N_PATIENTS} patients resampled with replacement as {N_PATIENTS} patients, {N_BOOT} times, '
                         f'numpy.random.default_rng({BOOT_SEED}).integers(0,{N_PATIENTS},size=({N_BOOT},{N_PATIENTS})), '
                         'patients numbered by ascending patient_id string; one set of multiplicities is used for all models, seeds and epochs; percentile interval 2.5%-97.5%',
            'scope': 'the interval reflects only validation-patient sampling error given these trained models and their selected epochs, '
                     'not training randomness, epoch selection or label uncertainty. The SD across three seeds is not a confidence interval.',
        },
        'per_seed': {str(s): per_seed(s) for s in SEEDS},
        'mean_diff': diff_mean, 'sd_diff_across_seeds_ddof1': diff_sd, 'ci95_mean_diff': ci_mean,
        'undefined_resamples': undefined, 'implementation_check_first_resample': verify,
        'gates_on_three_seed_mean': gates_mean, 'all_gates_pass': all_gates,
        'plateau_diff_all_seeds_positive': plat_all_positive, 'selected_diff_all_seeds_positive': sel_all_positive,
        'grade': grade,
        'grade_logic': 'D: D_sel or D_plat not positive; C: both positive, but a gate failed or a seed had a non-positive selected-epoch/plateau difference; '
                       'A: all four gates passed, three seeds agree in direction and the lower 95% bound of D_plat > 0; B: otherwise (interval includes 0).'
                       'per plan section 6.2; direction agreement is checked for both the selected epoch and the plateau.',
    })
    hand = {}
    for group in GROUPS:
        for k, expected in HAND_SEED17[group].items():
            value = point[(17, group)][k]
            hand[f'{group}.{k}'] = {'hand': expected, 'script': value,
                                    'match_4dp': (value == expected) if k == 'best_epoch' else round(value, 4) == expected}
    for k, expected in HAND_SEED17['diff'].items():
        hand[f'diff.{k}'] = {'hand': expected, 'script': diff_seed[17][k], 'match_4dp': round(diff_seed[17][k], 4) == expected}
    write_json(out / 'seed17_gates.json', {
        'P0': point[(17, 'P0')], 'P3': point[(17, 'P3')], 'diff_P3_minus_P0': diff_seed[17],
        'ci95_diff': ci_seed[17], 'gates': gates(diff_seed[17]),
        'all_gates_pass': all(g['pass'] for g in gates(diff_seed[17]).values()),
        'handcalc_check_plan_3.6': hand, 'handcalc_all_match': all(h['match_4dp'] for h in hand.values()),
    })
    write_json(out / 'curves.json', {f'seed{s}_{g}': [float(v) for v in qwk_all[keys.index((s, g))]]
                                     for s in SEEDS for g in GROUPS})
    code_path = Path(__file__).resolve()
    lines = ['# WP1 primary analysis output', '',
             f'- Start: {started}', f'- End: {time.strftime("%Y-%m-%d %H:%M:%S %Z")}',
             f'- Python {platform.python_version()}, numpy {np.__version__}',
             f'- Code: `{code_path.name}`, SHA256 `{sha256_file(code_path)}`',
             f'- Command: `{" ".join(sys.argv)}`',
             '- Reads existing train/val predictions only, CPU only; test not touched; outputs hold aggregate statistics only.', '',
             '## Input file SHA256', '']
    lines += [f'- `{name}` `{digest}`' for name, digest in sorted(input_hashes.items())]
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'selfcheck': selfcheck['pass'], 'grade': grade, 'mean_diff': diff_mean, 'ci95': ci_mean,
                      'undefined': undefined}, ensure_ascii=False))


if __name__ == '__main__':
    main()
