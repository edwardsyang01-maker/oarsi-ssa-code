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
ROOT_DEFAULT = Path('/path/to/drai')
sys.path.insert(0, str(HERE.parent.parent / 'analysis_v1/code'))
from wp1_primary import (B_REL, BOOT_SEED, CLASSES, EPOCHS, MANIFEST_REL, N_BOOT, N_KNEES, N_PATIENTS,  # noqa: E402
                         PLATEAU, TOL, WeightedAuc, auc_rank, cross2_from_confusion, interval, qwk_from_confusion,
                         sha256_file)
from wp2_secondary import (EXPECTED_SHA, OARSI_DIR, SWAP, TARGETS, oarsi_metrics, read_jsonl,  # noqa: E402
                           targets_from, write_json)

SEEDS = (17, 23, 29)
RUNS = {
    (17, 'P0'): 'attempt01/train_P0',
    (17, 'P1'): 'attempt02_lambda_v2/train_P1',
    (17, 'P2'): 'attempt02_lambda_v2/train_P2',
    (17, 'P3'): 'attempt02_lambda_v2/train_P3',
    (17, 'P4'): 'attempt02_lambda_v2/train_P4',
    (17, 'P3_lam050'): 'attempt05_x2_lam050/train_P3',
    (17, 'P3_lam200'): 'attempt05_x2_lam200/train_P3',
    (23, 'P0'): 'attempt03_seed23/train_P0',
    (23, 'P1'): 'attempt04_x1_seed23/train_P1',
    (23, 'P2'): 'attempt04_x1_seed23/train_P2',
    (23, 'P3'): 'attempt03_seed23/train_P3',
    (23, 'P4'): 'attempt05_x3_seed23/train_P4',
    (29, 'P0'): 'attempt03_seed29/train_P0',
    (29, 'P1'): 'attempt04_x1_seed29/train_P1',
    (29, 'P2'): 'attempt04_x1_seed29/train_P2',
    (29, 'P3'): 'attempt03_seed29/train_P3',
    (29, 'P4'): 'attempt05_x3_seed29/train_P4',
}
NEW_RUNS = {
    (23, 'P1'): 'launch_v4_x1_seed23/completion_verification_P1.json',
    (23, 'P2'): 'launch_v4_x1_seed23/completion_verification_P2.json',
    (29, 'P1'): 'launch_v4_x1_seed29/completion_verification_P1.json',
    (29, 'P2'): 'launch_v4_x1_seed29/completion_verification_P2.json',
    (17, 'P3_lam050'): 'launch_v5_x2_lam050/completion_verification_P3.json',
    (17, 'P3_lam200'): 'launch_v5_x2_lam200/completion_verification_P3.json',
    (23, 'P4'): 'launch_v5_x3_seed23/completion_verification_P4.json',
    (29, 'P4'): 'launch_v5_x3_seed29/completion_verification_P4.json',
}
LAMBDA_P3 = {'P3_lam050': 0.1474783234533083, 'P3': 0.2949566469066166, 'P3_lam200': 0.5899132938132332}
METRICS = ('q_sel', 'q_plat', 'auc_sel', 'x2_sel')
METRIC_NOTE = {'q_sel': 'QWK at the selected epoch', 'q_plat': 'mean QWK over epochs 11-20', 'auc_sel': 'KL>=2 AUC at the selected epoch (score p2+p3+p4)',
               'x2_sel': 'cross-two-grade error rate at the selected epoch'}
OUTPUTS = ('selfcheck_wp8.json', 'x1_factorial_three_seeds.json', 'x2_lambda_sensitivity.json',
           'x3_p4_three_seeds.json', 'curves_wp8.json', 'README_wp8.md')


def group_of(label):
    return label.split('_')[0]


def has_oarsi(label):
    return group_of(label) in ('P2', 'P3', 'P4')


def has_flip(label):
    return group_of(label) in ('P3', 'P4')


def effects(v):
    return {'pooling_main': ((v['P1'] - v['P0']) + (v['P3'] - v['P2'])) / 2,
            'auxiliary_main': ((v['P2'] - v['P0']) + (v['P3'] - v['P1'])) / 2,
            'interaction': v['P3'] - v['P2'] - v['P1'] + v['P0'],
            'simple_P1_minus_P0': v['P1'] - v['P0'], 'simple_P3_minus_P2': v['P3'] - v['P2'],
            'simple_P2_minus_P0': v['P2'] - v['P0'], 'simple_P3_minus_P1': v['P3'] - v['P1']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT_DEFAULT))
    parser.add_argument('--wp1-out', required=True, help='WP1/WP2 output directories (read-only, for exact cross-checks)')
    parser.add_argument('--out', required=True, help='output directory (must be empty or absent)')
    args = parser.parse_args()
    root, out, wp1_out = Path(args.root), Path(args.out), Path(args.wp1_out)
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f'output directory is not empty; refusing to overwrite: {out}'
    started = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    base = root / B_REL
    hashes = {}

    rows = read_jsonl(root / MANIFEST_REL, ('key', 'patient_id', 'split', 'visit', 'clinical_side', 'kl'))
    labels = read_jsonl(root / f'{OARSI_DIR}/val_oarsi.jsonl', ('key', 'kl', 'clinical_side', 'visit', *TARGETS))
    targets = targets_from(rows, labels)
    names = sorted({r['patient_id'] for r in rows})
    lookup = {name: i for i, name in enumerate(names)}
    patient_index = np.array([lookup[r['patient_id']] for r in rows])
    kl = np.array([int(r['kl']) for r in rows])
    assert all(r['split'] == 'val' for r in rows) and len(rows) == N_KNEES and len(names) == N_PATIENTS
    for rel in (MANIFEST_REL, f'{OARSI_DIR}/val_oarsi.jsonl'):
        hashes[rel] = sha256_file(root / rel)

    keys = list(RUNS)
    index = {key: i for i, key in enumerate(keys)}
    qwk_all = np.zeros((len(keys), EPOCHS))
    patient_conf = np.zeros((len(keys), EPOCHS, N_PATIENTS, CLASSES * CLASSES))
    best, sel, selfcheck = {}, {}, {'runs': {}}
    for r, key in enumerate(keys):
        rel, label = RUNS[key], key[1]
        run = base / rel
        for name in ('epoch_history.json', 'completed.json'):
            hashes[f'{B_REL}/{rel}/{name}'] = sha256_file(run / name)
        history = json.load(open(run / 'epoch_history.json'))
        completed = json.load(open(run / 'completed.json'))
        ok = {'history_rows': [h['epoch'] for h in history] == list(range(1, EPOCHS + 1)),
              'completed_epochs': completed['epochs'] == EPOCHS, 'test_accessed_false': completed['test_accessed'] is False,
              'ids': True, 'labels': True}
        if key in NEW_RUNS:
            receipt = base / NEW_RUNS[key]
            hashes[f'{B_REL}/{NEW_RUNS[key]}'] = sha256_file(receipt)
            v = json.load(open(receipt))
            ok['queue_completion_verified'] = (v['epochs'] == EPOCHS and v['best_epoch'] == completed['best_epoch']
                                               and v['same_disk_copies_verified'] is True)
        diffs = {'qwk': 0.0, 'auc_kl_ge2': 0.0, 'cross_two_error_rate': 0.0}
        for epoch in range(1, EPOCHS + 1):
            name = f'val_epoch{epoch:02d}.npz'
            hashes[f'{B_REL}/{rel}/{name}'] = sha256_file(run / name)
            z = np.load(run / name)
            y, prob = z['labels'].astype(int), z['probabilities']
            ok['ids'] &= bool(np.array_equal(z['ids'], np.arange(N_KNEES)))
            ok['labels'] &= bool(np.array_equal(y, kl))
            np.add.at(patient_conf[r, epoch - 1], (patient_index, y * CLASSES + prob.argmax(1)), 1)
            o = patient_conf[r, epoch - 1].sum(0).reshape(CLASSES, CLASSES)
            qwk_all[r, epoch - 1] = float(qwk_from_confusion(o))
            v = history[epoch - 1]['validation']
            diffs['qwk'] = max(diffs['qwk'], abs(qwk_all[r, epoch - 1] - v['qwk']))
            diffs['auc_kl_ge2'] = max(diffs['auc_kl_ge2'], abs(auc_rank(prob[:, 2:].sum(1), y >= 2) - v['auc_kl_ge2']))
            diffs['cross_two_error_rate'] = max(diffs['cross_two_error_rate'], abs(float(cross2_from_confusion(o)) - v['cross_two_error_rate']))
            if has_flip(label):
                hashes[f'{B_REL}/{rel}/val_extra_flip_epoch{epoch:02d}.npz'] = sha256_file(run / f'val_extra_flip_epoch{epoch:02d}.npz')
        best[key] = int(np.argmax(qwk_all[r])) + 1
        ok['best_epoch'] = best[key] == completed['best_epoch']
        ok['metrics'] = all(d <= TOL for d in diffs.values())
        selfcheck['runs'][f'seed{key[0]}_{label}'] = {'pass': all(ok.values()), 'checks': ok, 'best_epoch': best[key],
                                                      'max_abs_diff': diffs, 'path': rel}
        z = np.load(run / f'val_epoch{best[key]:02d}.npz')
        sel[key] = {'prob': z['probabilities']}
        if has_oarsi(label):
            sel[key]['oarsi'] = z['oarsi_probabilities']
        if has_flip(label):
            sel[key]['oarsi_flip'] = np.load(run / f'val_extra_flip_epoch{best[key]:02d}.npz')['oarsi_probabilities']
    selfcheck['input_sha256_ok'] = all(hashes[rel] == EXPECTED_SHA[rel] for rel in (MANIFEST_REL, f'{OARSI_DIR}/val_oarsi.jsonl'))

    plateau_idx = [e - 1 for e in PLATEAU]
    sel_idx = np.array([best[key] - 1 for key in keys])
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, N_PATIENTS, size=(N_BOOT, N_PATIENTS))
    weights = np.zeros((N_BOOT, N_PATIENTS))
    np.add.at(weights, (np.arange(N_BOOT)[:, None], draws), 1)
    conf_matrix = patient_conf.transpose(2, 0, 1, 3).reshape(N_PATIENTS, -1)
    aucs = {key: WeightedAuc(sel[key]['prob'][:, 2:].sum(1), kl >= 2) for key in keys}
    boot = {m: np.zeros((N_BOOT, len(keys))) for m in METRICS}
    every = np.arange(len(keys))
    for start in range(0, N_BOOT, 500):
        part = slice(start, start + 500)
        w = weights[part]
        o = (w @ conf_matrix).reshape(len(w), len(keys), EPOCHS, CLASSES, CLASSES)
        q = qwk_from_confusion(o)
        boot['q_sel'][part], boot['q_plat'][part] = q[:, every, sel_idx], q[:, :, plateau_idx].mean(2)
        boot['x2_sel'][part] = cross2_from_confusion(o[:, every, sel_idx])
        knee_w = w[:, patient_index]
        for key in keys:
            boot['auc_sel'][part, index[key]] = aucs[key](knee_w)
    selfcheck['undefined_resamples'] = {m: int(np.isnan(v).any(1).sum()) for m, v in boot.items()}

    point = {}
    for r, key in enumerate(keys):
        o = patient_conf[r, best[key] - 1].sum(0).reshape(CLASSES, CLASSES)
        point[key] = {'best_epoch': best[key], 'q_sel': float(qwk_all[r, best[key] - 1]),
                      'q_plat': float(qwk_all[r, plateau_idx].mean()),
                      'auc_sel': auc_rank(sel[key]['prob'][:, 2:].sum(1), kl >= 2),
                      'x2_sel': float(cross2_from_confusion(o)), 'kl2_as_kl0_count': int(o[2, 0])}
        if has_oarsi(key[1]):
            heads = oarsi_metrics(sel[key]['oarsi'], targets)
            point[key]['oarsi'] = heads
            point[key]['oarsi_mean_qwk'] = float(np.mean([heads[t]['qwk'] for t in TARGETS]))

    def bd(m, key):
        return boot[m][:, index[key]]

    wp1 = json.load(open(wp1_out / 'primary_p3_vs_p0.json'))
    rep = 0.0
    for m in METRICS:
        dist = np.mean([bd(m, (s, 'P3')) - bd(m, (s, 'P0')) for s in SEEDS], axis=0)
        rep = max(rep, max(abs(a - b) for a, b in zip(interval(dist), wp1['ci95_mean_diff'][m])))
        rep = max(rep, abs(np.mean([point[(s, 'P3')][m] - point[(s, 'P0')][m] for s in SEEDS]) - wp1['mean_diff'][m]))
    selfcheck['wp1_reproduced_max_abs_diff'] = rep
    fac = json.load(open(wp1_out / 'factorial_seed17.json'))
    rep2 = 0.0
    for m in ('q_sel', 'q_plat'):
        est = effects({g: point[(17, g)][m] for g in ('P0', 'P1', 'P2', 'P3')})
        dist = effects({g: bd(m, (17, g)) for g in ('P0', 'P1', 'P2', 'P3')})
        for k in est:
            rep2 = max(rep2, abs(est[k] - fac[m]['effects'][k]['estimate']),
                       *[abs(a - b) for a, b in zip(interval(dist[k]), fac[m]['effects'][k]['ci95'])])
    selfcheck['wp2_factorial_seed17_reproduced_max_abs_diff'] = rep2
    lat17 = json.load(open(wp1_out / 'p4_laterality.json'))['laterality']['seed17_P4']['overall']

    def flip(key):
        a, b = sel[key]['oarsi'].argmax(-1), sel[key]['oarsi_flip'].argmax(-1)
        return a, b
    a, b = flip((17, 'P4'))
    rep3 = max(abs(float((a == b).mean()) - lat17['same']), abs(float((a[:, SWAP] == b).mean()) - lat17['swapped']))
    selfcheck['wp2_p4_laterality_reproduced_max_abs_diff'] = rep3
    selfcheck['pass'] = bool(selfcheck['input_sha256_ok'] and rep <= 1e-12 and rep2 <= 1e-12 and rep3 == 0
                             and all(c['pass'] for c in selfcheck['runs'].values())
                             and not any(selfcheck['undefined_resamples'].values()))
    write_json(out / 'selfcheck_wp8.json', selfcheck)
    if not selfcheck['pass']:
        print('self-check failed; stopping. See selfcheck_wp8.json', file=sys.stderr)
        sys.exit(2)

    scope = ('Intervals are patient-clustered bootstrap (as in WP1) and reflect only validation-patient sampling error given these trained models and their selected epochs, '
             'not training randomness, epoch selection or label uncertainty. The SD across three seeds is not a confidence interval. All descriptive.')

    x1 = {'note': 'Amendment v4: three-seed 2x2 factorial after adding seeds 23 and 29 for P1 and P2. Effects are computed within each seed by the frozen-plan formulas, then averaged over the three seeds; '
                  'intervals use the same bootstrap multiplicities, computing effects within each seed before averaging. Post hoc supplement, descriptive on validation only, '
                  'without changing the WP1 primary comparison, the A/B/C/D grading or the test conclusions.' + scope,
          'metrics': METRIC_NOTE, 'formula': 'pooling main effect=((P1-P0)+(P3-P2))/2; auxiliary main effect=((P2-P0)+(P3-P1))/2; interaction=P3-P2-P1+P0'}
    groups = ('P0', 'P1', 'P2', 'P3')
    for m in METRICS:
        per_seed, per_seed_dist = {}, {}
        for s in SEEDS:
            est = effects({g: point[(s, g)][m] for g in groups})
            dist = effects({g: bd(m, (s, g)) for g in groups})
            per_seed[s] = {'values': {g: point[(s, g)][m] for g in groups},
                           'effects': {k: {'estimate': est[k], 'ci95': interval(dist[k])} for k in est}}
            per_seed_dist[s] = dist
        mean = {}
        for k in per_seed[17]['effects']:
            ests = [per_seed[s]['effects'][k]['estimate'] for s in SEEDS]
            mean[k] = {'estimate': float(np.mean(ests)), 'ci95': interval(np.mean([per_seed_dist[s][k] for s in SEEDS], axis=0)),
                       'per_seed': dict(zip(SEEDS, ests)), 'sd_across_seeds_ddof1': float(np.std(ests, ddof=1)),
                       'same_sign_all_seeds': bool(all(e > 0 for e in ests) or all(e < 0 for e in ests))}
        x1[m] = {'per_seed': per_seed, 'three_seed_mean': mean,
                 'group_means_across_seeds': {g: float(np.mean([point[(s, g)][m] for s in SEEDS])) for g in groups}}
    x1['best_epochs'] = {f'seed{s}_{g}': best[(s, g)] for s in SEEDS for g in groups}
    x1['oarsi_mean_qwk'] = {f'seed{s}_{g}': point[(s, g)]['oarsi_mean_qwk'] for s in SEEDS for g in ('P2', 'P3')}
    write_json(out / 'x1_factorial_three_seeds.json', x1)

    lam_keys = {'x0.5': (17, 'P3_lam050'), 'x1': (17, 'P3'), 'x2': (17, 'P3_lam200')}
    x2 = {'note': 'Amendment v5: P3 of seed17 changes only lambda (0.5x, original, 2x); everything else is byte-identical. One seed only; descriptive.' + scope,
          'lambda_P3': {k: LAMBDA_P3[key[1]] for k, key in lam_keys.items()}, 'metrics': METRIC_NOTE,
          'seed17_P0': point[(17, 'P0')]}
    for k, key in lam_keys.items():
        e = {'best_epoch': best[key], **{m: point[key][m] for m in METRICS}, 'kl2_as_kl0_count': point[key]['kl2_as_kl0_count'],
             'oarsi': point[key]['oarsi'], 'oarsi_mean_qwk': point[key]['oarsi_mean_qwk'],
             'minus_P0': {m: {'estimate': point[key][m] - point[(17, 'P0')][m],
                              'ci95': interval(bd(m, key) - bd(m, (17, 'P0')))} for m in METRICS}}
        if k != 'x1':
            e['minus_lambda_x1'] = {m: {'estimate': point[key][m] - point[(17, 'P3')][m],
                                        'ci95': interval(bd(m, key) - bd(m, (17, 'P3')))} for m in METRICS}
        fa, fb = flip(key)
        e['flip_consistency'] = {'same': float((fa == fb).mean()), 'swapped': float((fa[:, SWAP] == fb).mean())}
        x2[k] = e
    write_json(out / 'x2_lambda_sensitivity.json', x2)

    js_unequal = (targets[:, 4] >= 0) & (targets[:, 5] >= 0) & (targets[:, 4] != targets[:, 5])
    femoral = (targets[:, 0] >= 0) & (targets[:, 1] >= 0) & (targets[:, 0] != targets[:, 1])
    tibial = (targets[:, 2] >= 0) & (targets[:, 3] >= 0) & (targets[:, 2] != targets[:, 3])
    subsets = {'all_knees_all_heads': (np.ones(N_KNEES, bool), list(range(6))),
               'narrowing_jsm_ne_jsl': (js_unequal, [4, 5]),
               'osteophyte_osfm_ne_osfl_or_ostm_ne_ostl': (femoral | tibial, [0, 1, 2, 3])}

    def laterality(key):
        fa, fb = flip(key)
        res = {'per_head': {t: {'same': float((fa[:, j] == fb[:, j]).mean()), 'swapped': float((fa[:, SWAP[j]] == fb[:, j]).mean())}
                            for j, t in enumerate(TARGETS)}}
        for name, (mask, heads) in subsets.items():
            res[name] = {'knees': int(mask.sum()), 'same': float((fa[mask][:, heads] == fb[mask][:, heads]).mean()),
                         'swapped': float((fa[mask][:, SWAP][:, heads] == fb[mask][:, heads]).mean())}
        return res
    x3 = {'note': 'Amendment v5: P4 (medial/lateral labels not swapped) adds seeds 23 and 29, three seeds with seed17. P4-P3 and P4-P0 are differenced within each seed, then averaged over seeds.'
                  'Flip consistency: same = fraction of predictions for the same target unchanged after one more horizontal flip (high if anatomical orientation is learned); swapped = fraction equal to the prediction for the opposite-side target on the original image'
                  ' (high if a fixed image side is learned).' + scope, 'metrics': METRIC_NOTE, 'per_seed': {}}
    diffs = {'P4_minus_P3': {}, 'P4_minus_P0': {}}
    for s in SEEDS:
        x3['per_seed'][s] = {'P4': point[(s, 'P4')], 'P3': {m: point[(s, 'P3')][m] for m in (*METRICS, 'best_epoch', 'oarsi_mean_qwk')},
                             'P0': {m: point[(s, 'P0')][m] for m in (*METRICS, 'best_epoch')},
                             'laterality': {'P3': laterality((s, 'P3')), 'P4': laterality((s, 'P4'))}}
        for ref in ('P3', 'P0'):
            diffs[f'P4_minus_{ref}'][s] = {m: (point[(s, 'P4')][m] - point[(s, ref)][m], bd(m, (s, 'P4')) - bd(m, (s, ref))) for m in METRICS}
    for name, d in diffs.items():
        x3[name] = {m: {'per_seed': {s: {'estimate': d[s][m][0], 'ci95': interval(d[s][m][1])} for s in SEEDS},
                        'three_seed_mean': {'estimate': float(np.mean([d[s][m][0] for s in SEEDS])),
                                            'ci95': interval(np.mean([d[s][m][1] for s in SEEDS], axis=0)),
                                            'sd_across_seeds_ddof1': float(np.std([d[s][m][0] for s in SEEDS], ddof=1))}}
                    for m in METRICS}
    x3['laterality_three_seed_mean'] = {g: {sub: {k: float(np.mean([x3['per_seed'][s]['laterality'][g][sub][k] for s in SEEDS]))
                                                  for k in ('same', 'swapped')} for sub in subsets} for g in ('P3', 'P4')}
    write_json(out / 'x3_p4_three_seeds.json', x3)

    write_json(out / 'curves_wp8.json', {f'seed{k[0]}_{k[1]}': qwk_all[index[k]] for k in keys})

    code = Path(__file__).resolve()
    lines = ['# WP8 analysis output (X1, X2, X3, validation)', '', f'- Start: {started}', f'- End: {time.strftime("%Y-%m-%d %H:%M:%S %Z")}',
             f'- Python {platform.python_version()}, numpy {np.__version__}',
             f'- Code: `{code.name}`, SHA256 `{sha256_file(code)}`; depends on `wp1_primary.py` SHA256 '
             f'`{sha256_file(HERE.parent.parent / "analysis_v1/code/wp1_primary.py")}`, `wp2_secondary.py` SHA256 '
             f'`{sha256_file(HERE.parent.parent / "analysis_v1/code/wp2_secondary.py")}`',
             f'- Command: `{" ".join(sys.argv)}`',
             '- Reads existing validation predictions, lists and OARSI labels only, CPU only; test not touched; outputs hold aggregate statistics only.', '', '## Input file SHA256', '']
    lines += [f'- `{n}` `{d}`' for n, d in sorted(hashes.items())]
    (out / 'README_wp8.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    summary = {'selfcheck': selfcheck['pass'],
               'x1_mean': {m: {k: round(x1[m]['three_seed_mean'][k]['estimate'], 4) for k in ('pooling_main', 'auxiliary_main', 'interaction')}
                           for m in ('q_sel', 'q_plat')},
               'x2': {k: {m: round(x2[k][m], 4) for m in ('q_sel', 'q_plat')} for k in lam_keys},
               'x3_P4_minus_P3': {m: round(x3['P4_minus_P3'][m]['three_seed_mean']['estimate'], 4) for m in ('q_sel', 'q_plat')}}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
