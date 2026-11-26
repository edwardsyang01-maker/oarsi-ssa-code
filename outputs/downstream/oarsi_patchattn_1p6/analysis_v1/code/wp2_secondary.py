#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from wp1_primary import (B_REL, BOOT_SEED, CLASSES, EPOCHS, MANIFEST_REL, MANIFEST_SHA256, N_BOOT, N_KNEES,
                         N_PATIENTS, PLATEAU, TOL, WeightedAuc, auc_rank, cross2_from_confusion, interval,
                         qwk_from_confusion, sha256_file)

SEEDS = (17, 23, 29)
RUNS = {
    (17, 'P0'): 'attempt01/train_P0',
    (17, 'P1'): 'attempt02_lambda_v2/train_P1',
    (17, 'P2'): 'attempt02_lambda_v2/train_P2',
    (17, 'P3'): 'attempt02_lambda_v2/train_P3',
    (17, 'P4'): 'attempt02_lambda_v2/train_P4',
    (23, 'P0'): 'attempt03_seed23/train_P0',
    (23, 'P3'): 'attempt03_seed23/train_P3',
    (29, 'P0'): 'attempt03_seed29/train_P0',
    (29, 'P3'): 'attempt03_seed29/train_P3',
}
HAS_OARSI = ('P2', 'P3', 'P4')
HAS_FLIP = ('P3', 'P4')
MANIFEST_DIR = 'data/manifests/chen_yolo_usable448_1p6x_clean_v1_20261112'
OARSI_DIR = 'data/manifests/chen_yolo_1p6x_oarsi_aux_v1_20261113'
EXPECTED_SHA = {
    f'{MANIFEST_DIR}/train_usable.jsonl': '811046a7592e0da45ede08f45e297ea93949fa736ac395684c72e0683b292e16',
    f'{MANIFEST_DIR}/val_usable.jsonl': MANIFEST_SHA256,
    f'{OARSI_DIR}/train_oarsi.jsonl': '61898e235821c83fa96d5966154a0cd0bd0f524b29b4be883939895723c7c785',
    f'{OARSI_DIR}/val_oarsi.jsonl': '9ecf7b3e00bdb0b29cb9805d898553c018b2a77f83dd681b5219c0976280c939',
}
TARGETS = ['osfm', 'osfl', 'ostm', 'ostl', 'jsm', 'jsl']
SWAP = [1, 0, 3, 2, 5, 4]  # medial <-> lateral
ECE_BINS = 15
OUTPUT_NAMES = ('selfcheck_wp2.json', 'factorial_seed17.json', 'secondary_metrics.json', 'boundary_auc.json',
                'subgroups.json', 'cohort_table.json', 'p4_laterality.json', 'README_wp2.md')


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def write_json(path, obj):
    assert not Path(path).exists(), f'refusing to overwrite: {path}'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=2)
        f.write('\n')


def read_jsonl(path, fields):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            rows.append({k: row.get(k) for k in fields})
    return rows


def targets_from(rows, labels):
    assert len(rows) == len(labels), 'OARSI label row count does not match the list'
    out = np.full((len(rows), len(TARGETS)), -1, dtype=np.int64)
    for i, (row, lab) in enumerate(zip(rows, labels)):
        assert lab['key'] == row['key'] and int(lab['kl']) == int(row['kl'])
        assert lab['clinical_side'] == row['clinical_side'] and lab['visit'] == row['visit']
        for j, name in enumerate(TARGETS):
            if lab[name] is not None:
                assert lab[name] in (0, 1, 2, 3)
                out[i, j] = lab[name]
    return out


def confusion(y, pred, classes=CLASSES):
    o = np.zeros((classes, classes))
    np.add.at(o, (np.asarray(y, int), np.asarray(pred, int)), 1)
    return o


def qwk_general(o):
    o = np.asarray(o, float)
    c = o.shape[0]
    if o.sum() == 0:
        return float('nan')
    w = np.subtract.outer(np.arange(c), np.arange(c)) ** 2 / (c - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    denom = (w * e).sum()
    return float('nan') if denom == 0 else float(1 - (w * o).sum() / denom)


def conf_metrics(o):
    o = np.asarray(o, float)
    idx = np.arange(CLASSES)
    total = o.sum((-1, -2))
    diag, row, col = o[..., idx, idx], o.sum(-1), o.sum(-2)
    with np.errstate(invalid='ignore', divide='ignore'):
        precision = np.where(col > 0, diag / col, 0.0)
        recall = np.where(row > 0, diag / row, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
        kl2_rate = np.where(row[..., 2] > 0, o[..., 2, 0] / row[..., 2], np.nan)
    distance = np.abs(np.subtract.outer(idx, idx))
    return {'qwk': qwk_from_confusion(o), 'accuracy': diag.sum(-1) / total, 'mae': (o * distance).sum((-1, -2)) / total,
            'macro_f1': f1.mean(-1), 'kl1_precision': precision[..., 1], 'kl1_recall': recall[..., 1],
            'kl1_f1': f1[..., 1], 'kl2_as_kl0_rate': kl2_rate, 'cross_two_error_rate': cross2_from_confusion(o),
            'precision': precision, 'recall': recall, 'f1': f1}


def ece_parts(prob, labels, patient_index):
    confidence, correct = prob.max(1), prob.argmax(1) == labels
    bins = np.minimum((confidence * ECE_BINS).astype(int), ECE_BINS - 1)
    d = np.zeros((N_PATIENTS, ECE_BINS))
    np.add.at(d, (patient_index, bins), confidence - correct)
    return d


def oarsi_metrics(prob, targets):
    pred, out = prob.argmax(-1), {}
    for j, name in enumerate(TARGETS):
        m = targets[:, j] >= 0
        out[name] = {'n': int(m.sum()), 'qwk': qwk_general(confusion(targets[m, j], pred[m, j], 4)),
                     'accuracy': float((targets[m, j] == pred[m, j]).mean()) if m.any() else float('nan')}
    return out


def subset_metrics(y, prob, mask):
    y, p = y[mask], prob[mask]
    m = conf_metrics(confusion(y, p.argmax(1)))
    positive = y >= 2
    return {'accuracy': float(m['accuracy']), 'mae': float(m['mae']),
            'cross_two_error_rate': float(m['cross_two_error_rate']),
            'qwk': float(m['qwk']) if len(np.unique(y)) >= 2 else None,
            'auc_kl_ge2': auc_rank(p[:, 2:].sum(1), positive) if positive.any() and (~positive).any() else None}


def cohort(root, split, hashes):
    fields = ('key', 'patient_id', 'film_id', 'case_id', 'split', 'visit', 'modality', 'clinical_side', 'kl', 'cleanup_status')
    paths = {'usable': f'{MANIFEST_DIR}/{split}_usable.jsonl', 'held': f'{MANIFEST_DIR}/{split}_held.jsonl',
             'oarsi': f'{OARSI_DIR}/{split}_oarsi.jsonl'}
    for rel in paths.values():
        hashes[rel] = sha256_file(root / rel)
        assert rel not in EXPECTED_SHA or hashes[rel] == EXPECTED_SHA[rel], f'input file hashes differ from plan appendix A: {rel}'
    usable, held = read_jsonl(root / paths['usable'], fields), read_jsonl(root / paths['held'], fields)
    labels = read_jsonl(root / paths['oarsi'], ('key', 'kl', 'clinical_side', 'visit', *TARGETS))
    assert all(r['split'] == split for r in usable + held)
    t = targets_from(usable, labels)
    kl = np.array([r['kl'] for r in usable])
    all_ost, both_js = (t[:, :4] >= 0).all(1), (t[:, 4:] >= 0).all(1)
    held_kl = Counter(r['kl'] for r in held)
    per_patient = Counter(r['patient_id'] for r in usable)
    counts = np.array(list(per_patient.values()))
    return {
        'patients': len(per_patient), 'films_unique_film_id': len({r['film_id'] for r in usable}),
        'cases_unique_case_id': len({r['case_id'] for r in usable}), 'knees': len(usable),
        'knees_per_patient': {'min': int(counts.min()), 'median': float(np.median(counts)), 'max': int(counts.max())},
        'knees_by_side': dict(Counter(r['clinical_side'] for r in usable)),
        'knees_by_visit': dict(sorted(Counter(r['visit'] for r in usable).items())),
        'knees_by_modality': dict(sorted(Counter(r['modality'] for r in usable).items())),
        'knees_by_kl': {str(k): int((kl == k).sum()) for k in range(CLASSES)},
        'kl_fraction': {str(k): float((kl == k).mean()) for k in range(CLASSES)},
        'cleanup_status': dict(Counter(r['cleanup_status'] for r in usable)),
        'ruler_cleaned_fraction': float(np.mean([r['cleanup_status'] == 'accepted_cleaned' for r in usable])),
        'oarsi_present_per_target': {name: int((t[:, j] >= 0).sum()) for j, name in enumerate(TARGETS)},
        'all_four_osteophytes': {'n': int(all_ost.sum()), 'fraction': float(all_ost.mean())},
        'both_narrowing': {'n': int(both_js.sum()), 'fraction': float(both_js.mean())},
        'all_four_osteophytes_by_kl': {str(k): {'n': int(all_ost[kl == k].sum()), 'knees': int((kl == k).sum()),
                                                'fraction': float(all_ost[kl == k].mean())} for k in range(CLASSES)},
        'held': {'knees': len(held), 'patients_with_held_knees': len({r['patient_id'] for r in held}),
                 'fraction_of_all': len(held) / (len(held) + len(usable)),
                 'by_kl': {str(k): {'held': held_kl.get(k, 0), 'usable': int((kl == k).sum()),
                                    'held_fraction': held_kl.get(k, 0) / (held_kl.get(k, 0) + int((kl == k).sum()))}
                           for k in range(CLASSES)}},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='drai project root')
    parser.add_argument('--out', required=True, help='output directory (the WP1 output directory; existing files are not overwritten)')
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    for name in OUTPUT_NAMES:
        assert not (out / name).exists(), f'output already exists; refusing to overwrite: {name}'
    started = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    base = root / B_REL
    hashes = {}

    assert MANIFEST_REL == f'{MANIFEST_DIR}/val_usable.jsonl'
    rows = read_jsonl(root / MANIFEST_REL, ('key', 'patient_id', 'split', 'visit', 'clinical_side', 'kl', 'cleanup_status'))
    labels = read_jsonl(root / f'{OARSI_DIR}/val_oarsi.jsonl', ('key', 'kl', 'clinical_side', 'visit', *TARGETS))
    targets = targets_from(rows, labels)
    names = sorted({r['patient_id'] for r in rows})
    lookup = {name: i for i, name in enumerate(names)}
    patient_index = np.array([lookup[r['patient_id']] for r in rows])
    kl = np.array([int(r['kl']) for r in rows])
    assert len(rows) == N_KNEES and len(names) == N_PATIENTS

    keys = list(RUNS)
    index = {key: i for i, key in enumerate(keys)}
    qwk_all = np.zeros((len(keys), EPOCHS))
    ece_all = np.zeros((len(keys), EPOCHS))
    patient_conf = np.zeros((len(keys), EPOCHS, N_PATIENTS, CLASSES * CLASSES))
    patient_ece = np.zeros((len(keys), EPOCHS, N_PATIENTS, ECE_BINS))
    best, sel, selfcheck = {}, {}, {'runs': {}}
    for r, key in enumerate(keys):
        run, group = base / RUNS[key], key[1]
        for name in ('epoch_history.json', 'completed.json'):
            hashes[f'{B_REL}/{RUNS[key]}/{name}'] = sha256_file(run / name)
        history = json.load(open(run / 'epoch_history.json'))
        completed = json.load(open(run / 'completed.json'))
        diffs = Counter()
        ok = {'ids': True, 'labels': True, 'kl2_as_kl0_count': True, 'oarsi_n': True}

        def compare(name, value, recorded):
            if recorded is None or value != value:
                ok[f'{name}_defined'] = ok.get(f'{name}_defined', True) and (recorded is None) == (value != value)
            else:
                diffs[name] = max(diffs[name], abs(value - recorded))
        for epoch in range(1, EPOCHS + 1):
            name = f'val_epoch{epoch:02d}.npz'
            hashes[f'{B_REL}/{RUNS[key]}/{name}'] = sha256_file(run / name)
            z = np.load(run / name)
            y, prob = z['labels'].astype(int), z['probabilities']
            ok['ids'] &= bool(np.array_equal(z['ids'], np.arange(N_KNEES)))
            ok['labels'] &= bool(np.array_equal(y, kl))
            np.add.at(patient_conf[r, epoch - 1], (patient_index, y * CLASSES + prob.argmax(1)), 1)
            o = patient_conf[r, epoch - 1].sum(0).reshape(CLASSES, CLASSES)
            m = conf_metrics(o)
            patient_ece[r, epoch - 1] = ece_parts(prob, y, patient_index)
            qwk_all[r, epoch - 1] = m['qwk']
            ece_all[r, epoch - 1] = np.abs(patient_ece[r, epoch - 1].sum(0)).sum() / N_KNEES
            v = history[epoch - 1]['validation']
            for metric in ('qwk', 'accuracy', 'mae', 'macro_f1', 'cross_two_error_rate'):
                compare(metric, float(m[metric]), v[metric])
            compare('ece', float(ece_all[r, epoch - 1]), v['ece'])
            compare('auc_kl_ge2', auc_rank(prob[:, 2:].sum(1), y >= 2), v['auc_kl_ge2'])
            for part in ('precision', 'recall', 'f1'):
                compare(f'kl1_{part}', float(m[f'kl1_{part}']), v['per_class']['1'][part])
            ok['kl2_as_kl0_count'] &= int(o[2, 0]) == v['kl2_as_kl0_count']
            if group in HAS_OARSI:
                mine = oarsi_metrics(z['oarsi_probabilities'], targets)
                for target in TARGETS:
                    ok['oarsi_n'] &= mine[target]['n'] == v['oarsi'][target]['n']
                    compare('oarsi_qwk', mine[target]['qwk'], v['oarsi'][target]['qwk'])
                    compare('oarsi_accuracy', mine[target]['accuracy'], v['oarsi'][target]['accuracy'])
            if group in HAS_FLIP:
                name = f'val_extra_flip_epoch{epoch:02d}.npz'
                hashes[f'{B_REL}/{RUNS[key]}/{name}'] = sha256_file(run / name)
                a = z['oarsi_probabilities'].argmax(-1)
                b = np.load(run / name)['oarsi_probabilities'].argmax(-1)
                recorded = history[epoch - 1]['flip_consistency']
                compare('flip_same', float((a == b).mean()), recorded['same'])
                compare('flip_swapped', float((a[:, SWAP] == b).mean()), recorded['swapped'])
        best[key] = int(np.argmax(qwk_all[r])) + 1
        ok['best_epoch'] = best[key] == completed['best_epoch']
        ok['metrics'] = all(d <= TOL for d in diffs.values())
        selfcheck['runs'][f'seed{key[0]}_{group}'] = {'pass': all(ok.values()), 'checks': ok, 'best_epoch': best[key],
                                                      'max_abs_diff': dict(diffs)}
        z = np.load(run / f'val_epoch{best[key]:02d}.npz')
        sel[key] = {'prob': z['probabilities']}
        if group in HAS_OARSI:
            sel[key]['oarsi'] = z['oarsi_probabilities']
        if group in HAS_FLIP:
            sel[key]['oarsi_flip'] = np.load(run / f'val_extra_flip_epoch{best[key]:02d}.npz')['oarsi_probabilities']
    hashes[MANIFEST_REL] = sha256_file(root / MANIFEST_REL)
    hashes[f'{OARSI_DIR}/val_oarsi.jsonl'] = sha256_file(root / f'{OARSI_DIR}/val_oarsi.jsonl')
    selfcheck['input_sha256_ok'] = all(hashes[rel] == EXPECTED_SHA[rel] for rel in (MANIFEST_REL, f'{OARSI_DIR}/val_oarsi.jsonl'))

    plateau_idx = [e - 1 for e in PLATEAU]
    sel_idx = np.array([best[key] - 1 for key in keys])
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, N_PATIENTS, size=(N_BOOT, N_PATIENTS))
    weights = np.zeros((N_BOOT, N_PATIENTS))
    np.add.at(weights, (np.arange(N_BOOT)[:, None], draws), 1)
    knees_per_patient = np.bincount(patient_index, minlength=N_PATIENTS).astype(float)
    conf_matrix = patient_conf.transpose(2, 0, 1, 3).reshape(N_PATIENTS, -1)
    ece_matrix = patient_ece.transpose(2, 0, 1, 3).reshape(N_PATIENTS, -1)
    scalar = ('accuracy', 'mae', 'macro_f1', 'kl1_precision', 'kl1_recall', 'kl1_f1', 'kl2_as_kl0_rate', 'cross_two_error_rate')
    boot = {name: np.zeros((N_BOOT, len(keys))) for name in ('q_sel', 'q_plat', 'ece_sel', 'ece_plat', *scalar)}
    boot_q = np.zeros((N_BOOT, len(keys), EPOCHS))
    main_keys = [(s, g) for s in SEEDS for g in ('P0', 'P3')]
    boundary = {(key, k): WeightedAuc(sel[key]['prob'][:, k:].sum(1), kl >= k) for key in main_keys for k in range(1, CLASSES)}
    boot_auc = {pair: np.zeros(N_BOOT) for pair in boundary}
    every = np.arange(len(keys))
    for start in range(0, N_BOOT, 500):
        part = slice(start, start + 500)
        w = weights[part]
        n = w @ knees_per_patient
        o = (w @ conf_matrix).reshape(len(w), len(keys), EPOCHS, CLASSES, CLASSES)
        q = qwk_from_confusion(o)
        boot_q[part] = q
        boot['q_sel'][part], boot['q_plat'][part] = q[:, every, sel_idx], q[:, :, plateau_idx].mean(2)
        m = conf_metrics(o[:, every, sel_idx])
        for name in scalar:
            boot[name][part] = m[name]
        e = np.abs((w @ ece_matrix).reshape(len(w), len(keys), EPOCHS, ECE_BINS)).sum(-1) / n[:, None, None]
        boot['ece_sel'][part], boot['ece_plat'][part] = e[:, every, sel_idx], e[:, :, plateau_idx].mean(2)
        knee_w = w[:, patient_index]
        for pair, fn in boundary.items():
            boot_auc[pair][part] = fn(knee_w)

    def diff(name, a, b):
        return boot[name][:, index[a]] - boot[name][:, index[b]]

    def mean_diff(name):
        return np.mean([diff(name, (s, 'P3'), (s, 'P0')) for s in SEEDS], axis=0)

    wp1 = json.load(open(out / 'primary_p3_vs_p0.json'))
    reproduced = max(abs(a - b) for name in ('q_sel', 'q_plat')
                     for a, b in zip(interval(mean_diff(name)), wp1['ci95_mean_diff'][name]))
    selfcheck['wp1_interval_reproduced_max_abs_diff'] = reproduced
    selfcheck['undefined_resamples'] = {name: int(np.isnan(v).any(1).sum()) for name, v in boot.items()}
    selfcheck['undefined_resamples']['boundary_auc'] = int(sum(np.isnan(v).sum() for v in boot_auc.values()))
    selfcheck['pass'] = bool(selfcheck['input_sha256_ok'] and reproduced <= 1e-12
                             and all(c['pass'] for c in selfcheck['runs'].values()))
    write_json(out / 'selfcheck_wp2.json', selfcheck)
    if not selfcheck['pass']:
        print('self-check failed; stopping. See selfcheck_wp2.json', file=sys.stderr)
        sys.exit(2)

    point = {}
    for r, key in enumerate(keys):
        o = patient_conf[r, best[key] - 1].sum(0).reshape(CLASSES, CLASSES)
        m = conf_metrics(o)
        prob = sel[key]['prob']
        point[key] = {'best_epoch': best[key], 'q_sel': float(m['qwk']), 'q_plat': float(qwk_all[r, plateau_idx].mean()),
                      **{name: float(m[name]) for name in scalar},
                      'kl2_as_kl0_count': int(o[2, 0]), 'auc_kl_ge2': auc_rank(prob[:, 2:].sum(1), kl >= 2),
                      'ece_sel': float(ece_all[r, best[key] - 1]), 'ece_plat': float(ece_all[r, plateau_idx].mean()),
                      'per_class': {str(c): {'precision': float(m['precision'][c]), 'recall': float(m['recall'][c]),
                                             'f1': float(m['f1'][c]), 'support': int(o[c].sum())} for c in range(CLASSES)},
                      'confusion_matrix_rows_true_cols_pred': o.astype(int).tolist()}
        if 'oarsi' in sel[key]:
            point[key]['oarsi'] = oarsi_metrics(sel[key]['oarsi'], targets)
    label = lambda key: f'seed{key[0]}_{key[1]}'

    def effects(v):
        return {'pooling_main': ((v['P1'] - v['P0']) + (v['P3'] - v['P2'])) / 2,
                'auxiliary_main': ((v['P2'] - v['P0']) + (v['P3'] - v['P1'])) / 2,
                'interaction': v['P3'] - v['P2'] - v['P1'] + v['P0'],
                'simple_P1_minus_P0': v['P1'] - v['P0'], 'simple_P3_minus_P2': v['P3'] - v['P2'],
                'simple_P2_minus_P0': v['P2'] - v['P0'], 'simple_P3_minus_P1': v['P3'] - v['P1']}
    factorial = {'note': 'Descriptive only, no significance claims; seed17 only, and the interval excludes training randomness.'
                         'Formulas: pooling main effect=((P1-P0)+(P3-P2))/2; auxiliary main effect=((P2-P0)+(P3-P1))/2; interaction=P3-P2-P1+P0.'}
    for name in ('q_sel', 'q_plat'):
        groups = ('P0', 'P1', 'P2', 'P3')
        est = effects({g: point[(17, g)][name] for g in groups})
        dist = effects({g: boot[name][:, index[(17, g)]] for g in groups})
        factorial[name] = {'values': {g: point[(17, g)][name] for g in groups},
                           'effects': {k: {'estimate': est[k], 'ci95': interval(dist[k])} for k in est}}
    write_json(out / 'factorial_seed17.json', factorial)

    compared = ('accuracy', 'mae', 'macro_f1', 'kl1_precision', 'kl1_recall', 'kl1_f1', 'kl2_as_kl0_rate',
                'cross_two_error_rate', 'ece_sel', 'ece_plat')
    p3_p0 = {name: {'per_seed': {s: point[(s, 'P3')][name] - point[(s, 'P0')][name] for s in SEEDS},
                    'mean': float(np.mean([point[(s, 'P3')][name] - point[(s, 'P0')][name] for s in SEEDS])),
                    'ci95': interval(mean_diff(name))} for name in compared}
    curve = np.array([qwk_all[index[(s, 'P3')]] - qwk_all[index[(s, 'P0')]] for s in SEEDS])
    transitions = {}
    for s in SEEDS:
        right0 = sel[(s, 'P0')]['prob'].argmax(1) == kl
        right3 = sel[(s, 'P3')]['prob'].argmax(1) == kl
        table = {'P0_wrong_P3_right': ~right0 & right3, 'P0_right_P3_wrong': right0 & ~right3,
                 'both_right': right0 & right3, 'both_wrong': ~right0 & ~right3}
        transitions[s] = {name: {'all': int(mask.sum()), **{f'kl{k}': int((mask & (kl == k)).sum()) for k in range(CLASSES)}}
                          for name, mask in table.items()}
    write_json(out / 'secondary_metrics.json', {
        'note': 'The frozen plan requires reporting only, with no gates. Each metric uses its own selected epoch; ece_plat and q_plat are means over epochs 11-20.'
                'kl2_as_kl0_rate = fraction of true KL2 predicted as KL0. Intervals are patient-clustered bootstrap (as in WP1) and reflect validation-patient sampling error only.',
        'per_run': {label(key): point[key] for key in keys},
        'P3_minus_P0': p3_p0,
        'P4_minus_P3_seed17': {name: {'estimate': point[(17, 'P4')][name] - point[(17, 'P3')][name],
                                      'ci95': interval(diff(name, (17, 'P4'), (17, 'P3')))}
                               for name in ('q_sel', 'q_plat', 'accuracy', 'mae', 'ece_plat')},
        'per_epoch': {'qwk': {label(key): qwk_all[r] for r, key in enumerate(keys)},
                      'ece': {label(key): ece_all[r] for r, key in enumerate(keys)},
                      'qwk_P3_minus_P0': {'per_seed': {s: curve[i] for i, s in enumerate(SEEDS)},
                                          'mean': curve.mean(0), 'min': curve.min(0), 'max': curve.max(0),
                                          'ci95_of_mean': [interval(np.mean([boot_q[:, index[(s, 'P3')], e] - boot_q[:, index[(s, 'P0')], e]
                                                                             for s in SEEDS], axis=0)) for e in range(EPOCHS)]}},
        'knee_level_transitions_at_selected_epoch': transitions,
    })

    boundary_out = {'note': 'AUC for KL>=k: score sum_{j>=k} p_j at each model selected epoch; independent of the argmax decision boundary.'}
    for k in range(1, CLASSES):
        value = {key: auc_rank(sel[key]['prob'][:, k:].sum(1), kl >= k) for key in main_keys}
        dist = {s: boot_auc[((s, 'P3'), k)] - boot_auc[((s, 'P0'), k)] for s in SEEDS}
        boundary_out[f'kl_ge_{k}'] = {
            'positives': int((kl >= k).sum()), 'negatives': int((kl < k).sum()),
            'per_seed': {s: {'P0': value[(s, 'P0')], 'P3': value[(s, 'P3')], 'diff': value[(s, 'P3')] - value[(s, 'P0')],
                             'ci95_diff': interval(dist[s])} for s in SEEDS},
            'mean_P0': float(np.mean([value[(s, 'P0')] for s in SEEDS])),
            'mean_P3': float(np.mean([value[(s, 'P3')] for s in SEEDS])),
            'mean_diff': float(np.mean([value[(s, 'P3')] - value[(s, 'P0')] for s in SEEDS])),
            'ci95_mean_diff': interval(np.mean([dist[s] for s in SEEDS], axis=0))}
    write_json(out / 'boundary_auc.json', boundary_out)

    visit = np.array([r['visit'] for r in rows])
    side = np.array([r['clinical_side'] for r in rows])
    cleanup = np.array([r['cleanup_status'] for r in rows])
    complete = (targets[:, :4] >= 0).all(1)
    masks = {**{f'true_kl{k}': kl == k for k in range(CLASSES)},
             'osteophytes_all_four_read': complete, 'osteophytes_incomplete': ~complete,
             'visit_V00': visit == 'V00', 'visit_followup': visit != 'V00',
             'knee_right': side == 'R', 'knee_left': side == 'L',
             'ruler_accepted_cleaned': cleanup == 'accepted_cleaned', 'ruler_accepted_unchanged': cleanup == 'accepted_unchanged'}
    subgroups = {'note': 'Descriptive only, no significance tests. Subgroups follow plan WP2 step 6 without additions or removals. Metrics use each model selected epoch; '
                         'QWK and AUC are undefined in subgroups with a single true grade and are recorded as null.'}
    for name, mask in masks.items():
        per = {s: {g: subset_metrics(kl, sel[(s, g)]['prob'], mask) for g in ('P0', 'P3')} for s in SEEDS}
        summary = {}
        for metric in ('accuracy', 'mae', 'cross_two_error_rate', 'qwk', 'auc_kl_ge2'):
            if per[17]['P0'][metric] is None:
                summary[metric] = None
                continue
            p0, p3 = [per[s]['P0'][metric] for s in SEEDS], [per[s]['P3'][metric] for s in SEEDS]
            summary[metric] = {'mean_P0': float(np.mean(p0)), 'mean_P3': float(np.mean(p3)),
                               'mean_diff': float(np.mean(p3) - np.mean(p0)),
                               'diff_per_seed': {s: b - a for s, a, b in zip(SEEDS, p0, p3)}}
        subgroups[name] = {'knees': int(mask.sum()), 'patients': int(len(np.unique(patient_index[mask]))),
                           'kl_counts': {str(k): int((mask & (kl == k)).sum()) for k in range(CLASSES)},
                           'three_seed_summary': summary, 'per_seed': per}
    write_json(out / 'subgroups.json', subgroups)

    write_json(out / 'cohort_table.json', {'train': cohort(root, 'train', hashes), 'val': cohort(root, 'val', hashes),
                                           'note': 'held = knees deferred in the 1.6x version; held_fraction = held/(held+usable).'})

    js_unequal = (targets[:, 4] >= 0) & (targets[:, 5] >= 0) & (targets[:, 4] != targets[:, 5])
    femoral = (targets[:, 0] >= 0) & (targets[:, 1] >= 0) & (targets[:, 0] != targets[:, 1])
    tibial = (targets[:, 2] >= 0) & (targets[:, 3] >= 0) & (targets[:, 2] != targets[:, 3])
    subsets = {'all_knees_narrowing_heads': (np.ones(N_KNEES, bool), [4, 5]),
               'all_knees_osteophyte_heads': (np.ones(N_KNEES, bool), [0, 1, 2, 3]),
               'narrowing_jsm_ne_jsl': (js_unequal, [4, 5]),
               'osteophyte_osfm_ne_osfl_or_ostm_ne_ostl': (femoral | tibial, [0, 1, 2, 3]),
               'narrowing_jsl_gt_jsm': ((targets[:, 4] >= 0) & (targets[:, 5] > targets[:, 4]), [4, 5])}
    swapped_targets = targets[:, SWAP]

    def laterality(key):
        a, b = sel[key]['oarsi'].argmax(-1), sel[key]['oarsi_flip'].argmax(-1)  # [N, 6] original image, and the image flipped once more
        result = {'best_epoch': best[key],
                  'overall': {'same': float((a == b).mean()), 'swapped': float((a[:, SWAP] == b).mean())},
                  'per_head': {name: {'same': float((a[:, j] == b[:, j]).mean()),
                                      'swapped': float((a[:, SWAP[j]] == b[:, j]).mean())} for j, name in enumerate(TARGETS)},
                  'prediction_distribution': {name: {
                      'original': np.bincount(a[:, j], minlength=4), 'flipped': np.bincount(b[:, j], minlength=4),
                      'labels_where_read': np.bincount(targets[targets[:, j] >= 0, j], minlength=4)} for j, name in enumerate(TARGETS)},
                  'subsets': {}}
        for name, (mask, heads) in subsets.items():
            heads_out = {}
            for j in heads:
                read = mask & (targets[:, j] >= 0)
                both = read & (swapped_targets[:, j] >= 0)
                heads_out[TARGETS[j]] = {
                    'n': int(read.sum()),
                    'accuracy_original': float((a[read, j] == targets[read, j]).mean()),
                    'accuracy_flipped_vs_true_label': float((b[read, j] == targets[read, j]).mean()),
                    'accuracy_flipped_vs_contralateral_label': float((b[both, j] == swapped_targets[both, j]).mean()),
                    'same': float((a[read, j] == b[read, j]).mean()),
                    'swapped': float((a[read, SWAP[j]] == b[read, j]).mean())}
            result['subsets'][name] = {'knees': int(mask.sum()),
                                       'same': float((a[mask][:, heads] == b[mask][:, heads]).mean()),
                                       'swapped': float((a[mask][:, SWAP][:, heads] == b[mask][:, heads]).mean()),
                                       'heads': heads_out}
        return result
    kl_names = ('best_epoch', 'q_sel', 'q_plat', 'accuracy', 'mae', 'macro_f1', 'auc_kl_ge2', 'cross_two_error_rate',
                'kl2_as_kl0_count', 'ece_sel', 'ece_plat')
    write_json(out / 'p4_laterality.json', {
        'note': 'Exploratory, reported only, not part of the gates. P4 has seed17 only. Validation images are normalized to right-knee orientation; flipped means flipped horizontally once more.'
                'same = fraction of predictions for the same target unchanged after flipping (high if anatomical orientation is learned); swapped = fraction equal to the prediction for the opposite-side target on the original image'
                ' (high if a fixed image side is learned). Subsets follow N4: JSN uses jsm!=jsl; osteophytes use osfm!=osfl or ostm!=ostl; '
                'jsl>jsm is also reported. The two all_knees_* entries are controls over all knees. accuracy_flipped_* describes predictions on flipped images.',
        'kl_metrics': {g: {name: point[(17, g)][name] for name in kl_names} for g in ('P3', 'P4')},
        'kl_P4_minus_P3_ci95': {name: interval(diff(name, (17, 'P4'), (17, 'P3'))) for name in ('q_sel', 'q_plat')},
        'oarsi_heads_seed17': {g: point[(17, g)]['oarsi'] for g in HAS_OARSI},
        'oarsi_heads_P3_all_seeds': {s: point[(s, 'P3')]['oarsi'] for s in SEEDS},
        'laterality': {label(key): laterality(key) for key in keys if key[1] in HAS_FLIP},
    })

    code = Path(__file__).resolve()
    lines = ['# WP2 output', '', f'- Start: {started}', f'- End: {time.strftime("%Y-%m-%d %H:%M:%S %Z")}',
             f'- Python {platform.python_version()}, numpy {np.__version__}',
             f'- Code: `{code.name}`, SHA256 `{sha256_file(code)}`; depends on `wp1_primary.py`, SHA256 `{sha256_file(code.parent / "wp1_primary.py")}`',
             f'- Command: `{" ".join(sys.argv)}`',
             '- Reads existing train/val predictions, lists and OARSI labels only, CPU only; test not touched; outputs hold aggregate statistics only.', '',
             '## Input file SHA256', '']
    lines += [f'- `{name}` `{digest}`' for name, digest in sorted(hashes.items())]
    assert not (out / 'README_wp2.md').exists()
    (out / 'README_wp2.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'selfcheck': selfcheck['pass'], 'reproduced_wp1': reproduced}, ensure_ascii=False))


if __name__ == '__main__':
    main()
