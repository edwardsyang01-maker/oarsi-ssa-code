#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT_DEFAULT = Path('/path/to/drai')
sys.path.insert(0, str(HERE.parent.parent / 'analysis_v1/code'))
from wp1_primary import B_REL, MANIFEST_REL, N_KNEES, TOL, sha256_file  # noqa: E402
from wp2_secondary import (EXPECTED_SHA, OARSI_DIR, SWAP, TARGETS, oarsi_metrics, read_jsonl,  # noqa: E402
                           targets_from, write_json)

SEEDS = (17, 23, 29)
RUNS = {
    (17, 'P2'): 'attempt02_lambda_v2/train_P2',
    (23, 'P2'): 'attempt04_x1_seed23/train_P2',
    (29, 'P2'): 'attempt04_x1_seed29/train_P2',
    (17, 'P3'): 'attempt02_lambda_v2/train_P3',
    (23, 'P3'): 'attempt03_seed23/train_P3',
    (29, 'P3'): 'attempt03_seed29/train_P3',
    (17, 'P4'): 'attempt02_lambda_v2/train_P4',
    (23, 'P4'): 'attempt05_x3_seed23/train_P4',
    (29, 'P4'): 'attempt05_x3_seed29/train_P4',
}
OUT_NAME = 'oarsi_heads_and_laterality_three_seeds.json'


def summarize_across_seeds(per_seed_vals):
    vals = [float(per_seed_vals[s]) for s in SEEDS]
    return {
        'mean': float(np.mean(vals)),
        'sd_ddof1': float(np.std(vals, ddof=1)),
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
        'per_seed': {str(s): float(per_seed_vals[s]) for s in SEEDS},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(ROOT_DEFAULT))
    parser.add_argument('--wp2-out', required=True, help='WP2 output directory (read-only, to check p4_laterality.json)')
    parser.add_argument('--wp8-out', required=True, help='WP8 output directory (reads x1/x3 for checks and writes new JSON)')
    args = parser.parse_args()
    root, wp2_out, wp8_out = Path(args.root), Path(args.wp2_out), Path(args.wp8_out)
    out_path = wp8_out / OUT_NAME
    assert not out_path.exists(), f'refusing to overwrite an existing file: {out_path}'

    started = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    base = root / B_REL
    hashes = {}

    rows = read_jsonl(root / MANIFEST_REL, ('key', 'patient_id', 'split', 'visit', 'clinical_side', 'kl'))
    labels = read_jsonl(root / f'{OARSI_DIR}/val_oarsi.jsonl', ('key', 'kl', 'clinical_side', 'visit', *TARGETS))
    targets = targets_from(rows, labels)
    assert len(rows) == N_KNEES
    for rel in (MANIFEST_REL, f'{OARSI_DIR}/val_oarsi.jsonl'):
        hashes[rel] = sha256_file(root / rel)
        assert hashes[rel] == EXPECTED_SHA[rel], f'SHA256 mismatch: {rel}'

    best, sel, per_run_oarsi = {}, {}, {}
    max_hist_diff = 0.0
    for key, rel in RUNS.items():
        s, g = key
        run = base / rel
        for fn in ('epoch_history.json', 'completed.json'):
            hashes[f'{B_REL}/{rel}/{fn}'] = sha256_file(run / fn)
        history = json.load(open(run / 'epoch_history.json'))
        completed = json.load(open(run / 'completed.json'))
        be = int(completed['best_epoch'])
        best[key] = be
        npz_rel = f'{B_REL}/{rel}/val_epoch{be:02d}.npz'
        hashes[npz_rel] = sha256_file(root / npz_rel)
        z = np.load(root / npz_rel)
        oarsi_prob = z['oarsi_probabilities']
        sel[key] = {'oarsi': oarsi_prob}
        heads = oarsi_metrics(oarsi_prob, targets)
        v_oarsi = history[be - 1]['validation']['oarsi']
        for t in TARGETS:
            assert heads[t]['n'] == v_oarsi[t]['n']
            max_hist_diff = max(
                max_hist_diff,
                abs(heads[t]['qwk'] - v_oarsi[t]['qwk']),
                abs(heads[t]['accuracy'] - v_oarsi[t]['accuracy']),
            )
        per_run_oarsi[key] = heads
        if g in ('P3', 'P4'):
            flip_rel = f'{B_REL}/{rel}/val_extra_flip_epoch{be:02d}.npz'
            hashes[flip_rel] = sha256_file(root / flip_rel)
            sel[key]['oarsi_flip'] = np.load(root / flip_rel)['oarsi_probabilities']

    assert max_hist_diff <= TOL, f'difference from epoch_history exceeds the limit: {max_hist_diff}'

    oarsi_summary = {}
    for g in ('P2', 'P3', 'P4'):
        oarsi_summary[g] = {
            'per_seed': {str(s): per_run_oarsi[(s, g)] for s in SEEDS},
            'three_seed_summary': {},
            'mean_across_six_heads': summarize_across_seeds(
                {s: float(np.mean([per_run_oarsi[(s, g)][t]['qwk'] for t in TARGETS])) for s in SEEDS}
            ),
        }
        for t in TARGETS:
            n_val = per_run_oarsi[(17, g)][t]['n']
            assert all(per_run_oarsi[(s, g)][t]['n'] == n_val for s in SEEDS)
            oarsi_summary[g]['three_seed_summary'][t] = {
                'n': n_val,
                'qwk': summarize_across_seeds({s: per_run_oarsi[(s, g)][t]['qwk'] for s in SEEDS}),
                'accuracy': summarize_across_seeds({s: per_run_oarsi[(s, g)][t]['accuracy'] for s in SEEDS}),
            }

    js_unequal = (targets[:, 4] >= 0) & (targets[:, 5] >= 0) & (targets[:, 4] != targets[:, 5])
    femoral = (targets[:, 0] >= 0) & (targets[:, 1] >= 0) & (targets[:, 0] != targets[:, 1])
    tibial = (targets[:, 2] >= 0) & (targets[:, 3] >= 0) & (targets[:, 2] != targets[:, 3])
    subsets = {
        'all_knees_all_heads': (np.ones(N_KNEES, bool), [0, 1, 2, 3, 4, 5]),
        'all_knees_narrowing_heads': (np.ones(N_KNEES, bool), [4, 5]),
        'all_knees_osteophyte_heads': (np.ones(N_KNEES, bool), [0, 1, 2, 3]),
        'narrowing_jsm_ne_jsl': (js_unequal, [4, 5]),
        'osteophyte_osfm_ne_osfl_or_ostm_ne_ostl': (femoral | tibial, [0, 1, 2, 3]),
        'narrowing_jsl_gt_jsm': ((targets[:, 4] >= 0) & (targets[:, 5] > targets[:, 4]), [4, 5]),
    }
    swapped_targets = targets[:, SWAP]

    def compute_laterality(key):
        a = sel[key]['oarsi'].argmax(-1)
        b = sel[key]['oarsi_flip'].argmax(-1)  # [N, 6] prediction after flipping horizontally once more
        result = {
            'best_epoch': best[key],
            'overall': {'same': float((a == b).mean()), 'swapped': float((a[:, SWAP] == b).mean())},
            'per_head': {
                name: {
                    'same': float((a[:, j] == b[:, j]).mean()),
                    'swapped': float((a[:, SWAP[j]] == b[:, j]).mean()),
                }
                for j, name in enumerate(TARGETS)
            },
            'subsets': {},
        }
        for sub_name, (mask, heads) in subsets.items():
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
                    'swapped': float((a[read, SWAP[j]] == b[read, j]).mean()),
                }
            result['subsets'][sub_name] = {
                'knees': int(mask.sum()),
                'same': float((a[mask][:, heads] == b[mask][:, heads]).mean()),
                'swapped': float((a[mask][:, SWAP][:, heads] == b[mask][:, heads]).mean()),
                'heads': heads_out,
            }
        return result

    lat_per_run = {}
    for g in ('P3', 'P4'):
        for s in SEEDS:
            lat_per_run[f'seed{s}_{g}'] = compute_laterality((s, g))

    lat_three_seed = {}
    for g in ('P3', 'P4'):
        lat_three_seed[g] = {'subsets': {}, 'per_head': {}}
        for sub_name in subsets:
            knees_n = lat_per_run[f'seed17_{g}']['subsets'][sub_name]['knees']
            lat_three_seed[g]['subsets'][sub_name] = {
                'knees': knees_n,
                'same': summarize_across_seeds({s: lat_per_run[f'seed{s}_{g}']['subsets'][sub_name]['same'] for s in SEEDS}),
                'swapped': summarize_across_seeds({s: lat_per_run[f'seed{s}_{g}']['subsets'][sub_name]['swapped'] for s in SEEDS}),
                'heads': {},
            }
            for h_name in lat_per_run[f'seed17_{g}']['subsets'][sub_name]['heads']:
                lat_three_seed[g]['subsets'][sub_name]['heads'][h_name] = {
                    metric: summarize_across_seeds(
                        {s: lat_per_run[f'seed{s}_{g}']['subsets'][sub_name]['heads'][h_name][metric] for s in SEEDS}
                    )
                    for metric in (
                        'accuracy_original',
                        'accuracy_flipped_vs_true_label',
                        'accuracy_flipped_vs_contralateral_label',
                        'same',
                        'swapped',
                    )
                }
        for t in TARGETS:
            lat_three_seed[g]['per_head'][t] = {
                k: summarize_across_seeds({s: lat_per_run[f'seed{s}_{g}']['per_head'][t][k] for s in SEEDS})
                for k in ('same', 'swapped')
            }

    wp2_lat = json.load(open(wp2_out / 'p4_laterality.json'))
    x1_out = json.load(open(wp8_out / 'x1_factorial_three_seeds.json'))
    x3_out = json.load(open(wp8_out / 'x3_p4_three_seeds.json'))
    repro_diff = 0.0
    for s in SEEDS:
        for g in ('P2', 'P3'):
            repro_diff = max(
                repro_diff,
                abs(oarsi_summary[g]['mean_across_six_heads']['per_seed'][str(s)] - x1_out['oarsi_mean_qwk'][f'seed{s}_{g}']),
            )
        for t in TARGETS:
            repro_diff = max(
                repro_diff,
                abs(per_run_oarsi[(s, 'P3')][t]['qwk'] - wp2_lat['oarsi_heads_P3_all_seeds'][str(s)][t]['qwk']),
                abs(per_run_oarsi[(s, 'P4')][t]['qwk'] - x3_out['per_seed'][str(s)]['P4']['oarsi'][t]['qwk']),
            )
    for run_label in ('seed17_P3', 'seed23_P3', 'seed29_P3', 'seed17_P4'):
        for sub_name in ('all_knees_narrowing_heads', 'all_knees_osteophyte_heads', 'narrowing_jsm_ne_jsl',
                         'osteophyte_osfm_ne_osfl_or_ostm_ne_ostl', 'narrowing_jsl_gt_jsm'):
            for k in ('same', 'swapped'):
                repro_diff = max(
                    repro_diff,
                    abs(lat_per_run[run_label]['subsets'][sub_name][k] - wp2_lat['laterality'][run_label]['subsets'][sub_name][k]),
                )
    assert repro_diff == 0.0, f'differs from existing outputs: {repro_diff}'

    out_obj = {
        'note': 'WP8 supplementary summary: per-head OARSI metrics of P2, P3 and P4 over three seeds, and flip consistency of P3 and P4 over three seeds for all subsets (including jsl>jsm).'
                'Reads existing selected-epoch validation predictions only, CPU only, test not touched. Items overlapping p4_laterality.json, x1 and x3 match exactly (difference 0).',
        'started': started,
        'finished': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'selfcheck': {
            'pass': True,
            'max_abs_diff_vs_epoch_history': max_hist_diff,
            'max_abs_diff_vs_wp2_wp8_outputs': repro_diff,
            'best_epochs': {f'seed{s}_{g}': best[(s, g)] for (s, g) in RUNS},
        },
        'oarsi_heads': oarsi_summary,
        'laterality_per_run': lat_per_run,
        'laterality_three_seed_summary': lat_three_seed,
        'input_sha256': hashes,
    }
    write_json(out_path, out_obj)
    print(json.dumps({'pass': True, 'out': str(out_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
