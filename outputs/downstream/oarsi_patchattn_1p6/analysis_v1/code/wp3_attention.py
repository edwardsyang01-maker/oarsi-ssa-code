#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

from wp1_primary import B_REL, MANIFEST_REL, MANIFEST_SHA256, N_KNEES, sha256_file
from wp2_secondary import EXPECTED_SHA, OARSI_DIR, TARGETS, clean, read_jsonl, targets_from

RUNS = {
    'seed17_P1': 'attempt02_lambda_v2/train_P1',
    'seed17_P3': 'attempt02_lambda_v2/train_P3',
    'seed17_P4': 'attempt02_lambda_v2/train_P4',
    'seed23_P3': 'attempt03_seed23/train_P3',
    'seed29_P3': 'attempt03_seed29/train_P3',
}
QUERIES = ['kl', *TARGETS]
GRID, PATCHES = 28, 784
PANEL_SEED, PANEL_IMAGES = 170330, 32
TRACK_EPOCHS = (1, 5, 10, 13, 20)
TOP_PATCHES = math.ceil(0.05 * PATCHES)  # top 5% of patches = 40
METRICS = ('right_half_fraction', 'vertical_centroid_row', 'entropy_nats', 'top5pct_share')
OUTPUT_NAMES = ('attention_summary.json', 'README_wp3.md')


def attention_metrics(attention):
    a = attention / attention.sum(-1, keepdims=True)
    grid = a.reshape(*a.shape[:2], GRID, GRID)
    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -np.where(a > 0, a * np.log(a), 0.0).sum(-1)
    return {'right_half_fraction': grid[..., GRID // 2:].sum((-1, -2)),
            'vertical_centroid_row': (grid.sum(-1) * np.arange(GRID)).sum(-1),
            'entropy_nats': entropy,
            'top5pct_share': np.sort(a, axis=-1)[..., -TOP_PATCHES:].sum(-1)}


def spread(x):
    x = np.asarray(x, float)
    return {'mean': float(x.mean()), 'min': float(x.min()), 'max': float(x.max()), 'sd': float(x.std(ddof=1)) if len(x) > 1 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='drai project root')
    parser.add_argument('--out', required=True, help='output directory (existing files are not overwritten)')
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    for name in OUTPUT_NAMES:
        assert not (out / name).exists(), f'output already exists; refusing to overwrite: {name}'
    started = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    base, hashes = root / B_REL, {}

    oarsi_rel = f'{OARSI_DIR}/val_oarsi.jsonl'
    hashes[MANIFEST_REL], hashes[oarsi_rel] = sha256_file(root / MANIFEST_REL), sha256_file(root / oarsi_rel)
    rows = read_jsonl(root / MANIFEST_REL, ('key', 'split', 'visit', 'clinical_side', 'kl'))
    labels = read_jsonl(root / oarsi_rel, ('key', 'kl', 'clinical_side', 'visit', *TARGETS))
    targets = targets_from(rows, labels)
    kl = np.array([int(r['kl']) for r in rows])
    panel = np.array(sorted(np.random.default_rng(PANEL_SEED).choice(N_KNEES, PANEL_IMAGES, replace=False).tolist()))
    panel_targets = targets[panel]
    check = {'input_sha256_ok': hashes[MANIFEST_REL] == MANIFEST_SHA256 and hashes[oarsi_rel] == EXPECTED_SHA[oarsi_rel],
             'knees_ok': len(rows) == N_KNEES, 'runs': {}}

    result = {'note': '32 fixed validation images (random seed 170330, set before training, not reselected by results); examples only.'
                      'Images are in right-knee orientation with the medial side on the image right; row 0 is at the top. Metric definitions are in metric_definition.'
                      'Conclusions can only state whether the attention distribution agrees with an anatomical region, not that the model bases its decision on that region.',
              'metric_definition': {
                  'right_half_fraction': 'fraction of attention in columns >= 14; expected high for medial queries (osfm, ostm, jsm) and low for lateral queries',
                  'vertical_centroid_row': 'vertical center of attention (row, 0=top, 27=bottom); expected smaller for femoral queries than for the tibial query on the same side',
                  'entropy_nats': f'entropy of attention (natural log); ln 784 = {math.log(PATCHES):.4f}',
                  'top5pct_share': f'attention fraction of the top {TOP_PATCHES} patches (top 5%); for a uniform distribution {TOP_PATCHES / PATCHES:.4f}'},
              'queries': QUERIES,
              'panel': {'images': PANEL_IMAGES, 'kl_counts': {str(k): int((kl[panel] == k).sum()) for k in range(5)},
                        'left_knees_flipped_to_right_orientation': int(sum(rows[i]['clinical_side'] == 'L' for i in panel)),
                        'site_grade_counts': {name: {'not_read': int((panel_targets[:, j] < 0).sum()),
                                                     **{f'grade{g}': int((panel_targets[:, j] == g).sum()) for g in range(4)}}
                                              for j, name in enumerate(TARGETS)}},
              'runs': {}}

    for label, rel in RUNS.items():
        run = base / rel
        hashes[f'{B_REL}/{rel}/completed.json'] = sha256_file(run / 'completed.json')
        best = json.load(open(run / 'completed.json'))['best_epoch']
        epochs = sorted(set(TRACK_EPOCHS) | {best})
        ok = {'ids': True, 'labels': True, 'sum_close_to_1': True, 'nonnegative': True, 'query_count': True}
        prob_diff, by_epoch, selected = 0.0, {}, None
        for epoch in epochs:
            name = f'attention_panel_epoch{epoch:02d}.npz'
            hashes[f'{B_REL}/{rel}/{name}'] = sha256_file(run / name)
            z = np.load(run / name)
            attention = z['attention']
            ok['ids'] &= bool(np.array_equal(z['ids'], panel))
            ok['labels'] &= bool(np.array_equal(z['labels'], kl[panel]))
            ok['sum_close_to_1'] &= bool(np.abs(attention.sum(-1) - 1).max() < 0.01)
            ok['nonnegative'] &= bool(attention.min() >= 0)
            ok['query_count'] &= attention.shape == (PANEL_IMAGES, 1 if label.endswith('P1') else len(QUERIES), PATCHES)
            full = np.load(run / f'val_epoch{epoch:02d}.npz')['probabilities'][panel]
            prob_diff = max(prob_diff, float(np.abs(full - z['probabilities']).max()))
            m = attention_metrics(attention)
            by_epoch[epoch] = {QUERIES[q]: {name: float(m[name][:, q].mean()) for name in METRICS}
                               for q in range(attention.shape[1])}
            if epoch == best:
                selected = (attention, m)
        check['runs'][label] = {'pass': all(ok.values()), 'checks': ok,
                                'max_abs_diff_panel_vs_full_validation_probabilities': prob_diff}
        attention, m = selected
        normalized = attention / attention.sum(-1, keepdims=True)
        entry = {'best_epoch': best,
                 'mean_attention_map_28x28': {QUERIES[q]: normalized[:, q].mean(0).reshape(GRID, GRID)
                                              for q in range(attention.shape[1])},
                 'metrics_at_selected_epoch': {QUERIES[q]: {name: spread(m[name][:, q]) for name in METRICS}
                                               for q in range(attention.shape[1])},
                 'metrics_mean_by_epoch': by_epoch}
        if attention.shape[1] == len(QUERIES):
            index = {name: q for q, name in enumerate(QUERIES)}
            right, row = m['right_half_fraction'], m['vertical_centroid_row']
            entry['expected_direction_checks'] = {
                'medial_query_more_right_than_lateral': {
                    f'{a}_vs_{b}': {'images_as_expected': int((right[:, index[a]] > right[:, index[b]]).sum()), 'images': PANEL_IMAGES,
                                    'mean_difference': float((right[:, index[a]] - right[:, index[b]]).mean())}
                    for a, b in (('osfm', 'osfl'), ('ostm', 'ostl'), ('jsm', 'jsl'))},
                'femoral_query_higher_than_tibial_same_side': {
                    f'{a}_vs_{b}': {'images_as_expected': int((row[:, index[a]] < row[:, index[b]]).sum()), 'images': PANEL_IMAGES,
                                    'mean_rows_tibial_minus_femoral': float((row[:, index[b]] - row[:, index[a]]).mean())}
                    for a, b in (('osfm', 'ostm'), ('osfl', 'ostl'))}}
            lesion = {}
            for j, name in enumerate(TARGETS):
                grade = panel_targets[:, j]
                lesion[name] = {f'grade{g}': {'images': int((grade == g).sum()),
                                              'entropy_nats_mean': float(m['entropy_nats'][grade == g, j + 1].mean()) if (grade == g).any() else None,
                                              'top5pct_share_mean': float(m['top5pct_share'][grade == g, j + 1].mean()) if (grade == g).any() else None}
                                for g in range(4)}
                for key, mask in (('grade0', grade == 0), ('grade_ge1', grade >= 1), ('not_read', grade < 0)):
                    lesion[name][f'{key}_summary'] = {'images': int(mask.sum()),
                                                      'entropy_nats_mean': float(m['entropy_nats'][mask, j + 1].mean()) if mask.any() else None,
                                                      'top5pct_share_mean': float(m['top5pct_share'][mask, j + 1].mean()) if mask.any() else None}
            entry['concentration_by_site_grade'] = lesion
        result['runs'][label] = entry
    check['pass'] = bool(check['input_sha256_ok'] and check['knees_ok'] and all(c['pass'] for c in check['runs'].values()))
    result['selfcheck'] = check
    if not check['pass']:
        print(json.dumps(clean(check), ensure_ascii=False, indent=1), file=sys.stderr)
        print('self-check failed; stopping without writing outputs.', file=sys.stderr)
        sys.exit(2)
    with open(out / 'attention_summary.json', 'w', encoding='utf-8') as f:
        json.dump(clean(result), f, ensure_ascii=False, indent=1)
        f.write('\n')
    code = Path(__file__).resolve()
    lines = ['# WP3 output', '', f'- Start: {started}', f'- End: {time.strftime("%Y-%m-%d %H:%M:%S %Z")}',
             f'- Python {platform.python_version()}, numpy {np.__version__}',
             f'- Code: `{code.name}`, SHA256 `{sha256_file(code)}`; depends on `wp1_primary.py` (`{sha256_file(code.parent / "wp1_primary.py")}`)'
             f'and `wp2_secondary.py` (`{sha256_file(code.parent / "wp2_secondary.py")}`)',
             f'- Command: `{" ".join(sys.argv)}`',
             '- Reads only the attention panels saved during training, validation predictions, lists and OARSI labels, CPU only; no images read; test not touched; outputs hold aggregate statistics only.', '',
             '## Input file SHA256', '']
    lines += [f'- `{name}` `{digest}`' for name, digest in sorted(hashes.items())]
    (out / 'README_wp3.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'selfcheck': True, 'panel_vs_full_prob_max_diff': {k: v['max_abs_diff_panel_vs_full_validation_probabilities']
                                                                        for k, v in check['runs'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
