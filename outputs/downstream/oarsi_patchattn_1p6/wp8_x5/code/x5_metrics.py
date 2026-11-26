#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/path/to/drai')
B = ROOT / 'outputs/downstream/oarsi_patchattn_1p6'
X = B / 'wp8_x5'
sys.path.insert(0, str(B / 'analysis_v1/code'))
from wp1_primary import MANIFEST_REL, sha256_file as sha  # noqa: E402
from wp2_secondary import OARSI_DIR, TARGETS, clean, read_jsonl, targets_from  # noqa: E402
from wp3_attention import METRICS, QUERIES, attention_metrics  # noqa: E402

PAIRS_LR = (('osfm', 'osfl'), ('ostm', 'ostl'), ('jsm', 'jsl'))
PAIRS_FT = (('osfm', 'ostm'), ('osfl', 'ostl'))


def spread(x):
    x = np.asarray(x, float)
    return {'mean': float(x.mean()), 'sd': float(x.std(ddof=1)), 'p5': float(np.percentile(x, 5)),
            'median': float(np.median(x)), 'p95': float(np.percentile(x, 95)), 'min': float(x.min()), 'max': float(x.max())}


def main():
    out = X / 'x5_attention_summary.json'
    assert not out.exists()
    rows = read_jsonl(ROOT / MANIFEST_REL, ('key', 'split', 'visit', 'clinical_side', 'kl'))
    labels = read_jsonl(ROOT / f'{OARSI_DIR}/val_oarsi.jsonl', ('key', 'kl', 'clinical_side', 'visit', *TARGETS))
    targets = targets_from(rows, labels)
    kl = np.array([int(r['kl']) for r in rows])
    result = {'note': 'All 3,962 validation knees; attention comes from the evaluate(attention=True) path, the same as for the training panels.'
                      'Conclusions can only state whether the attention distribution agrees with anatomical regions.', 'runs': {}}
    for f in sorted((X / 'attention').glob('*.npz')):
        z = np.load(f)
        att = z['attention']
        assert np.array_equal(z['ids'], np.arange(len(rows))) and np.array_equal(z['labels'], kl)
        m = attention_metrics(att)
        norm = att / att.sum(-1, keepdims=True)
        entry = {'input_sha256': sha(f), 'knees': int(len(att)),
                 'metrics': {QUERIES[q]: {k: spread(m[k][:, q]) for k in METRICS} for q in range(att.shape[1])},
                 'mean_attention_map_28x28': {QUERIES[q]: norm[:, q].mean(0).reshape(28, 28) for q in range(att.shape[1])}}
        if att.shape[1] == len(QUERIES):
            ix = {n: i for i, n in enumerate(QUERIES)}
            right, row = m['right_half_fraction'], m['vertical_centroid_row']
            checks = {}
            for a, b in PAIRS_LR:
                ok = right[:, ix[a]] > right[:, ix[b]]
                checks[f'{a}_more_right_than_{b}'] = {'fraction': float(ok.mean()), 'mean_difference': float((right[:, ix[a]] - right[:, ix[b]]).mean()),
                                                      'fraction_by_kl': {str(k): float(ok[kl == k].mean()) for k in range(5)}}
            for a, b in PAIRS_FT:
                ok = row[:, ix[a]] < row[:, ix[b]]
                checks[f'{a}_higher_than_{b}'] = {'fraction': float(ok.mean()), 'mean_rows': float((row[:, ix[b]] - row[:, ix[a]]).mean()),
                                                  'fraction_by_kl': {str(k): float(ok[kl == k].mean()) for k in range(5)}}
            entry['expected_direction_checks'] = checks
            lesion = {}
            for j, t in enumerate(TARGETS):
                g = targets[:, j]
                lesion[t] = {f'grade{v}': {'knees': int((g == v).sum()),
                                            'entropy_mean': float(m['entropy_nats'][g == v, j + 1].mean()) if (g == v).any() else None,
                                            'top5_share_mean': float(m['top5pct_share'][g == v, j + 1].mean()) if (g == v).any() else None}
                             for v in (-1, 0, 1, 2, 3)}
                read = g >= 0
                lesion[t]['spearman_grade_vs_top5_share'] = float(pd.Series(g[read]).corr(pd.Series(m['top5pct_share'][read, j + 1]), method='spearman'))
            entry['concentration_by_site_grade'] = lesion
        result['runs'][f.stem] = entry
    out.write_text(json.dumps(clean(result), ensure_ascii=False, indent=1) + '\n')
    for name, e in result['runs'].items():
        print(name, {q: round(v['right_half_fraction']['mean'], 3) for q, v in e['metrics'].items()})
        if 'expected_direction_checks' in e:
            print('   ', {k: round(v['fraction'], 3) for k, v in e['expected_direction_checks'].items()})


if __name__ == '__main__':
    main()
