#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

B = Path('/path/to/drai/outputs/downstream/oarsi_patchattn_1p6')
sys.path.insert(0, str(B / 'pvalues_v1/code'))
import pvalues_v1 as pv  # noqa: E402

SEEDS = pv.SEEDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f'output directory is not empty; refusing to overwrite: {out}'
    started = time.strftime('%Y-%m-%d %H:%M:%S %Z')

    pv.sha(pv.ROOT / pv.MANIFEST_REL)
    pv.sha(pv.ROOT / f'{pv.OARSI_DIR}/val_oarsi.jsonl')
    vpidx, vy, vP = pv.load_manifest(pv.ROOT / pv.MANIFEST_REL)
    vrows = pv.read_jsonl(pv.ROOT / pv.MANIFEST_REL, ('key', 'patient_id', 'split', 'visit', 'clinical_side', 'kl'))
    vlabels = pv.read_jsonl(pv.ROOT / f'{pv.OARSI_DIR}/val_oarsi.jsonl', ('key', 'kl', 'clinical_side', 'visit', *pv.TARGETS))
    t = pv.targets_from(vrows, vlabels)
    vw = pv.boot_weights(vP)
    n = len(vy)

    sel = {}
    for s in SEEDS:
        for g in ('P3', 'P4'):
            be = pv.best_epoch((s, g))
            z = pv.load(B / pv.RUNS[(s, g)] / f'val_epoch{be:02d}.npz')
            assert np.array_equal(z['labels'].astype(int), vy)
            f = pv.load(B / pv.RUNS[(s, g)] / f'val_extra_flip_epoch{be:02d}.npz')
            sel[(s, g)] = (z['oarsi_probabilities'].argmax(-1), f['oarsi_probabilities'].argmax(-1))

    js_unequal = (t[:, 4] >= 0) & (t[:, 5] >= 0) & (t[:, 4] != t[:, 5])
    femoral = (t[:, 0] >= 0) & (t[:, 1] >= 0) & (t[:, 0] != t[:, 1])
    tibial = (t[:, 2] >= 0) & (t[:, 3] >= 0) & (t[:, 2] != t[:, 3])
    subsets = {
        'all_knees_all_heads': (np.ones(n, bool), [0, 1, 2, 3, 4, 5]),
        'all_knees_narrowing_heads': (np.ones(n, bool), [4, 5]),
        'all_knees_osteophyte_heads': (np.ones(n, bool), [0, 1, 2, 3]),
        'narrowing_jsm_ne_jsl': (js_unequal, [4, 5]),
        'osteophyte_osfm_ne_osfl_or_ostm_ne_ostl': (femoral | tibial, [0, 1, 2, 3]),
        'narrowing_jsl_gt_jsm': ((t[:, 4] >= 0) & (t[:, 5] > t[:, 4]), [4, 5]),
    }
    wp8 = json.load(open(B / 'wp8_analysis/outputs/oarsi_heads_and_laterality_three_seeds.json'))
    old = json.load(open(B / 'pvalues_v1/outputs/pvalues_v1.json'))['D_laterality_flip_validation']
    res, d_wp8, d_old = {}, 0.0, 0.0
    for sub, (mask, heads) in subsets.items():
        pt, bt = {}, {}
        for g in ('P3', 'P4'):
            for s in SEEDS:
                a, b = sel[(s, g)][0][:, heads], sel[(s, g)][1][:, heads]
                same = (a == b).sum(1).astype(float) * mask
                num = np.zeros(vP); np.add.at(num, vpidx, same)
                den = np.zeros(vP); np.add.at(den, vpidx, mask * float(len(heads)))
                pt[(s, g)] = float(num.sum() / den.sum())
                bt[(s, g)] = (vw @ num) / (vw @ den)
                d_wp8 = max(d_wp8, abs(pt[(s, g)] - wp8['laterality_per_run'][f'seed{s}_{g}']['subsets'][sub]['same']))
        diff = pv.summarize_diff({s: pt[(s, 'P4')] - pt[(s, 'P3')] for s in SEEDS},
                                 {s: bt[(s, 'P4')] - bt[(s, 'P3')] for s in SEEDS})
        res[sub] = {'knees': int(mask.sum()), 'heads': heads,
                    'same_fraction_three_seed_mean': {g: float(np.mean([pt[(s, g)] for s in SEEDS])) for g in ('P3', 'P4')},
                    'P4_minus_P3': diff}
        if sub in old:
            o = old[sub]['P4_minus_P3']
            d_old = max(d_old, abs(o['mean_diff'] - diff['mean_diff']), abs(o['p_two_sided'] - diff['p_two_sided']),
                        *[abs(x - y) for x, y in zip(o['ci95'], diff['ci95'])])
    holm = pv.holm({k: v['P4_minus_P3']['p_two_sided'] for k, v in res.items()})
    for k in res:
        res[k]['P4_minus_P3']['p_holm_6_subsets'] = holm[k]
    checks = {'same_vs_wp8': {'max_abs_diff': d_wp8, 'pass': d_wp8 <= pv.TOL},
              'three_subsets_vs_pvalues_v1': {'max_abs_diff': d_old, 'pass': d_old <= pv.TOL}}
    ok = all(c['pass'] for c in checks.values())
    result = {'schema': 'drai_posthoc_flip_pvalues_v2',
              'note': 'Post hoc and descriptive; reuses the resampling and P-value definition of pvalues_v1; reflects participant sampling only, not training randomness or epoch selection.',
              'p_value_definition': 'p = min(1, 2*min((#d*<=0 + 1)/(R+1), (#d*>=0 + 1)/(R+1))); R = 10,000, smallest reportable about 0.0002',
              'selfcheck': {'pass': ok, **checks}, 'D_laterality_flip_validation_all_six': res,
              'started': started, 'finished': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
              'code_sha256': pv.sha(Path(__file__).resolve()), 'input_sha256': pv.HASHES}
    (out / 'pvalues_flip_v2.json').write_text(json.dumps(pv.clean(result), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    for k, v in res.items():
        dd = v['P4_minus_P3']
        print(f"{k:42s} n={v['knees']:5d} P3={v['same_fraction_three_seed_mean']['P3']:.4f} "
              f"P4={v['same_fraction_three_seed_mean']['P4']:.4f} diff={dd['mean_diff']:+.4f} "
              f"CI=({dd['ci95'][0]:+.4f},{dd['ci95'][1]:+.4f}) p={dd['p_two_sided']:.4f} holm={dd['p_holm_6_subsets']:.4f}")
    print('selfcheck', json.dumps(checks))
    if not ok:
        sys.exit(2)


if __name__ == '__main__':
    main()
