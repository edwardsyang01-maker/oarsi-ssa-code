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
B_REL = 'outputs/downstream/oarsi_patchattn_1p6'
B = ROOT / B_REL
sys.path.insert(0, str(B / 'analysis_v1/code'))
from wp1_primary import MANIFEST_REL, interval, load_manifest, qwk_from_confusion  # noqa: E402
from wp2_secondary import OARSI_DIR, SWAP, TARGETS, clean, oarsi_metrics, read_jsonl, targets_from  # noqa: E402
from wp3_attention import QUERIES, attention_metrics  # noqa: E402

SEEDS = (17, 23, 29)
N_BOOT, BOOT_SEED, C = 10000, 20261115, 5
PLATEAU = tuple(range(11, 21))
TOL = 1e-12
TEST_MANIFEST = ROOT / 'data/manifests/oarsi_test_eval_v1_20261117'
T4 = B / 'test_eval_v1/T4/test'
RUNS = {
    (17, 'P0'): 'attempt01/train_P0', (17, 'P2'): 'attempt02_lambda_v2/train_P2',
    (17, 'P3'): 'attempt02_lambda_v2/train_P3', (17, 'P4'): 'attempt02_lambda_v2/train_P4',
    (23, 'P0'): 'attempt03_seed23/train_P0', (23, 'P2'): 'attempt04_x1_seed23/train_P2',
    (23, 'P3'): 'attempt03_seed23/train_P3', (23, 'P4'): 'attempt05_x3_seed23/train_P4',
    (29, 'P0'): 'attempt03_seed29/train_P0', (29, 'P2'): 'attempt04_x1_seed29/train_P2',
    (29, 'P3'): 'attempt03_seed29/train_P3', (29, 'P4'): 'attempt05_x3_seed29/train_P4',
}
PAIRS_LR = (('osfm', 'osfl'), ('ostm', 'ostl'), ('jsm', 'jsl'))
PAIRS_FT = (('osfm', 'ostm'), ('osfl', 'ostl'))
HASHES = {}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    digest = h.hexdigest()
    try:
        HASHES[str(Path(path).relative_to(ROOT))] = digest
    except ValueError:
        HASHES[str(path)] = digest
    return digest


def load(path):
    sha(path)
    return np.load(path)


def boot_weights(n_patients):
    draws = np.random.default_rng(BOOT_SEED).integers(0, n_patients, size=(N_BOOT, n_patients))
    w = np.zeros((N_BOOT, n_patients))
    np.add.at(w, (np.arange(N_BOOT)[:, None], draws), 1)
    return w


def boot_p(d):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    r = len(d)
    le, ge = int((d <= 0).sum()), int((d >= 0).sum())
    p = min(1.0, 2 * min((le + 1) / (r + 1), (ge + 1) / (r + 1)))
    return {'p_two_sided': p, 'resamples': r, 'count_le_0': le, 'count_ge_0': ge}


def holm(pvals):
    names = sorted(pvals, key=pvals.get)
    m, out, running = len(names), {}, 0.0
    for i, n in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvals[n]))
        out[n] = running
    return out


def qwk_c(o, c):
    o = np.asarray(o, float)
    w = np.subtract.outer(np.arange(c), np.arange(c)) ** 2 / (c - 1) ** 2
    total = o.sum((-1, -2))
    e = o.sum(-1)[..., :, None] * o.sum(-2)[..., None, :] / total[..., None, None]
    denom = (w * e).sum((-1, -2))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom == 0, np.nan, 1 - (w * o).sum((-1, -2)) / denom)


def patient_conf(pidx, n_patients, y, pred, c, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    out = np.zeros((n_patients, c * c))
    np.add.at(out, (pidx[m], y[m] * c + pred[m]), 1)
    return out


def best_epoch(key):
    return int(json.load(open(B / RUNS[key] / 'completed.json'))['best_epoch'])


def summarize_diff(per_seed_point, per_seed_boot):
    dist = np.mean([per_seed_boot[s] for s in per_seed_point], axis=0)
    return {'mean_diff': float(np.mean(list(per_seed_point.values()))),
            'per_seed_diff': {str(s): float(v) for s, v in per_seed_point.items()},
            'ci95': interval(dist), **boot_p(dist),
            'per_seed': {str(s): {'ci95': interval(per_seed_boot[s]), **boot_p(per_seed_boot[s])} for s in per_seed_point}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='output directory (must be empty or absent)')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f'output directory is not empty; refusing to overwrite: {out}'
    started = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    checks = {}

    sha(ROOT / MANIFEST_REL)
    sha(ROOT / f'{OARSI_DIR}/val_oarsi.jsonl')
    vpidx, vy, vP = load_manifest(ROOT / MANIFEST_REL)
    vrows = read_jsonl(ROOT / MANIFEST_REL, ('key', 'patient_id', 'split', 'visit', 'clinical_side', 'kl'))
    vlabels = read_jsonl(ROOT / f'{OARSI_DIR}/val_oarsi.jsonl', ('key', 'kl', 'clinical_side', 'visit', *TARGETS))
    vtargets = targets_from(vrows, vlabels)
    vw = boot_weights(vP)
    vkw = vw[:, vpidx]

    q_sel_pt, q_plat_pt, q_sel_bt, q_plat_bt = {}, {}, {}, {}
    for s in SEEDS:
        for g in ('P0', 'P3'):
            key = (s, g)
            be = best_epoch(key)
            sel = load(B / RUNS[key] / f'val_epoch{be:02d}.npz')
            assert np.array_equal(sel['labels'].astype(int), vy)
            cbp = patient_conf(vpidx, vP, vy, sel['probabilities'].argmax(1), C)
            q_sel_pt[key] = float(qwk_from_confusion(cbp.sum(0).reshape(C, C)))
            q_sel_bt[key] = qwk_from_confusion((vw @ cbp).reshape(N_BOOT, C, C))
            plat_pt, plat_bt = [], []
            for e in PLATEAU:
                z = load(B / RUNS[key] / f'val_epoch{e:02d}.npz')
                c = patient_conf(vpidx, vP, vy, z['probabilities'].argmax(1), C)
                plat_pt.append(float(qwk_from_confusion(c.sum(0).reshape(C, C))))
                plat_bt.append(qwk_from_confusion((vw @ c).reshape(N_BOOT, C, C)))
            q_plat_pt[key] = float(np.mean(plat_pt))
            q_plat_bt[key] = np.mean(plat_bt, axis=0)
    kl_val = {
        'selected': summarize_diff({s: q_sel_pt[(s, 'P3')] - q_sel_pt[(s, 'P0')] for s in SEEDS},
                                   {s: q_sel_bt[(s, 'P3')] - q_sel_bt[(s, 'P0')] for s in SEEDS}),
        'plateau_epochs_11_20': summarize_diff({s: q_plat_pt[(s, 'P3')] - q_plat_pt[(s, 'P0')] for s in SEEDS},
                                               {s: q_plat_bt[(s, 'P3')] - q_plat_bt[(s, 'P0')] for s in SEEDS}),
    }
    wp1 = json.load(open(B / 'analysis_v1/outputs/primary_p3_vs_p0.json'))
    d = max(abs(kl_val['selected']['mean_diff'] - wp1['mean_diff']['q_sel']),
            abs(kl_val['plateau_epochs_11_20']['mean_diff'] - wp1['mean_diff']['q_plat']),
            *[abs(a - b) for a, b in zip(kl_val['selected']['ci95'], wp1['ci95_mean_diff']['q_sel'])],
            *[abs(a - b) for a, b in zip(kl_val['plateau_epochs_11_20']['ci95'], wp1['ci95_mean_diff']['q_plat'])])
    checks['kl_val_vs_wp1'] = {'max_abs_diff': d, 'pass': d <= TOL}

    sha(TEST_MANIFEST / 'test_usable.jsonl')
    sha(TEST_MANIFEST / 'test_oarsi.jsonl')
    trows = read_jsonl(TEST_MANIFEST / 'test_usable.jsonl', ('key', 'patient_id', 'split', 'visit', 'clinical_side', 'kl', 'cleanup_status'))
    tlabels = read_jsonl(TEST_MANIFEST / 'test_oarsi.jsonl', ('key', 'kl', 'clinical_side', 'visit', *TARGETS))
    ttargets = targets_from(trows, tlabels)
    ty = np.array([r['kl'] for r in trows])
    tnames = sorted({r['patient_id'] for r in trows})
    tlook = {p: i for i, p in enumerate(tnames)}
    tpidx = np.array([tlook[r['patient_id']] for r in trows])
    tP = len(tnames)
    tw = boot_weights(tP)
    tpred = {}

    def test_model(name):
        if name not in tpred:
            z = load(T4 / f'{name}.npz')
            assert np.array_equal(z['labels'], ty)
            tpred[name] = {k: z[k] for k in z.files}
        return tpred[name]

    kl_test = {}
    for role in ('primary_selected', 'sensitivity_epoch20_end'):
        pt, bt = {}, {}
        for s in SEEDS:
            q = {}
            for g in ('P0', 'P3'):
                cbp = patient_conf(tpidx, tP, ty, test_model(f'{role}_seed{s}_{g}')['probabilities'].argmax(1), C)
                q[g] = (float(qwk_from_confusion(cbp.sum(0).reshape(C, C))), qwk_from_confusion((tw @ cbp).reshape(N_BOOT, C, C)))
            pt[s], bt[s] = q['P3'][0] - q['P0'][0], q['P3'][1] - q['P0'][1]
        kl_test[role] = summarize_diff(pt, bt)
    t5 = json.load(open(B / 'test_eval_v1/T5/t5_results.json'))
    ref = {'primary_selected': t5['primary'], 'sensitivity_epoch20_end': t5['sensitivity_P3_minus_P0_epoch20_end']['qwk']}
    d = max(max(abs(kl_test[r]['mean_diff'] - ref[r]['mean']), *[abs(a - b) for a, b in zip(kl_test[r]['ci95'], ref[r]['ci95'])]) for r in ref)
    checks['kl_test_vs_t5'] = {'max_abs_diff': d, 'pass': d <= TOL}

    wp8 = json.load(open(B / 'wp8_analysis/outputs/oarsi_heads_and_laterality_three_seeds.json'))
    oarsi_pt, oarsi_bt, sel_prob = {}, {}, {}
    for s in SEEDS:
        for g in ('P2', 'P3', 'P4'):
            key = (s, g)
            be = best_epoch(key)
            z = load(B / RUNS[key] / f'val_epoch{be:02d}.npz')
            prob = z['oarsi_probabilities']
            sel_prob[key] = {'oarsi': prob}
            if g in ('P3', 'P4'):
                sel_prob[key]['flip'] = load(B / RUNS[key] / f'val_extra_flip_epoch{be:02d}.npz')['oarsi_probabilities']
            pred = prob.argmax(-1)
            for j, t in enumerate(TARGETS):
                m = vtargets[:, j] >= 0
                cbp = patient_conf(vpidx, vP, vtargets[:, j], pred[:, j], 4, m)
                oarsi_pt[(s, g, t)] = float(qwk_c(cbp.sum(0).reshape(4, 4), 4))
                oarsi_bt[(s, g, t)] = qwk_c((vw @ cbp).reshape(N_BOOT, 4, 4), 4)
    d = max(abs(oarsi_pt[(s, g, t)] - wp8['oarsi_heads'][g]['per_seed'][str(s)][t]['qwk'])
            for s in SEEDS for g in ('P2', 'P3', 'P4') for t in TARGETS)
    checks['oarsi_val_points_vs_wp8'] = {'max_abs_diff': d, 'pass': d <= TOL}
    oarsi_val = {}
    for t in TARGETS:
        oarsi_val[t] = {'n_knees': int((vtargets[:, TARGETS.index(t)] >= 0).sum()),
                        'P2_mean': float(np.mean([oarsi_pt[(s, 'P2', t)] for s in SEEDS])),
                        'P3_mean': float(np.mean([oarsi_pt[(s, 'P3', t)] for s in SEEDS])),
                        'P3_minus_P2': summarize_diff({s: oarsi_pt[(s, 'P3', t)] - oarsi_pt[(s, 'P2', t)] for s in SEEDS},
                                                      {s: oarsi_bt[(s, 'P3', t)] - oarsi_bt[(s, 'P2', t)] for s in SEEDS})}
    adj6 = holm({t: oarsi_val[t]['P3_minus_P2']['p_two_sided'] for t in TARGETS})
    adj4 = holm({t: oarsi_val[t]['P3_minus_P2']['p_two_sided'] for t in TARGETS[:4]})
    for t in TARGETS:
        oarsi_val[t]['P3_minus_P2']['p_holm_6_heads'] = adj6[t]
        if t in adj4:
            oarsi_val[t]['P3_minus_P2']['p_holm_4_osteophyte_heads'] = adj4[t]

    oarsi_test = {}
    p3t, p2t = test_model('primary_selected_seed17_P3'), test_model('secondary_selected_seed17_P2')
    for j, t in enumerate(TARGETS):
        m = ttargets[:, j] >= 0
        q = {}
        for g, z in (('P2', p2t), ('P3', p3t)):
            cbp = patient_conf(tpidx, tP, ttargets[:, j], z['oarsi_probabilities'].argmax(-1)[:, j], 4, m)
            q[g] = (float(qwk_c(cbp.sum(0).reshape(4, 4), 4)), qwk_c((tw @ cbp).reshape(N_BOOT, 4, 4), 4))
        dist = q['P3'][1] - q['P2'][1]
        oarsi_test[t] = {'n_knees': int(m.sum()), 'P2': q['P2'][0], 'P3': q['P3'][0], 'diff': q['P3'][0] - q['P2'][0],
                         'ci95': interval(dist), **boot_p(dist)}
    adj6t = holm({t: oarsi_test[t]['p_two_sided'] for t in TARGETS})
    for t in TARGETS:
        oarsi_test[t]['p_holm_6_heads'] = adj6t[t]
    d = max(max(abs(oarsi_test[t]['P3'] - t5['per_model']['primary_selected_seed17_P3']['oarsi'][t]['qwk']),
                abs(oarsi_test[t]['P2'] - t5['per_model']['secondary_selected_seed17_P2']['oarsi'][t]['qwk'])) for t in TARGETS)
    checks['oarsi_test_points_vs_t5'] = {'max_abs_diff': d, 'pass': d <= TOL}

    x5 = json.load(open(B / 'wp8_x5/x5_attention_summary.json'))
    ix = {n: i for i, n in enumerate(QUERIES)}
    attention, d = {}, 0.0
    for s in SEEDS:
        z = load(B / f'wp8_x5/attention/seed{s}_P3.npz')
        assert np.array_equal(z['labels'], vy)
        mets = attention_metrics(z['attention'])
        right, row = mets['right_half_fraction'], mets['vertical_centroid_row']
        tests = {f'{a}_more_right_than_{b}': right[:, ix[a]] > right[:, ix[b]] for a, b in PAIRS_LR}
        tests.update({f'{a}_higher_than_{b}': row[:, ix[a]] < row[:, ix[b]] for a, b in PAIRS_FT})
        tests['all_five_checks'] = np.all(np.stack(list(tests.values()), 1), 1)
        res = {}
        for name, ok in tests.items():
            num = np.zeros(vP); np.add.at(num, vpidx, ok.astype(float))
            den = np.bincount(vpidx, minlength=vP).astype(float)
            dist = (vw @ num) / (vw @ den)
            fails_pat = int((num < den).sum())
            entry = {'fraction': float(ok.mean()), 'knees_failing': int((~ok).sum()), 'patients_with_any_failing_knee': fails_pat,
                     'ci95_cluster_bootstrap': interval(dist)}
            if fails_pat == 0:
                entry['patient_level_clopper_pearson_lower_95'] = float(0.025 ** (1 / vP))
            res[name] = entry
            if name in x5['runs'][f'seed{s}_P3']['expected_direction_checks']:
                d = max(d, abs(entry['fraction'] - x5['runs'][f'seed{s}_P3']['expected_direction_checks'][name]['fraction']))
        attention[str(s)] = res
    checks['attention_vs_x5'] = {'max_abs_diff': d, 'pass': d <= TOL}

    js_unequal = (vtargets[:, 4] >= 0) & (vtargets[:, 5] >= 0) & (vtargets[:, 4] != vtargets[:, 5])
    subsets = {'all_knees_all_heads': (np.ones(len(vy), bool), [0, 1, 2, 3, 4, 5]),
               'narrowing_jsm_ne_jsl': (js_unequal, [4, 5]),
               'narrowing_jsl_gt_jsm': ((vtargets[:, 4] >= 0) & (vtargets[:, 5] > vtargets[:, 4]), [4, 5])}
    lat, d = {}, 0.0
    for sub, (mask, heads) in subsets.items():
        frac_pt, frac_bt = {}, {}
        for g in ('P3', 'P4'):
            for s in SEEDS:
                a = sel_prob[(s, g)]['oarsi'].argmax(-1)[:, heads]
                b = sel_prob[(s, g)]['flip'].argmax(-1)[:, heads]
                same = (a == b).sum(1).astype(float) * mask
                num = np.zeros(vP); np.add.at(num, vpidx, same)
                den = np.zeros(vP); np.add.at(den, vpidx, mask * float(len(heads)))
                frac_pt[(s, g)] = float(num.sum() / den.sum())
                frac_bt[(s, g)] = (vw @ num) / (vw @ den)
                d = max(d, abs(frac_pt[(s, g)] - wp8['laterality_per_run'][f'seed{s}_{g}']['subsets'][sub]['same']))
        lat[sub] = {'knees': int(mask.sum()),
                    'same_fraction': {g: {str(s): {'value': frac_pt[(s, g)], 'ci95': interval(frac_bt[(s, g)])} for s in SEEDS}
                                      for g in ('P3', 'P4')},
                    'P4_minus_P3': summarize_diff({s: frac_pt[(s, 'P4')] - frac_pt[(s, 'P3')] for s in SEEDS},
                                                  {s: frac_bt[(s, 'P4')] - frac_bt[(s, 'P3')] for s in SEEDS})}
    checks['laterality_vs_wp8'] = {'max_abs_diff': d, 'pass': d <= TOL}

    checks_pass = all(c['pass'] for c in checks.values())
    result = {
        'schema': 'drai_posthoc_pvalues_v1',
        'note': 'Post hoc descriptive P values and proportion intervals; reads existing predictions only, CPU only, no new inference/training. The prespecified A-D grading is unchanged.'
                'P values and intervals reflect participant sampling only, not training randomness or epoch selection; no multiplicity correction except where Holm is stated.',
        'bootstrap': f'whole-patient clusters, default_rng({BOOT_SEED}).integers(0, P, size=({N_BOOT}, P)), patients ordered by ascending ID string; '
                     'same as WP1 (validation, P=435) and T5 (test, P=448); per-seed differences first, then the mean; percentile interval 2.5%-97.5%',
        'p_value_definition': 'p = min(1, 2·min((#Δ*≤0 + 1)/(R+1), (#Δ*≥0 + 1)/(R+1)))',
        'selfcheck': {'pass': checks_pass, **checks},
        'A_kl_validation_P3_minus_P0': kl_val,
        'A_kl_test_P3_minus_P0': kl_test,
        'B_oarsi_validation_P3_minus_P2_three_seeds': oarsi_val,
        'B_oarsi_test_seed17_P3_minus_P2': oarsi_test,
        'C_attention_order_P3_validation': attention,
        'D_laterality_flip_validation': lat,
        'started': started, 'finished': time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'input_sha256': HASHES,
    }
    (out / 'pvalues_v1.json').write_text(json.dumps(clean(result), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    code = Path(__file__).resolve()
    lines = ['# Post hoc P values and proportion intervals (pvalues_v1)', '', f'- Start: {started}', f'- End: {result["finished"]}',
             f'- Python {platform.python_version()}, numpy {np.__version__}',
             f'- Code: `{code.name}`, SHA256 `{sha(code)}`', f'- Command: `{" ".join(sys.argv)}`',
             f'- Self-check: {"passed" if checks_pass else "failed"}', '- Reads existing predictions only, CPU only; outputs hold aggregate statistics only.', '',
             '## Input file SHA256', '']
    lines += [f'- `{k}` `{v}`' for k, v in sorted(HASHES.items()) if not k.endswith(code.name)]
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(clean({'selfcheck': result['selfcheck']}), ensure_ascii=False))
    if not checks_pass:
        sys.exit(2)


if __name__ == '__main__':
    main()
