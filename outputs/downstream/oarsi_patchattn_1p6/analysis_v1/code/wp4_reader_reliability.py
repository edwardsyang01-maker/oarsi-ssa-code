#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from wp1_primary import BOOT_SEED, N_BOOT, interval, sha256_file
from wp2_secondary import MANIFEST_DIR, clean, read_jsonl

NAS = Path('/path/to/nas/OAI/OAICompleteData_SAS')
VISITS = ('00', '01', '03', '05', '06')
VARIABLES = {'KL': 5, 'JSM': 4, 'JSL': 4, 'OSFM': 4, 'OSFL': 4, 'OSTM': 4, 'OSTL': 4}
OUTPUT_NAMES = ('reader_reliability.json', 'README_wp4.md')
DOC_TABLE2 = {'baseline': ([[73, 22, 5, 0, 0], [20, 28, 25, 0, 0], [2, 11, 62, 5, 0], [0, 1, 8, 36, 1], [0, 0, 0, 0, 1]], 0.70),
              'month36': ([[53, 10, 3, 0, 0], [9, 18, 7, 1, 0], [3, 10, 75, 6, 0], [0, 1, 13, 39, 0], [0, 0, 0, 3, 9]], 0.78)}


def kappa(o, weighting):
    o = np.asarray(o, float)
    k = o.shape[-1]
    d = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    w = {'quadratic': d ** 2 / (k - 1) ** 2, 'linear': d / (k - 1), 'unweighted': (d > 0).astype(float)}[weighting]
    total = o.sum((-1, -2))
    e = o.sum(-1)[..., :, None] * o.sum(-2)[..., None, :] / total[..., None, None]
    denom = (w * e).sum((-1, -2))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom == 0, np.nan, 1 - (w * o).sum((-1, -2)) / denom)


def load(name, visit, hashes):
    path = NAS / f'{name}{visit}.sas7bdat'
    hashes[f'nas:{path.name}'] = sha256_file(path)
    table = pd.read_sas(path, encoding='latin-1')
    table.columns = [re.sub(r'^V\d\dXR', '', c.upper()) for c in table.columns]
    table['ID'] = table['ID'].astype(str).str.strip()
    table['READPRJ'] = table['READPRJ'].astype(str).str.strip()
    return table


def agreement(first, second, participant, k, weights, n_participants):
    conf = np.zeros((n_participants, k * k))
    np.add.at(conf, (participant, first * k + second), 1)
    o = conf.sum(0).reshape(k, k)
    boot = kappa((weights @ conf).reshape(len(weights), k, k), 'quadratic')
    return {'knee_visits': int(len(first)), 'participants': int(len(np.unique(participant))),
            'kappa_quadratic': float(kappa(o, 'quadratic')), 'ci95_kappa_quadratic': interval(boot),
            'undefined_resamples': int(np.isnan(boot).sum()),
            'kappa_linear': float(kappa(o, 'linear')), 'kappa_unweighted': float(kappa(o, 'unweighted')),
            'exact_agreement': float(np.trace(o) / o.sum()) if o.sum() else None,
            'crosstab_rows_first_cols_second': o.astype(int)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='drai project root')
    parser.add_argument('--out', required=True, help='output directory (existing files are not overwritten)')
    args = parser.parse_args()
    root, out = Path(args.root), Path(args.out)
    for name in OUTPUT_NAMES:
        assert not (out / name).exists(), f'output already exists; refusing to overwrite: {name}'
    started = time.strftime('%Y-%m-%d %H:%M:%S %Z')
    hashes = {}

    split_of = {}
    for split in ('train', 'val'):
        for kind in ('usable', 'held'):
            rel = f'{MANIFEST_DIR}/{split}_{kind}.jsonl'
            hashes[rel] = sha256_file(root / rel)
            for row in read_jsonl(root / rel, ('patient_id', 'split')):
                assert row['split'] == split
                pid = str(row['patient_id']).strip()
                assert split_of.setdefault(pid, split) == split, 'the same participant appears in two splits'

    frames, design, same_as_project15 = [], {}, {'compared': 0, 'kl_equal': 0}
    all_ids = set()
    for visit in VISITS:
        table = load('kxr_sq_rel_bu', visit, hashes)
        assert set(table['READPRJ']) == {'19A', '19B'} and set(table['SIDE']) == {1.0, 2.0}
        all_ids |= set(table['ID'])
        design[f'V{visit}'] = {'participants_in_file': int(table['ID'].nunique()), 'rows_in_file': int(len(table))}
        table = table[table['ID'].isin(split_of)]
        first = table[table['READPRJ'] == '19A'].set_index(['ID', 'SIDE'])
        second = table[table['READPRJ'] == '19B'].set_index(['ID', 'SIDE'])
        assert first.index.is_unique and second.index.is_unique and set(first.index) == set(second.index)
        second = second.loc[first.index]
        frame = pd.DataFrame({'visit': f'V{visit}', 'ID': first.index.get_level_values('ID')})
        for name in VARIABLES:
            # Fractional JSN values are floored (codebook section 2.2); osteophytes and KL are already integers.
            frame[f'{name}_1'] = np.floor(first[name].to_numpy(float) + 1e-9)
            frame[f'{name}_2'] = np.floor(second[name].to_numpy(float) + 1e-9)
        frames.append(frame)
        original = load('kxr_sq_bu', visit, hashes)
        original = original[(original['READPRJ'] == '15') & original['ID'].isin(split_of)].set_index(['ID', 'SIDE'])
        common = first.index.intersection(original.index)
        same_as_project15['compared'] += int(len(common))
        same_as_project15['kl_equal'] += int((first.loc[common, 'KL'].to_numpy() == original.loc[common, 'KL'].to_numpy()).sum())
    pairs = pd.concat(frames, ignore_index=True)
    ids = sorted(pairs['ID'].unique())
    lookup = {pid: i for i, pid in enumerate(ids)}
    participant = pairs['ID'].map(lookup).to_numpy()
    n_participants = len(ids)

    # ---------- patient-clustered bootstrap ----------
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, n_participants, size=(N_BOOT, n_participants))
    weights = np.zeros((N_BOOT, n_participants))
    np.add.at(weights, (np.arange(N_BOOT)[:, None], draws), 1)

    def block(mask_rows, label):
        result = {}
        for name, k in VARIABLES.items():
            x, y = pairs[f'{name}_1'].to_numpy(), pairs[f'{name}_2'].to_numpy()
            both = mask_rows & ~np.isnan(x) & ~np.isnan(y)
            entry = agreement(x[both].astype(int), y[both].astype(int), participant[both], k, weights, n_participants)
            entry['read_in_only_one_reading'] = int((mask_rows & (np.isnan(x) != np.isnan(y))).sum())
            entry['not_read_in_either'] = int((mask_rows & np.isnan(x) & np.isnan(y)).sum())
            result[name] = entry
        x, y = pairs['KL_1'].to_numpy(), pairs['KL_2'].to_numpy()
        ok = mask_rows & ~np.isnan(x) & ~np.isnan(y)
        both_low = ok & (x <= 2) & (y <= 2)
        first_low = ok & (x <= 2)
        result['KL_range_0_2'] = {
            'both_readings_kl0_2_three_grades': agreement(x[both_low].astype(int), y[both_low].astype(int), participant[both_low], 3, weights, n_participants),
            'first_reading_kl0_2_five_grades': agreement(x[first_low].astype(int), y[first_low].astype(int), participant[first_low], 5, weights, n_participants),
            'binary_kl_ge2_all_knees': agreement((x[ok] >= 2).astype(int), (y[ok] >= 2).astype(int), participant[ok], 2, weights, n_participants)}
        return result

    every = np.ones(len(pairs), bool)
    results = {'all_visits_pooled': block(every, 'all')}
    for visit in VISITS:
        results[f'V{visit}'] = block((pairs['visit'] == f'V{visit}').to_numpy(), visit)

    included = {split: int(sum(split_of[pid] == split for pid in ids)) for split in ('train', 'val')}
    kl_first = pairs['KL_1'].to_numpy()
    summary = {
        'design': {
            'source': 'OAI kXR_SQ_Rel_BUxx (V00, V01, V03, V05, V06); codebook kXR_SQ_Rel_BU_Descrip.pdf (2016-11-22)',
            'what': 'Project 15 reread: films from all time points of 150 participants were sent for reading again after the original reading; the reading center did not know it was a reread and could not see the original reads; '
                    'both readings used the same workflow (screening, discrepancy identification, adjudication), 3-9 months apart. READPRJ=19A is the original read and 19B the reread.',
            'sample': 'The codebook states the sample is representative of the whole cohort in OA grade, progression and incidence, and number of time points with films; '
                      'but screening-positive participants (KL>=2 in any knee at any time point) make up 123/150 (codebook table 6), and osteophytes were graded only in screening-positive participants.',
            'jsn_fraction': 'fractional JSN values are floored, as in codebook section 2.2 and the training labels',
            'unit': 'knee-visit; the two knees and several visits of one participant are correlated, so intervals use a participant-clustered bootstrap',
            'bootstrap': f'{n_participants} participants resampled with replacement, {n_participants} patients, {N_BOOT} times, numpy.random.default_rng({BOOT_SEED}), '
                         'participants ordered by ascending ID string; percentile interval 2.5%-97.5%',
            'restriction': 'only participants in the train or val lists (usable or deferred) are kept; other participants are not used in any statistic'},
        'limitations': [
            'A reread by the same reading center with the same workflow is not agreement between different hospitals or physicians.',
            'It shows only the noise of the reference standard itself and cannot show that the model reaches or exceeds physician level.',
            'kappa depends on the grade distribution; the KL distribution of the reread sample differs from the validation set, so values cannot be compared point by point with the model validation QWK, only in magnitude.',
            'The reread covers whole-film reads under the original workflow (including adjudication); the model sees cropped single-knee images.',
            'The codebook notes that the second reading graded fewer knees as incidence-eligible (KL0-1); the KL2 threshold shifted between readings.'],
        'sample': {'participants_in_reliability_files': len(all_ids), 'participants_included': n_participants,
                   'included_by_split': included, 'participants_excluded_not_in_train_val': len(all_ids) - n_participants,
                   'per_visit_file': design,
                   'knee_visits_included': int(len(pairs)),
                   'first_reading_kl_counts_pooled': {str(g): int((kl_first == g).sum()) for g in range(5)}},
        'documentation_checks': {
            'first_reading_19A_equals_project15_kl': same_as_project15,
            'doc_table2_kappa_reproduction': {name: {'reported': reported, 'linear': float(kappa(np.array(table), 'linear')),
                                                     'quadratic': float(kappa(np.array(table), 'quadratic')),
                                                     'unweighted': float(kappa(np.array(table), 'unweighted'))}
                                              for name, (table, reported) in DOC_TABLE2.items()}},
        'results': results,
    }
    with open(out / 'reader_reliability.json', 'w', encoding='utf-8') as f:
        json.dump(clean(summary), f, ensure_ascii=False, indent=1)
        f.write('\n')
    code = Path(__file__).resolve()
    lines = ['# WP4 output', '', f'- Start: {started}', f'- End: {time.strftime("%Y-%m-%d %H:%M:%S %Z")}',
             f'- Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}',
             f'- Code: `{code.name}`, SHA256 `{sha256_file(code)}`; depends on `wp1_primary.py` (`{sha256_file(code.parent / "wp1_primary.py")}`)'
             f'and `wp2_secondary.py` (`{sha256_file(code.parent / "wp2_secondary.py")}`)',
             f'- Command: `{" ".join(sys.argv)}`',
             '- Reads only the OAI reading tables on the NAS and the train/val lists, CPU only; participants not in train/val are dropped right after joining; outputs hold aggregate statistics only.', '',
             '## Input file SHA256', '']
    lines += [f'- `{name}` `{digest}`' for name, digest in sorted(hashes.items())]
    (out / 'README_wp4.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    pooled = results['all_visits_pooled']
    print(json.dumps(clean({'included': n_participants, 'by_split': included, 'knee_visits': len(pairs),
                            'kl_qwk': pooled['KL']['kappa_quadratic'], 'ci': pooled['KL']['ci95_kappa_quadratic']}), ensure_ascii=False))


if __name__ == '__main__':
    main()
