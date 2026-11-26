from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path('/path/to/drai')
SAS = Path('/path/to/workspace/test/OAI_audit_20261007/cache/sas_tables')
SOURCE = None
TARGET = None
PROJECT_BY_VISIT = {'V00': '15', 'V01': '15', 'V03': '15', 'V05': '15', 'V06': '15', 'V08': '37', 'V10': '37'}
OSTEOPHYTE = ['OSFM', 'OSFL', 'OSTM', 'OSTL']  # femur medial/lateral, tibia medial/lateral
NARROWING = ['JSM', 'JSL']  # medial, lateral
SPLITS = ['train', 'val']


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def load_visit(visit: str) -> pd.DataFrame:
    table = pd.read_sas(SAS / f'kxr_sq_bu{visit[1:]}.sas7bdat', encoding='latin1')
    table.columns = [re.sub(r'^V\d\dXR', '', c.upper()) for c in table.columns]
    table['ID'] = table['ID'].astype(str).str.strip()
    table['READPRJ'] = table['READPRJ'].astype(str).str.strip().replace({'42': '37'})
    return table


def grade(value, kind: str):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None, None, False
    raw = float(value)
    fractional = abs(raw - round(raw)) > 1e-9
    if kind == 'os' and fractional:
        raise ValueError(f'OSTEOPHYTE_FRACTION:{raw}')
    whole = int(math.floor(raw + 1e-9))
    if not 0 <= whole <= 3:
        raise ValueError(f'GRADE_OUT_OF_RANGE:{raw}')
    return whole, raw, fractional


def build_split(split: str, tables: dict) -> tuple[list[dict], Counter]:
    rows = [json.loads(line) for line in (SOURCE / f'{split}_usable.jsonl').open()]
    index = {v: t.groupby(['ID', 'SIDE', 'READPRJ']).indices for v, t in tables.items()}
    counts = Counter()
    out = []
    for row in rows:
        visit, project = row['visit'], PROJECT_BY_VISIT[row['visit']]
        side = {'R': 1.0, 'L': 2.0}[row['clinical_side']]
        positions = index[visit].get((str(row['patient_id']).strip(), side, project), [])
        if len(positions) != 1:
            raise ValueError(f'ROW_COUNT_{len(positions)}:{row["key"]}')
        reading = tables[visit].iloc[positions[0]]
        if reading['KL'] != row['kl']:
            raise ValueError(f'KL_MISMATCH:{row["key"]}')
        record = {'key': row['key'], 'split': split, 'visit': visit, 'clinical_side': row['clinical_side'],
                  'kl': int(row['kl']), 'readprj': project}
        for name in OSTEOPHYTE + NARROWING:
            kind = 'os' if name in OSTEOPHYTE else 'jsn'
            value, raw, fractional = grade(reading[name], kind)
            record[name.lower()] = value
            record[name.lower() + '_raw'] = raw
            record[name.lower() + '_fraction'] = fractional
            counts[f'{name}_present'] += value is not None
            counts[f'{name}_fraction'] += fractional
        all_os = all(record[n.lower()] is not None for n in OSTEOPHYTE)
        counts['knees'] += 1
        counts[f'kl{row["kl"]}_knees'] += 1
        counts[f'kl{row["kl"]}_all_osteophytes'] += all_os
        counts['all_osteophytes'] += all_os
        counts['both_narrowing'] += all(record[n.lower()] is not None for n in NARROWING)
        out.append(record)
    return out, counts


def atomic_write(path: Path, text: str) -> None:
    temp = path.with_suffix(path.suffix + '.partial')
    with temp.open('x') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temp.rename(path)


def main() -> None:
    global SOURCE, TARGET
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='usable list directory relative to the drai root')
    parser.add_argument('--target', required=True, help='new output directory relative to the drai root')
    args = parser.parse_args()
    assert Path.cwd().resolve() == ROOT, 'RUN_FROM_DRAI_ROOT'
    SOURCE, TARGET = ROOT / args.source, ROOT / args.target
    assert SOURCE.is_dir() and 'manifests' in TARGET.parts and 'test' not in args.source.lower()
    if TARGET.exists():
        raise SystemExit('TARGET_EXISTS')
    tables = {v: load_visit(v) for v in PROJECT_BY_VISIT}
    results = {s: build_split(s, tables) for s in SPLITS}
    TARGET.mkdir(parents=True)
    summary = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
               'source_manifest_dir': str(SOURCE.relative_to(ROOT)),
               'source_manifest_sha256': {f'{s}_usable.jsonl': sha(SOURCE / f'{s}_usable.jsonl') for s in SPLITS},
               'sas_sha256': {f'kxr_sq_bu{v[1:]}.sas7bdat': sha(SAS / f'kxr_sq_bu{v[1:]}.sas7bdat') for v in PROJECT_BY_VISIT},
               'code_sha256': sha(Path(__file__).resolve()),
               'project_by_visit': PROJECT_BY_VISIT, 'readprj_42_recoded_as_37': True, 'project_43_used': False,
               'osteophyte_missing_policy': 'null + mask; never imputed; missingness correlates with KL',
               'narrowing_fraction_policy': 'integer part used as grade (assumed within-grade worsening); raw value kept',
               'fraction_meaning_verified_against_codebook': True,
               'codebook': 'nas main/OAI/OAICompleteData_SAS/kXR_SQ_BU_Descrip.pdf (2016-04-29) p7',
               'kl_equals_manifest_for_every_knee': True, 'test_accessed': False, 'counts': {}}
    for split, (records, counts) in results.items():
        text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
        atomic_write(TARGET / f'{split}_oarsi.jsonl', text)
        summary['counts'][split] = dict(sorted(counts.items()))
        summary[f'{split}_oarsi_sha256'] = sha(TARGET / f'{split}_oarsi.jsonl')
    atomic_write(TARGET / 'summary.json', json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary['counts'], ensure_ascii=False))


if __name__ == '__main__':
    main()
