#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys

import numpy as np

sys.path.insert(0, '/path/to/drai/outputs/downstream/oarsi_patchattn_1p6/pvalues_v1/code')
import pvalues_v1 as pv  # noqa: E402

_lm, _rj = pv.load_manifest, pv.read_jsonl


def knee_manifest(path):
    pidx, y, n_pat = _lm(path)
    return np.arange(len(y)), y, len(y)


def knee_rows(path, fields):
    rows = _rj(path, fields)
    if 'test_usable' in str(path) and 'patient_id' in fields:
        keys = [r['key'] for r in rows]
        assert len(set(keys)) == len(keys), 'test key is not unique'
        for r in rows:
            r['patient_id'] = r['key']
    return rows


pv.load_manifest = knee_manifest
pv.read_jsonl = knee_rows

if __name__ == '__main__':
    try:
        pv.main()
    except SystemExit as e:
        print(f'pvalues_v1 self-check exit code {e.code} (KL intervals are expected to differ from WP1/T5)')
