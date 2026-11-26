#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/path/to/drai')
CHUNK = 512


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--train-dir', required=True)
    ap.add_argument('--group', required=True, choices=['P1', 'P3', 'P4'])
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = ROOT / a.out
    assert not out.exists(), f'refusing to overwrite: {out}'
    sys.path.insert(0, str(ROOT / a.source))
    import numpy as np
    import torch.distributed as dist
    import runtime as R
    import runner as N
    started = time.time()
    R.setup()
    engine, _ = R.engine_init(a.group)
    run = ROOT / a.train_dir
    manager = R.Checkpoints(run / 'checkpoints', run / 'recovery_copy', R.binding(a.group))
    progress, _, receipt = manager.load(engine, tag=a.tag)
    val = R.Dataset('val')
    parts = []
    for start in range(0, len(val), CHUNK):
        ids = list(range(start, min(start + CHUNK, len(val))))
        parts.append(N.evaluate(engine, val, subset=ids, attention=True))
    if R.RANK == 0:
        rows = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        assert np.array_equal(rows['ids'], np.arange(len(val)))
        rows['attention'] = rows['attention'].astype(np.float32)
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_name(out.stem + '.partial.npz')
        np.savez_compressed(temp, **rows)
        os.replace(temp, out)
        record = {'source': a.source, 'train_dir': a.train_dir, 'group': a.group, 'tag': a.tag,
                  'manifest_sha256': receipt['manifest_sha256'], 'progress': progress, 'cohort': 'val',
                  'rows': int(len(rows['ids'])), 'attention_shape': list(rows['attention'].shape), 'chunk': CHUNK,
                  'output_sha256': sha(out), 'seconds': time.time() - started, 'finished': time.strftime('%F %T %Z')}
        out.with_suffix('.json').write_text(json.dumps(record, ensure_ascii=False, indent=1) + '\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
