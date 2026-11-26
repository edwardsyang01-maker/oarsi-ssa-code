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
TEST_MANIFEST = ROOT / 'data/manifests/oarsi_test_eval_v1_20261117'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='frozen source directory (relative to the project root)')
    ap.add_argument('--run', required=True, help='training output directory (relative to the project root)')
    ap.add_argument('--group', required=True, choices=['P0', 'P1', 'P2', 'P3', 'P4'])
    ap.add_argument('--tag', required=True, help='checkpoint tag')
    ap.add_argument('--cohort', required=True, choices=['val', 'test'])
    ap.add_argument('--out', required=True, help='output npz (relative to the project root, not overwritten)')
    a = ap.parse_args()
    out = ROOT / a.out
    assert not out.exists(), f'output already exists; refusing to overwrite: {out}'
    sys.path.insert(0, str(ROOT / a.source))
    import numpy as np
    import torch
    import torch.distributed as dist
    import runtime as R
    import runner as N
    from core import canonical_flip, load_targets

    class TestDataset:
        def __init__(self):
            summary = json.load(open(TEST_MANIFEST / 'summary.json'))
            path = TEST_MANIFEST / 'test_usable.jsonl'
            assert sha(path) == summary['manifest_sha256']['test_usable.jsonl'], 'TEST_MANIFEST_CHANGED'
            self.rows = [json.loads(line) for line in path.open()]
            assert len({r['key'] for r in self.rows}) == len(self.rows)
            assert all(r['split'] == 'test' and r['engineering_input_ready'] for r in self.rows)
            assert all(r['cleanup_status'] in ('accepted_cleaned', 'accepted_unchanged') for r in self.rows)
            assert all(r['source_is_native_pixels'] and not r['resized_from_224'] and r['shape'] == [1, 448, 448] for r in self.rows)
            assert all(r['detector_expansion'] == 1.6 and r['unmasked_pixels_bit_exact'] for r in self.rows)
            assert all(r['clinical_side'] in ('R', 'L') for r in self.rows)
            self.labels = np.asarray([int(r['kl']) for r in self.rows], dtype=np.int64)
            self.is_left = np.asarray([r['clinical_side'] == 'L' for r in self.rows])
            oarsi = TEST_MANIFEST / 'test_oarsi.jsonl'
            self.targets = load_targets(self.rows, oarsi, summary['manifest_sha256']['test_oarsi.jsonl'])

        def __len__(self):
            return len(self.rows)

        def batch(self, ids):
            xs = []
            for i in ids:
                image = np.load(self.rows[int(i)]['gray01_npy'], allow_pickle=False)
                assert image.shape == (1, 448, 448) and image.dtype == np.float32
                assert np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
                xs.append(torch.from_numpy(image))
            x = canonical_flip(torch.stack(xs).to(R.DEVICE), torch.as_tensor(self.is_left[ids]))
            return x, torch.as_tensor(self.labels[ids], device=R.DEVICE), torch.as_tensor(self.targets[ids], device=R.DEVICE)

    started = time.time()
    R.setup()
    engine, _ = R.engine_init(a.group)
    run = ROOT / a.run
    manager = R.Checkpoints(run / 'checkpoints', run / 'recovery_copy', R.binding(a.group))
    progress, _, receipt = manager.load(engine, tag=a.tag)
    dataset = R.Dataset('val') if a.cohort == 'val' else TestDataset()
    rows = N.evaluate(engine, dataset)
    if R.RANK == 0:
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_name(out.stem + '.partial.npz')
        np.savez_compressed(temp, **rows)
        os.replace(temp, out)
        commit = run / 'checkpoints' / f'{a.tag}.commit.json'
        record = {'source': a.source, 'source_binding': R.binding(a.group), 'run': a.run, 'group': a.group, 'tag': a.tag,
                  'commit_receipt_sha256': sha(commit), 'manifest_sha256': receipt['manifest_sha256'],
                  'progress': progress, 'cohort': a.cohort, 'rows': int(len(rows['ids'])),
                  'keys': sorted(rows), 'output': a.out, 'output_sha256': sha(out), 'seconds': time.time() - started,
                  'finished': time.strftime('%F %T %Z')}
        (out.with_suffix('.json')).write_text(json.dumps(record, ensure_ascii=False, indent=1) + '\n')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
