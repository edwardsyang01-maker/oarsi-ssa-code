#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/path/to/drai')
B = 'outputs/downstream/oarsi_patchattn_1p6'
X = ROOT / B / 'wp8_x5'
PLAN = ROOT / 'outputs/downstream/oa_kl_vitl_l5_reviewed_v3_training_v1/seed17/source_v1/plan.json'
MODELS = [('seed17_P1', 'P1', 'source_v2', 'attempt02_lambda_v2/train_P1'),
          ('seed17_P3', 'P3', 'source_v2', 'attempt02_lambda_v2/train_P3'),
          ('seed23_P3', 'P3', 'source_v3_seed23', 'attempt03_seed23/train_P3'),
          ('seed29_P3', 'P3', 'source_v3_seed29', 'attempt03_seed29/train_P3'),
          ('seed17_P4', 'P4', 'source_v2', 'attempt02_lambda_v2/train_P4')]


def environment():
    plan = json.load(open(PLAN))
    env = dict(os.environ)
    env.update({'CUDA_VISIBLE_DEVICES': ','.join(plan['gpu_uuids']), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                'NCCL_P2P_DISABLE': '1', 'NCCL_DEBUG': 'WARN', 'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
                'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    os.sched_setaffinity(0, set(plan['cpu_affinity_logical_ids']))
    return env


def main():
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    env = environment()
    checks = []
    for name, group, source, run in MODELS:
        if only and name not in only:
            continue
        done = json.load(open(ROOT / B / run / 'completed.json'))
        out = X / 'attention' / f'{name}.npz'
        if not out.exists():
            pids = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, check=True).stdout.split()
            assert not pids, f'other compute processes on the GPU: {pids}'
            command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                       str(X / 'code/x5_attention.py'), '--source', f'{B}/{source}', '--train-dir', f'{B}/{run}',
                       '--group', group, '--tag', done['best_checkpoint'], '--out', str(out.relative_to(ROOT))]
            with open(X / 'x5_infer.log', 'a') as log:
                log.write(f'\n== {name} {time.strftime("%F %T")}\n')
                log.flush()
                code = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            assert code == 0 and out.exists(), f'inference failed: {name} code={code}'
        z = np.load(out)
        ref = np.load(ROOT / B / run / f"val_epoch{done['best_epoch']:02d}.npz")
        panel = np.load(ROOT / B / run / f"attention_panel_epoch{done['best_epoch']:02d}.npz")
        att = z['attention'][panel['ids']]
        check = {'name': name, 'epoch': done['best_epoch'],
                 'prob_bitwise_equal': bool(np.array_equal(z['probabilities'], ref['probabilities'])),
                 'prob_max_abs_diff': float(np.abs(z['probabilities'] - ref['probabilities']).max()),
                 'panel_attention_max_abs_diff': float(np.abs(att - panel['attention']).max()),
                 'attention_shape': list(z['attention'].shape)}
        checks.append(check)
        print(json.dumps(check, ensure_ascii=False), flush=True)
    if not only:
        (X / 'x5_infer_checks.json').write_text(json.dumps(checks, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
