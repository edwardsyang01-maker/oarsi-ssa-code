#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/path/to/drai')
T = ROOT / 'outputs/downstream/oarsi_patchattn_1p6/test_eval_v1'
MATERIAL_SHA256 = 'd7385610c4a8dcbfd993a66443439ccb03243c64554fb51c28f6b33541f6561c'
PLAN = ROOT / 'outputs/downstream/oa_kl_vitl_l5_reviewed_v3_training_v1/seed17/source_v1/plan.json'
TOL = 1e-6


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def models():
    assert sha(T / 'T3/t3_lock_material.json') == MATERIAL_SHA256, 'locked materials changed'
    auth = json.load(open(T / 'T3/authorization_T3.json'))
    assert auth['approved_file_sha256'] == MATERIAL_SHA256
    m = json.load(open(T / 'T3/t3_lock_material.json'))
    out = []
    for x in m['models']:
        out.append({**x, 'name': f"{x['role']}_seed{x['seed']}_{x['group']}"})
    assert len(out) == 15 and len({x['name'] for x in out}) == 15
    return out


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


def gpu_free():
    pids = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True).stdout.split()
    assert not pids, f'other compute processes on the GPU: {pids}'


def infer(model, cohort, out, env, log):
    command = [sys.executable, '-B', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
               str(T / 'inference_v1/infer.py'), '--source', model['source'], '--run', model['run'],
               '--group', model['group'], '--tag', model['tag'], '--cohort', cohort,
               '--out', str(out.relative_to(ROOT))]
    with open(log, 'a') as stream:
        stream.write(f"\n== {model['name']} {cohort} {time.strftime('%F %T')}\n")
        stream.flush()
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
    assert code == 0 and out.exists(), f"inference failed: {model['name']} {cohort} code={code}"


def compare(model, out):
    ref = np.load(ROOT / model['run'] / f"val_epoch{model['validation_epoch']:02d}.npz")
    new = np.load(out)
    result = {'name': model['name'], 'epoch': model['validation_epoch'],
              'ids_equal': bool(np.array_equal(ref['ids'], new['ids'])),
              'labels_equal': bool(np.array_equal(ref['labels'], new['labels'])),
              'keys_equal': sorted(ref.files) == sorted(new.files)}
    d = np.abs(ref['probabilities'] - new['probabilities'])
    result.update(prob_max_abs_diff=float(d.max()), prob_bitwise_equal=bool(np.array_equal(ref['probabilities'], new['probabilities'])),
                  argmax_equal=bool(np.array_equal(ref['probabilities'].argmax(1), new['probabilities'].argmax(1))))
    if 'oarsi_probabilities' in ref.files:
        o = np.abs(ref['oarsi_probabilities'] - new['oarsi_probabilities'])
        result.update(oarsi_max_abs_diff=float(o.max()),
                      oarsi_argmax_equal=bool(np.array_equal(ref['oarsi_probabilities'].argmax(-1), new['oarsi_probabilities'].argmax(-1))))
    result['pass'] = bool(result['ids_equal'] and result['labels_equal'] and result['keys_equal']
                          and result['prob_max_abs_diff'] <= TOL and result['argmax_equal'])
    return result


def replay(only):
    env = environment()
    folder = T / 'T4/val_replay'
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for m in models():
        if only and m['name'] not in only.split(','):
            continue
        out = folder / f"{m['name']}.npz"
        if not out.exists():
            gpu_free()
            infer(m, 'val', out, env, T / 'T4/val_replay.log')
        r = compare(m, out)
        results.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        if not r['pass']:
            break
    if not only:
        record = {'results': results, 'all_pass': len(results) == 15 and all(r['pass'] for r in results),
                  'tolerance': TOL, 'material_sha256': MATERIAL_SHA256, 'finished': time.strftime('%F %T %Z')}
        target = T / 'T4/replay_check.json'
        assert not target.exists(), 'replay_check.json already exists'
        target.write_text(json.dumps(record, ensure_ascii=False, indent=1) + '\n')
        print(json.dumps({'all_pass': record['all_pass'], 'n': len(results)}), flush=True)


def test():
    check = json.load(open(T / 'T4/replay_check.json'))
    assert check['all_pass'] is True and check['material_sha256'] == MATERIAL_SHA256, 'validation replay did not fully pass; no test inference'
    env = environment()
    folder = T / 'T4/test'
    folder.mkdir(parents=True, exist_ok=True)
    for m in models():
        out = folder / f"{m['name']}.npz"
        assert not out.exists() and not out.with_suffix('.json').exists(), f"test output exists; refusing to run inference again: {m['name']}"
        gpu_free()
        infer(m, 'test', out, env, T / 'T4/test.log')
        print(json.dumps({'name': m['name'], 'output_sha256': sha(out)}), flush=True)
    receipt = {'models': 15, 'outputs': {p.name: sha(p) for p in sorted(folder.glob('*.npz'))},
               'material_sha256': MATERIAL_SHA256, 'finished': time.strftime('%F %T %Z')}
    (T / 'T4/test_inference_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['replay', 'test'])
    ap.add_argument('--only', default='', help='replay only these models (for trial runs, comma separated); replay_check.json is not written')
    a = ap.parse_args()
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
    replay(a.only) if a.phase == 'replay' else test()
