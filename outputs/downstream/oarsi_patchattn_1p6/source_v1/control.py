import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE/'spec.json').read_text())
ROOT = Path(SPEC['project'])
OUTPUT = ROOT/SPEC['output']


def status(state, **extra):
    target = OUTPUT/'controller_status.json'
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps({'state': state, 'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}, indent=2))
    os.replace(tmp, target)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock_lambda():
    spec_sha = sha(HERE/'spec.json')
    result = {'lambda': {}, 'calibration_sha256': {}, 'lambda_target': SPEC['lambda_target'], 'seed': SPEC['seed'],
              'spec_sha256': spec_sha, 'train_sha256': SPEC['train_sha256'], 'training_updates_before_lock': 0,
              'shared': {'P4': 'P3'}, 'method': SPEC['lambda_acceptance']}
    for group in ('P2', 'P3'):
        paths = [OUTPUT/'calibration'/f'calibration_{group}_rank{r}.json' for r in (0, 1)]
        reports = [json.loads(p.read_text()) for p in paths]
        value = reports[0]['lambda']
        assert math.isfinite(value) and value > 0
        for report in reports:
            assert report['group'] == group and report['lambda'] == value and report['spec_sha256'] == spec_sha
            assert report['valid_batches'] >= SPEC['calibration_min_valid_batches']
            assert report['real_optimizer_steps'] == 0 and report['model_optimizer_rng_unchanged']
            assert math.isclose(value, SPEC['lambda_target']*statistics.median(report['all_ratios_main_over_oarsi']), rel_tol=1e-12)
        result['lambda'][group] = value
        result['calibration_sha256'][group] = {str(r): sha(p) for r, p in enumerate(paths)}
    result['locked_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    target = OUTPUT/SPEC['lambda_lock']
    assert not target.exists(), 'LAMBDA_ALREADY_LOCKED'
    temporary = target.with_suffix('.pending')
    with temporary.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
    os.rename(temporary, target)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert not (HERE/'training_authorization.json').exists(), 'UNEXPECTED_TRAINING_AUTHORIZATION'
    assert not SPEC['training_authorized'] and not SPEC['test_access_permitted']
    assert not (OUTPUT/SPEC['lambda_lock']).exists(), 'LAMBDA_ALREADY_LOCKED_DO_NOT_RERUN'
    pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).split()
    desktop = [p for p in pids if os.path.realpath(f'/proc/{p}/exe') == '/usr/libexec/gnome-remote-desktop-daemon']
    assert sorted(desktop) == sorted(pids), 'GPU_BUSY_DO_NOT_START'
    (OUTPUT/'gpu_precheck.json').write_text(json.dumps({'compute_pids': pids, 'ignored_desktop_daemon_pids': desktop,
        'at': datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=2)+'\n')
    os.environ.update({'CUDA_VISIBLE_DEVICES': 'GPU-214fb1e9-6b50-4251-e989-073b71ceb7b5,GPU-dc85ba0e-cc3b-9b95-bfb9-063c8df0c2ae',
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'NCCL_P2P_DISABLE': '1', 'NCCL_DEBUG': 'WARN',
        'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
        'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
    os.sched_setaffinity(0, {8, 9, 18, 19})
    os.nice(10)
    run = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', str(HERE/'runner.py')]
    stages = [('unit', [sys.executable, '-B', str(HERE/'test_core.py')]),
              ('calibration_P2', run+['--group', 'P2', '--mode', 'calibration']),
              ('calibration_P3', run+['--group', 'P3', '--mode', 'calibration'])]
    for group in SPEC['run_order']:
        for mode in ('preflight_save', 'preflight_resume'):
            stages.append((group+'_'+mode, run+['--group', group, '--mode', mode]))
    done = []
    for name, command in stages:
        status('running', stage=name, completed=done, real_training_started=False)
        with (OUTPUT/(name+'.log')).open('x') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
        if result.returncode:
            status('failed', stage=name, returncode=result.returncode, completed=done, real_training_started=False)
            return result.returncode
        if name == 'calibration_P3':
            lock_lambda()
        done.append(name)
    status('preflight_completed', completed=done, real_training_started=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
