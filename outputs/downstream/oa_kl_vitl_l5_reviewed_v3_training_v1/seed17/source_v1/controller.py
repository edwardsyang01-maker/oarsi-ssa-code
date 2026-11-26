import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import traceback

ROOT=Path('/path/to/drai')
HERE=Path(__file__).resolve().parent
PLAN=json.loads((HERE/'plan.json').read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(p,data):
    t=p.with_suffix('.tmp')
    with t.open('w') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(t,p)


def identity(pid):
    s=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()
    return {'pid':pid,'start_ticks':int(s[19])}


def guard():
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    for row in rows.splitlines():
        if not row.strip():continue
        pid,uuid=[v.strip() for v in row.split(',')]
        if uuid not in PLAN['gpu_uuids']:continue
        p=int(pid); ident=identity(p)
        assert ident=={k:PLAN['allowed_desktop_process'][k] for k in ['pid','start_ticks']} and str((Path('/proc')/pid/'exe').resolve())==PLAN['allowed_desktop_process']['exe'],'GPU_BUSY'


def verify_sources():
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    for name,digest in manifest['local'].items():assert sha(HERE/name)==digest,'LOCAL_SOURCE_CHANGED'
    for name,digest in manifest['dependencies'].items():assert sha(ROOT/name)==digest,'DEPENDENCY_SOURCE_CHANGED'
    for name,digest in PLAN['weights_sha256'].items():assert sha(ROOT/PLAN['weights_directory']/name)==digest,'WEIGHT_CHANGED'
    assert sha(ROOT/PLAN['manifest']['path'])==PLAN['manifest']['sha256'],'MANIFEST_CHANGED'
    return {'plan_sha256':sha(HERE/'plan.json'),'source_manifest_sha256':sha(HERE/'source_manifest.json')}


def launch(mode, run, state, resume=False):
    guard()
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env=os.environ.copy();env.update({'CUDA_VISIBLE_DEVICES':','.join(PLAN['gpu_uuids']),
        'CUDA_DEVICE_ORDER':'PCI_BUS_ID','NCCL_P2P_DISABLE':'1','NCCL_DEBUG':'WARN',
        'TORCH_NCCL_ASYNC_ERROR_HANDLING':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8',
        'OMP_NUM_THREADS':str(PLAN['cpu_threads_per_rank']),'MKL_NUM_THREADS':str(PLAN['cpu_threads_per_rank']),'OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
    command=[sys.executable,'-B','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=2',
             '--max_restarts=0','--master_addr=127.0.0.1',f'--master_port={port}',str(HERE/'trainer.py'),'--mode',mode]
    if resume:command.append('--resume')
    log_path=run/(mode+('_resume' if resume else '')+'_raw.log')
    with log_path.open('xb') as log:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(state='running',phase=mode,child=identity(child.pid),updated_at=dt.datetime.now(dt.timezone.utc).isoformat())
        write(run/'controller_status.json',state)
        try:rc=child.wait(timeout=None if mode=='train' else 3600)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=20)
            raise RuntimeError('PREFLIGHT_TIMEOUT')
    assert rc==0,'CHILD_FAILED'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['preflight','train']);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();assert Path.cwd()==ROOT
    os.sched_setaffinity(0,set(PLAN['cpu_affinity_logical_ids']))
    os.nice(max(0,PLAN['cpu_nice']-os.getpriority(os.PRIO_PROCESS,0)))
    assert sorted(os.sched_getaffinity(0))==PLAN['cpu_affinity_logical_ids']
    run=ROOT/PLAN['run_directory' if args.mode=='train' else 'preflight_directory'];run.mkdir(parents=True,exist_ok=True)
    state={'state':'preparing','mode':args.mode,'identity':identity(os.getpid()),'started_at':dt.datetime.now(dt.timezone.utc).isoformat()}
    locks=[]
    try:
        marker=run/('explicit_resume_marker' if args.resume else 'started_once')
        fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
        for i in [0,1]:
            f=(ROOT/f'outputs/downstream/oa_kl_wholefilm_v2/gpu{i}_queue.lock').open('a+')
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        guard();binding=verify_sources();state['binding']=binding
        assert shutil.disk_usage(ROOT).free>PLAN['minimum_free_data_gib']*1024**3,'DATA_DISK_SPACE'
        backup=Path(PLAN['backup_directory']); backup.parent.mkdir(parents=True,exist_ok=True)
        assert backup.is_relative_to('/path/to/drai/outputs/downstream/oa_kl_vitl_l5_reviewed_v3_training_v1/seed17/recovery'),'BACKUP_PATH_OUTSIDE_NEW_RUN'
        assert ROOT.stat().st_dev==backup.parent.stat().st_dev,'BACKUP_STORAGE_IDENTITY'
        assert shutil.disk_usage(backup.parent).free>max(PLAN['minimum_free_backup_gib'],PLAN['aggregate_storage_budget_gib'])*1024**3,'BACKUP_DISK_SPACE'
        if args.mode=='preflight':
            launch('preflight_save',run,state);launch('preflight_resume',run,state)
            receipts=[json.loads((run/f'preflight_rank{i}.json').read_text()) for i in [0,1]]
            assert all(x['passed'] for x in receipts),'PREFLIGHT_FAILED'
            memory=[json.loads((run/f'{mode}_memory_rank{i}.json').read_text())
                    for mode in ['preflight_save','preflight_resume'] for i in [0,1]]
            assert all(m['passed'] for m in memory),'PREFLIGHT_MEMORY_STAGE_FAILED'
            assert all(m['reserved_peak_bytes'] < PLAN['gpu_reserved_limit_bytes'] for m in memory),'GPU_MEMORY_BUDGET'
            write(run/'acceptance.json',{'passed':True,'binding':binding,'rank_checks':[r['checks'] for r in receipts],
                'completed_at':dt.datetime.now(dt.timezone.utc).isoformat(),'synthetic_only':True,'memory_reserved_limit_bytes':PLAN['gpu_reserved_limit_bytes']})
        else:
            auth=json.loads((HERE/'authorization.json').read_text())
            assert auth['authorized'] is True and auth['binding']==binding and auth['scope']==PLAN['run_name']
            preflight=ROOT/PLAN['preflight_directory']/'acceptance.json'
            assert sha(preflight)==auth['preflight_acceptance_sha256'] and json.loads(preflight.read_text())['passed']
            launch('train',run,state,resume=args.resume)
            result=json.loads((run/'result.json').read_text())
            assert result['progress']['step']==PLAN['total_steps'] and result['progress']['epoch']==PLAN['epochs']
        state['state']='completed'
    except BaseException as exc:
        traceback.print_exc();state.update(state='failed',error_type=type(exc).__name__,
            error_code=str(exc) if str(exc).isupper() and len(str(exc))<100 else 'SEE_SERVER_RAW_LOG')
    finally:
        state['finished_at']=dt.datetime.now(dt.timezone.utc).isoformat();write(run/'controller_status.json',state)
        for f in locks:f.close()
    return 0 if state['state']=='completed' else 2


if __name__=='__main__':sys.exit(main())
