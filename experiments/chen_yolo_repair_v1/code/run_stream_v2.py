import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import candidates as c
p=c.p


def eligible(result):
    if result.get('prior_case_id','').startswith('C'):return True
    if result['status']!='completed':return True
    saved=[x for x in result['yolo'] if x['status']=='saved']
    if len(saved)!=2:return True
    return any(set(x['flags'])-{'detector_center_padding'} for x in saved)


def verify_existing(task,repair_identity):
    path=c.DATA/'full'/task['row']['case_id']/'result.json'
    if not path.exists():return False
    r=json.loads(path.read_text())
    if r['repair_identity']!=repair_identity or r['film_id']!=task['row']['film_id']:
        raise RuntimeError('Repair checkpoint identity mismatch')
    if r['baseline_result_sha256']!=p.sha(Path(task['baseline_path'])):
        raise RuntimeError('Baseline checkpoint changed')
    for items in r['variants'].values():
        for roi in items:
            if roi['status']=='saved':
                for field in ['gray01','valid_mask']:
                    if p.sha(roi[field+'_npy'])!=roi[field+'_sha256']:
                        raise RuntimeError('Repair checkpoint hash mismatch')
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--max-tasks-per-child',type=int,default=24)
    args=parser.parse_args()
    assert 1<=args.workers<=4 and 1<=args.max_tasks_per_child<=48
    ident=c.identity();key=ident['identity'];pre=json.loads((c.OUT/'pilot_technical_verification.json').read_text())
    if pre['repair_identity']!=key or pre['cases_checked']!=48 or not pre['source_hashes_verified']:
        raise RuntimeError('Pilot evidence does not match repair implementation')
    auth=json.loads((c.OUT/'processing_authorization.json').read_text())
    if auth.get('approved') is not True or auth['repair_identity']!=key:raise PermissionError('Repair authorization mismatch')
    if (c.OUT/'stream_completion.json').exists():raise RuntimeError('Repair generation already completed')
    if (c.OUT/'stream_status.json').exists() and not args.resume:raise RuntimeError('Explicit --resume required')
    with (c.OUT/'stream.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p.save(c.OUT/'full_identity.json',ident)
        rows=json.loads((p.OUT/'selection_private.json').read_text())
        seen=set();pending=[];tasks={};done=set();errors=[];futures={}
        started=time.time();last_scan=0;last_save=0
        state={'repair_identity':key,'base_run_id':ident['base_run_id'],'started_unix':started,
               'pid':os.getpid(),'orchestration_sha256':p.sha(Path(__file__)),'resume':args.resume,
               'training':False,'test_accessed':False,'quality_decisions_finalized':False,'workers':args.workers,
               'max_tasks_per_child':args.max_tasks_per_child,'runtime_revision':'bounded-worker-v2'}
        try:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=c.initialize,
                                     max_tasks_per_child=args.max_tasks_per_child) as pool:
                while True:
                    now=time.time();base_complete=(p.OUT/'completion.json').exists()
                    if now-last_scan>=20 or not pending and not futures:
                        base_job=json.loads((p.OUT/'job_status.json').read_text())
                        if base_job['phase']=='failed':raise RuntimeError('Base crop job failed; inspect it before continuing')
                        for row in rows:
                            case=row['case_id']
                            if case in seen:continue
                            path=p.DATA/case/'result.json'
                            if not path.exists():continue
                            baseline=json.loads(path.read_text())
                            if baseline['run_id']!=ident['base_run_id']:raise RuntimeError('Unexpected base identity')
                            seen.add(case)
                            if not eligible(baseline):continue
                            task={'row':row,'phase':'full','identity':key,'baseline_path':str(path)}
                            tasks[case]=task
                            if verify_existing(task,key):
                                if not args.resume:raise RuntimeError('Preexisting repair output requires --resume')
                                done.add(case)
                            else:pending.append(case)
                        last_scan=time.time()
                    cap=args.workers
                    while pending and len(futures)<cap:
                        case=pending.pop(0);futures[pool.submit(c.process_case,tasks[case])]=case
                    if futures:
                        ready,_=wait(futures,timeout=2,return_when=FIRST_COMPLETED)
                        for future in ready:
                            case=futures.pop(future);record=future.result();done.add(case)
                            if record['status']=='case_error':errors.append(record)
                            print(json.dumps(record),flush=True)
                    else:time.sleep(2)
                    if time.time()-last_save>=10:
                        p.save(c.OUT/'stream_status.json',{**state,'phase':'generating_candidates',
                               'base_cases_seen':len(seen),'base_target':len(rows),'eligible_cases':len(tasks),
                               'completed':len(done),'pending':len(pending),'in_flight':len(futures),
                               'case_errors':len(errors),'base_complete':base_complete,'updated_unix':time.time()})
                        last_save=time.time()
                    if base_complete and len(seen)==len(rows) and not futures and not pending:break
            p.save(c.OUT/'full_selection_private.json',{'repair_identity':key,'case_ids':sorted(tasks),
                    'rule':'any missing/failed YOLO side or flags other than padding-only, plus original48 audit cases'})
            p.save(c.OUT/'full_tasks_private.json',[tasks[k] for k in sorted(tasks)])
            result={**state,'phase':'candidates_complete','eligible_cases':len(tasks),'completed':len(done),
                    'case_errors':errors,'base_cases_seen':len(seen),'elapsed_seconds':time.time()-started}
            p.save(c.OUT/'stream_completion.json',result);p.save(c.OUT/'stream_status.json',result)
        except Exception as exc:
            p.save(c.OUT/'stream_status.json',{**state,'phase':'failed','completed':len(done),
                    'error':str(exc),'traceback':traceback.format_exc()})
            raise


if __name__=='__main__':main()

