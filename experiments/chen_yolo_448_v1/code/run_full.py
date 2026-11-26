import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import time
import export_448 as e


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    with (e.OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ident=e.identity();key=ident['identity'];assert json.loads((e.OUT/'identity.json').read_text())==ident
        assert json.loads((e.OUT/'pilot_completion.json').read_text())['passed']
        if (e.OUT/'completion.json').exists():raise RuntimeError('448 export already completed')
        dispatch={'generator_identity':key,'workers':12,'max_tasks_per_child':40,'max_inflight':12,
                  'cpu_quota_percent':1200,'memory_high_gib':16,'memory_max_gib':20,
                  'code_sha256':e.p.sha(Path(__file__))}
        frozen=e.OUT/'dispatch_identity.json'
        if frozen.exists():assert json.loads(frozen.read_text())==dispatch
        else:e.p.save(frozen,dispatch)
        all_tasks=e.tasks();done=[];pending=[];started=time.time();resumed=0
        for task in all_tasks:
            path=e.DATA/task['row']['case_id']/'result.json'
            if path.exists():
                if not args.resume:raise RuntimeError('Existing result requires --resume')
                e.verify_result(task,key);done.append({'status':'completed'});resumed+=1
            else:pending.append({**task,'identity':key})
        def status(phase):
            state={'phase':phase,'stage':'full','identity':key,'target_cases':len(all_tasks),
                   'completed_cases':len(done),'resumed_cases':resumed,
                   'failed_cases':sum(x['status']!='completed' for x in done),'workers':12,
                   'elapsed_seconds':time.time()-started,'updated_unix':time.time()}
            e.p.save(e.OUT/'status.json',state);return state
        status('running')
        with ProcessPoolExecutor(max_workers=12,mp_context=mp.get_context('spawn'),initializer=e.initialize,max_tasks_per_child=40) as pool:
            iterator=iter(pending);futures=set();exhausted=False
            while futures or not exhausted:
                while not exhausted and len(futures)<12:
                    task=next(iterator,None)
                    if task is None:exhausted=True;break
                    futures.add(pool.submit(e.process,task))
                if not futures:break
                ready,_=wait(futures,return_when=FIRST_COMPLETED)
                for future in ready:
                    futures.remove(future);record=future.result();done.append(record)
                    if record['status']!='completed':print(json.dumps(record),flush=True)
                    if len(done)%10==0:status('running')
                    if len(done)%100==0:print(json.dumps(status('running')),flush=True)
        state=status('generated')
        if state['failed_cases']:
            status('failed');raise RuntimeError('Some native reconstructions failed; investigate before resuming')
        status('verifying_all448');e.finalize(all_tasks,key)
        e.p.save(e.OUT/'dispatch_completion.json',{'phase':'completed','dispatcher':dispatch,
                 'summary_sha256':e.p.sha(e.OUT/'completion.json'),'elapsed_seconds':time.time()-started})
        print(json.dumps(status('completed')),flush=True)


if __name__=='__main__':main()
