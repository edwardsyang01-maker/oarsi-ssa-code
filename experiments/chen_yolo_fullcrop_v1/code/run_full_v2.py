import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import pipeline as p



def bounded_crop(workers,run_id):
    import fcntl
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    auth=json.loads((p.OUT/'cropping_authorization.json').read_text())
    assert auth['approved'] and auth['run_id']==run_id
    with (p.OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=json.loads((p.OUT/'selection_private.json').read_text())
        done={row['case_id'] for row in rows if p.check_checkpoint(row,run_id)}
        pending=iter(row for row in rows if row['case_id'] not in done)
        futures={};exhausted=False
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),
                                 initializer=p.initializer,initargs=(run_id,),max_tasks_per_child=24) as pool:
            while futures or not exhausted:
                while not exhausted and len(futures)<workers:
                    row=next(pending,None)
                    if row is None:exhausted=True;break
                    futures[pool.submit(p.process_case,row)]=row['case_id']
                if not futures:break
                ready,_=wait(futures,return_when=FIRST_COMPLETED)
                for future in ready:
                    case=futures.pop(future);record=future.result();done.add(case)
                    p.save(p.OUT/'status.json',{'phase':'running','completed':len(done),'total':len(rows),
                           'run_id':run_id,'last':record,'workers':workers,'max_tasks_per_child':24,
                           'pid':os.getpid(),'updated_unix':time.time(),'runtime_revision':'bounded-worker-v2'})
                    print(json.dumps(record),flush=True)
        p.save(p.OUT/'status.json',{'phase':'completed','completed':len(done),'total':len(rows),'run_id':run_id})
        p.report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, choices=[2,4], default=4)
    args = parser.parse_args()
    run_id = p.verify_identity()
    preflight = json.loads((p.OUT/'preflight.json').read_text())
    if not preflight.get('passed') or preflight.get('run_id') != run_id:
        raise RuntimeError('Matching smoke and resume evidence is required')
    if (p.OUT/'completion.json').exists():
        raise RuntimeError('Completed full crop job must not be restarted')
    start = time.time()
    status = {'run_id':run_id,'pid':os.getpid(),'started_unix':start,
              'workers':args.workers,'device':'cpu','detector_retraining':False,
              'new_classification_training':False,'explicit_resume':True,
              'runtime_revision':'bounded-worker-v2','orchestration_sha256':p.sha(__file__)}
    try:
        p.save(p.OUT/'job_status.json',{**status,'phase':'cropping'})
        bounded_crop(args.workers,run_id)
        p.save(p.OUT/'job_status.json',{**status,'phase':'verifying'})
        subprocess.run([sys.executable,str(p.ROOT/'code/verify_outputs.py')],check=True)
        p.save(p.OUT/'job_status.json',{**status,'phase':'building_review'})
        subprocess.run([sys.executable,str(p.ROOT/'code/make_review.py')],check=True)
        summary = json.loads((p.OUT/'summary.json').read_text())
        verification = json.loads((p.OUT/'verification.json').read_text())
        gallery = json.loads((p.OUT/'review_gallery_receipt.json').read_text())
        if not summary['all_candidates_complete'] or not verification['passed']:
            raise RuntimeError('Final completeness or integrity gate failed')
        if gallery['records'] != summary['processed_films']:
            raise RuntimeError('Review inventory is incomplete')
        result = {**status,'phase':'completed','elapsed_seconds':time.time()-start,
                  'processed_films':summary['processed_films'],
                  'summary_sha256':p.sha(p.OUT/'summary.json'),
                  'verification_sha256':p.sha(p.OUT/'verification.json'),
                  'review_gallery_sha256':gallery['gallery_sha256'],
                  'clinical_quality_approved':False,'training_ready':False,
                  'test_images_or_labels_opened':False}
        p.save(p.OUT/'completion.json',result)
        p.save(p.OUT/'job_status.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as exc:
        p.save(p.OUT/'job_status.json',{**status,'phase':'failed','elapsed_seconds':time.time()-start,
                                     'error':str(exc),'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()

