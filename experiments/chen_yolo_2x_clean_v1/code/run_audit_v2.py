from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import argparse
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import time

import audit_rulers as a
import run_audit as base

AUDIT=a.ROOT/'audit_v2'
REPAIR=a.PROJECT/'data/processed/chen_yolo_2x_clean_v1/raw_repaired_lower'


def worker(path):
    base.AUDIT=AUDIT
    return base.worker(path)


def source_paths():
    paths={p.parent.name:p for p in base.DATA.glob('*/result.json')}
    paths.update({p.parent.name:p for p in REPAIR.glob('*/result.json')})
    return paths


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');args=p.parse_args()
    AUDIT.mkdir(parents=True,exist_ok=True);lock=(AUDIT/'audit.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    repair=json.loads((a.ROOT/'repair_lower_v1/completion.json').read_text());assert repair['phase']=='completed'
    identity={'code_sha256':a.sha(__file__),'pixel_audit_sha256':a.sha(base.__file__),'detector_sha256':a.sha(a.__file__),
              'crop_identity':json.loads((a.ROOT/'outputs/identity.json').read_text())['identity'],'lower_repair_identity':repair['identity']}
    f=AUDIT/'identity.json'
    if f.exists():assert args.resume and json.loads(f.read_text())==identity
    else:a.save(f,identity)
    assert not (AUDIT/'completion.json').exists(),'do not resubmit the completed full review'
    done=set();counts=Counter();start=time.time();sources=source_paths()
    for f in (AUDIT/'cases').glob('*.json'):
        r=json.loads(f.read_text());assert a.sha(sources[r['case_id']])==r['source_result_sha256']
        done.add(r['case_id']);counts['knees']+=len(r['entries'])
        counts['likely_ruler']+=sum(x['likely_ruler_count']>0 for x in r['entries'])
        counts['other_edge']+=sum(x['likely_ruler_count']==0 and bool(x['edge_candidates']) for x in r['entries'])
        counts['risk']+=sum(bool(x['review_flags']) for x in r['entries'])
    def status(phase):
        v={'phase':phase,'reviewed_cases':len(done),'target_cases':19264,**dict(counts),'elapsed_seconds':time.time()-start,'updated_unix':time.time()}
        a.save(AUDIT/'status.json',v);return v
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        while True:
            sources=source_paths();pending=[]
            for case,path in sorted(sources.items()):
                if case in done:continue
                r=json.loads(path.read_text())
                if r['status']=='completed':pending.append(str(path))
            for r in pool.map(worker,pending,chunksize=1):
                done.add(r.pop('case_id'));counts.update(r)
                if len(done)%25==0:status('screening')
            status('screened_available')
            raw=a.ROOT/'outputs/raw_completion.json'
            if raw.exists():
                report=json.loads(raw.read_text());assert report['all_raw_arrays_verified']
                if len(done)==report['cases']:
                    assert counts['knees']==sum(report['counts'].values())
                    a.save(AUDIT/'completion.json',{**status('full_automatic_screening_completed'),'identity':identity,
                           'raw_completion_sha256':a.sha(raw),'full_visual_review_completed':False,'ruler_cleaning_completed':False})
                    break
            status('waiting_for_more_completed_crops');time.sleep(20)
    print(json.dumps(status('full_automatic_screening_completed')),flush=True)


if __name__=='__main__':main()
