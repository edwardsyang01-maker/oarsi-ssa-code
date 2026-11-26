import argparse,fcntl,json,multiprocessing as mp,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import audit_rulers as a
import proposal_v1 as p
import ruler_refinement as ref
import clean_policy
from run_audit_v2 import source_paths


def main():
    arg=argparse.ArgumentParser();arg.add_argument('--resume',action='store_true');args=arg.parse_args()
    p.OUT.mkdir(parents=True,exist_ok=True);lock=(p.OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity={'files':{Path(m).name:a.sha(m) for m in [__file__,p.__file__,ref.__file__,clean_policy.__file__,a.__file__]},
              'initial_audit_identity':json.loads((a.ROOT/'audit_v2/identity.json').read_text())}
    frozen=p.OUT/'identity.json'
    if frozen.exists():assert args.resume and json.loads(frozen.read_text())==identity
    else:a.save(frozen,identity)
    assert not (p.OUT/'completion.json').exists(),'completed proposals are not resubmitted'
    done=set();counts=Counter();start=time.time();sources=source_paths()
    for file in (p.OUT/'cases').glob('*/result.json'):
        r=json.loads(file.read_text());assert a.sha(sources[r['case_id']])==r['source_result_sha256']
        done.add(r['case_id']);entries=r['entries'];counts['knees']+=len(entries)
        counts['proposed_clean']+=sum(e['metrics']['mask_pixels']>0 for e in entries)
        counts['review']+=sum(bool(e['review_flags']) for e in entries)
        counts['safe_proposed']+=sum(e['metrics']['mask_pixels']>0 and not e['review_flags'] for e in entries)
    def status(phase):
        record={'phase':phase,'cases':len(done),'target_cases':19264,**dict(counts),'elapsed_seconds':time.time()-start,'updated_unix':time.time()}
        a.save(p.OUT/'status.json',record);return record
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        while True:
            sources=source_paths();available=sorted(f.stem for f in (a.ROOT/'audit_v2/cases').glob('*.json'))
            pending=[str(sources[case]) for case in available if case not in done]
            for result in pool.map(p.worker,pending,chunksize=1):
                done.add(result.pop('case_id'));counts.update(result)
                if len(done)%50==0:status('proposing_masks')
            if len(done)==19264 and (a.ROOT/'audit_v2/completion.json').exists():
                assert counts['knees']==38520
                a.save(p.OUT/'completion.json',{**status('all_proposals_completed'),'identity':identity,
                                               'full_manual_review_completed':False,'clean_arrays_written':False})
                break
            status('waiting_for_more_crops');time.sleep(20)
    print(json.dumps(status('all_proposals_completed')),flush=True)


if __name__=='__main__':main()
