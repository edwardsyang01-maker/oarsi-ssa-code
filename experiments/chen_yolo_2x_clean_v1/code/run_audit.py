import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import cv2
import numpy as np

import audit_rulers as a

DATA=a.PROJECT/'data/processed/chen_yolo_2x_clean_v1/raw'
AUDIT=a.ROOT/'audit_v1'


def worker(path):
    cv2.setNumThreads(1)
    path=Path(path);r=json.loads(path.read_text());assert r['status']=='completed'
    result={'case_id':r['case_id'],'source_result_sha256':a.sha(path),'native_hw':r['native_hw'],'entries':[]}
    for item in r['entries']:
        for field in ['gray01','valid_mask']:assert a.sha(item[field+'_npy'])==item[field+'_sha256']
        g=np.load(item['gray01_npy'],allow_pickle=False);v=np.load(item['valid_mask_npy'],allow_pickle=False)
        assert g.dtype==np.float32 and v.dtype==bool and g.shape==v.shape==(1,448,448)
        assert (g[~v]==0).all()
        d=a.detect(g[0],v[0]);coverage=a.coverage_metrics(item,v[0])
        lines=[]
        for line in d['lines']:
            c=a.classify_line(line,item)
            lines.append({**line,'classification':c})
        likely=[ln for ln in lines if ln['classification']['likely_ruler']]
        mask,met=a.ruler_mask(likely,item,v[0])
        relevant_edge=[b for b in d['edge_blobs'] if b['contrast']/max(b['peak'],1)>=.45 and b['peak']>=75]
        risks=[]
        if met['protected_overlap_pixels']:risks.append('proposed_mask_overlaps_core')
        if met['mask_fraction']>.25:risks.append('large_proposed_mask')
        if any(ln['classification']['regular'] and not ln['classification']['outside_detector_core'] for ln in lines):risks.append('regular_chain_near_joint')
        if not coverage['core_inside_output'] or coverage['core_valid_fraction']<.995:risks.append('joint_core_missing_pixels')
        if not coverage['contains_entire_parent_crop']:raise AssertionError('the expanded box does not contain the original box')
        result['entries'].append({'key':item['key'],'split':item['split'],'side':item['clinical_side'],
                                 'gray01_sha256':item['gray01_sha256'],'valid_mask_sha256':item['valid_mask_sha256'],
                                 'lines':lines,'likely_ruler_count':len(likely),'edge_candidates':relevant_edge,
                                 'gray_metrics':d['gray_metrics'],'coverage':coverage,'mask_proposal_metrics':met,
                                 'review_flags':risks,'ruler_cleaning_applied':False,'visual_review':'pending'})
    a.save(AUDIT/'cases'/(r['case_id']+'.json'),result)
    return {'case_id':r['case_id'],'knees':len(result['entries']),
            'likely_ruler':sum(x['likely_ruler_count']>0 for x in result['entries']),
            'other_edge':sum(x['likely_ruler_count']==0 and bool(x['edge_candidates']) for x in result['entries']),
            'risk':sum(bool(x['review_flags']) for x in result['entries'])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--stream',action='store_true');p.add_argument('--limit',type=int,default=0);args=p.parse_args()
    AUDIT.mkdir(parents=True,exist_ok=True)
    lock=(AUDIT/'audit.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity={'code_sha256':a.sha(__file__),'detector_sha256':a.sha(a.__file__),
              'crop_identity':json.loads((a.ROOT/'outputs/identity.json').read_text())['identity']}
    frozen=AUDIT/'identity.json'
    if frozen.exists():assert args.resume and json.loads(frozen.read_text())==identity
    else:a.save(frozen,identity)
    assert not (AUDIT/'completion.json').exists(),'the completed full screening must not be resubmitted'
    done=set();counts=Counter();start=time.time()
    for f in (AUDIT/'cases').glob('*.json'):
        r=json.loads(f.read_text());path=DATA/r['case_id']/'result.json';assert a.sha(path)==r['source_result_sha256']
        done.add(r['case_id']);counts['knees']+=len(r['entries'])
        counts['likely_ruler']+=sum(x['likely_ruler_count']>0 for x in r['entries'])
        counts['other_edge']+=sum(x['likely_ruler_count']==0 and bool(x['edge_candidates']) for x in r['entries'])
        counts['risk']+=sum(bool(x['review_flags']) for x in r['entries'])
    def status(phase):
        value={'phase':phase,'reviewed_cases':len(done),'target_cases':19264,**dict(counts),'elapsed_seconds':time.time()-start,'updated_unix':time.time()}
        a.save(AUDIT/'status.json',value);return value
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        while True:
            pending=[str(p) for p in sorted(DATA.glob('*/result.json')) if p.parent.name not in done]
            if args.limit:pending=pending[:max(0,args.limit-len(done))]
            for r in pool.map(worker,pending,chunksize=1):
                done.add(r.pop('case_id'));counts.update(r)
                if len(done)%25==0:status('screening')
            status('screened_available')
            if args.limit and len(done)>=args.limit:break
            raw=a.ROOT/'outputs/raw_completion.json'
            if raw.exists():
                completion=json.loads(raw.read_text());assert completion['all_raw_arrays_verified']
                if len(done)==completion['cases']:
                    assert counts['knees']==sum(completion['counts'].values())
                    a.save(AUDIT/'completion.json',{**status('full_automatic_screening_completed'),'identity':identity,
                           'raw_completion_sha256':a.sha(raw),'full_visual_review_completed':False,'ruler_cleaning_completed':False})
                    break
            if not args.stream:break
            status('waiting_for_more_completed_crops');time.sleep(20)
    print(json.dumps(status('completed_available_batch' if not (AUDIT/'completion.json').exists() else 'full_automatic_screening_completed')),flush=True)


if __name__=='__main__':main()
