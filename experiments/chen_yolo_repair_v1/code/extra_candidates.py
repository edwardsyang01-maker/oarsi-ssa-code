import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import fcntl
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import numpy as np
import candidates as c
p=c.p
POLICY={'variants':['inv_center_raw','inv_full_raw','lower_original','lower_raw','lower_inv_raw'],
        'lower_field_native_y_fraction':.45,'invert':'1-gray on valid pixels; invalid remains zero',
        'spatial_flip':False,'threshold':.12,'nms':.3,'expand':1.3,'training':False,
        'side_policy':p.SIDES,'selection_uses_labels':False}


def identity():
    d={'policy':POLICY,'parent_repair_identity':c.identity()['identity'],'code_sha256':p.sha(Path(__file__))}
    d['identity']=p.digest(d);return d


def process(task):
    from drai.data.dicom_gray import read_pa_archive,presentation_values,normalize_roi
    case,key=task;src=c.DATA/'full'/case/'result.json';baseline=json.loads(src.read_text())
    directory=c.DATA/'extra'/case;directory.mkdir(parents=True,exist_ok=True)
    r={k:v for k,v in baseline.items() if k not in ['variants'] and not k.endswith('_detection')}
    r.update(extra_identity=key,parent_repair_result_sha256=p.sha(src),variants={},phase='extra',
             training_ready=False,clinical_quality_approved=False,source_generator_sha256=p.sha(Path(__file__)))
    try:
        ds,pixels,receipt=read_pa_archive(p.NAS/r['input_archive_relpath'],expected_bytes=int(r['archive_bytes']),source_root=p.NAS)
        assert receipt['archive_sha256']==r['archive_sha256'] and receipt['native_pixel_sha256']==r['native_pixel_sha256']
        values,valid,pres=presentation_values(pixels,ds);original,_=normalize_roi(values,valid);del values
        raw,raw_values,raw_receipt=c.modality_gray(pixels,ds,valid,pres);del raw_values,pixels
        inverted=np.where(valid,1-raw,0).astype(np.float32);h,w=raw.shape;y0=round(h*.45)
        specs=[('inv_center_raw',inverted,'center',0),('inv_full_raw',inverted,'full',0),
               ('lower_original',original,'full',y0),('lower_raw',raw,'full',y0),('lower_inv_raw',inverted,'full',y0)]
        for variant,gray,mode,origin in specs:
            boxes,evidence=c.mapped_candidates(gray[origin:,:],ds,mode)
            if origin:
                for item in boxes:
                    for field in ['bbox','unclipped_bbox']:
                        item[field][1]+=origin;item[field][3]+=origin
                evidence['source_field_origin_xy']=[0,origin]
                evidence['source_native_hw']=[h,w]
            r['variants'][variant]=[p.write_roi(gray,valid,item,directory,variant,r['labels']) for item in boxes]
            r[variant+'_detection']=evidence
        r['status']='candidates_generated'
    except Exception as exc:r.update(status='case_error',error=str(exc),traceback=traceback.format_exc())
    p.save(directory/'result.json',r)
    return {'case_id':case,'status':r['status'],'saved':{v:sum(x['status']=='saved' for x in xs) for v,xs in r['variants'].items()}}


def main():
    a=argparse.ArgumentParser();a.add_argument('--selection',required=True);a.add_argument('--resume',action='store_true');args=a.parse_args()
    selection=Path(args.selection);cases=json.loads(selection.read_text());assert cases and len(cases)==len(set(cases))
    ident=identity();key=ident['identity'];run=selection.stem
    auth=json.loads((c.OUT/'processing_authorization.json').read_text());assert auth['approved']
    with (c.OUT/'extra.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=c.OUT/'extra_identity.json'
        if frozen.exists():assert json.loads(frozen.read_text())==ident
        else:p.save(frozen,ident)
        done=[];pending=[]
        for case in cases:
            path=c.DATA/'extra'/case/'result.json'
            if path.exists():
                if not args.resume:raise RuntimeError('Existing extra candidate requires explicit --resume')
                r=json.loads(path.read_text());assert r['extra_identity']==key
                assert r['parent_repair_result_sha256']==p.sha(c.DATA/'full'/case/'result.json')
                for xs in r['variants'].values():
                    for x in xs:
                        if x['status']=='saved':
                            for f in ['gray01','valid_mask']:assert p.sha(x[f+'_npy'])==x[f+'_sha256']
                done.append({'case_id':case,'status':r['status']})
            else:pending.append((case,key))
        state={'extra_identity':key,'selection_sha256':p.sha(selection),'target':len(cases),'training':False,'test_accessed':False}
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=c.initialize,max_tasks_per_child=20) as pool:
            iterator=iter(pending);futures=set();exhausted=False
            while futures or not exhausted:
                while not exhausted and len(futures)<2:
                    task=next(iterator,None)
                    if task is None:exhausted=True;break
                    futures.add(pool.submit(process,task))
                if not futures:break
                ready,_=wait(futures,return_when=FIRST_COMPLETED)
                for future in ready:
                    futures.remove(future);record=future.result();done.append(record)
                    p.save(c.OUT/(run+'_status.json'),{**state,'phase':'running','completed':len(done),'updated_unix':time.time()})
                    print(json.dumps(record),flush=True)
        p.save(c.OUT/(run+'_completion.json'),{**state,'phase':'completed','completed':len(done),
                                             'case_errors':sum(x['status']=='case_error' for x in done),'quality_decisions_finalized':False})


if __name__=='__main__':main()
