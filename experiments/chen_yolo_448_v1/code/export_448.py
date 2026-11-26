from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import gc
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'experiments/chen_yolo_repair_v1/code'))
import candidates as c
from export_review import saved
from downstream.oa_kl_unified_v1.geometry import extract_regions
from drai.data.dicom_gray import read_pa_archive, presentation_values, normalize_roi
p=c.p
OUT=ROOT/'outputs'
DATA=PROJECT/'data/processed/chen_yolo_448_v1'
PARENT=PROJECT/'data/manifests/chen_yolo_usable_v1_20261111'
MANIFEST=PROJECT/'data/manifests/chen_yolo_usable448_v1_20261111'
POLICY={'output_size':448,'source':'original archive native pixels, never upsample224',
        'inherit':'same selected knee keys, patient split, labels, native bbox and gray variant',
        'side_policy':'image_left=R,image_right=L','spatial_flip':False,
        'resize':'existing float bilinear and bool nearest letterbox',
        'reproduce224':'every selected ROI image, mask and transform must exactly match parent',
        'scope':['train','val'],'training':False,'test_accessed':False}


def identity():
    parent=json.loads((PARENT/'summary.json').read_text())
    assert parent['phase']=='completed' and parent['pending']==0
    for filename,expected in parent['manifest_sha256'].items():assert p.sha(PARENT/filename)==expected
    deps=[Path(__file__),Path(c.__file__),Path(p.__file__),
          PROJECT/'src/drai/data/dicom_gray.py',PROJECT/'downstream/oa_kl_unified_v1/geometry.py']
    value={'policy':POLICY,'parent_summary_sha256':p.sha(PARENT/'summary.json'),
           'parent_manifests':parent['manifest_sha256'],'parent_repair_identity':c.identity()['identity'],
           'dependencies':{str(x):p.sha(x) for x in deps},'numpy_version':np.__version__}
    value['identity']=p.digest(value)
    return value


def tasks():
    rows={x['case_id']:x for x in json.loads((p.OUT/'selection_private.json').read_text())}
    grouped=defaultdict(list);seen=set();counts=Counter();patients=defaultdict(set)
    for split in ['train','val']:
        for line in (PARENT/(split+'_usable.jsonl')).open():
            item=json.loads(line);assert item['split']==split and item['key'] not in seen
            assert item['shape']==[1,224,224] and item['clinical_side']==p.SIDES[item['image_position']]
            row=rows[item['case_id']]
            assert row['split']==split and row['patient_id']==item['patient_id']
            assert row['labels'][item['clinical_side']]==item['kl']
            seen.add(item['key']);grouped[item['case_id']].append(item);counts[split]+=1
            patients[split].add(item['patient_id'])
    assert counts=={'train':34258,'val':4262} and len(grouped)==19264
    assert not patients['train']&patients['val']
    return [{'row':rows[case],'items':items} for case,items in sorted(grouped.items())]


def initialize():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)


def check_array(image,mask,size):
    assert image.shape==mask.shape==(1,size,size)
    assert image.dtype==np.float32 and mask.dtype==bool and mask.any()
    assert np.isfinite(image).all() and 0<=float(image.min())<=float(image.max())<=1
    assert not np.any(image[~mask])


def native_input(item,original,raw,raw_values,inverted,valid):
    variant=item['variant'];bbox=item['bbox'];local=False
    if variant in ['original','full_original','lower_original']:gray=original
    elif variant in ['modality_samebox','center_modality','full_modality','lower_raw']:gray=raw
    elif variant in ['inv_center_raw','inv_full_raw','lower_inv_raw']:gray=inverted
    elif variant=='modality_roi_samebox':
        x0,y0,x1,y1=bbox;valid=valid[y0:y1,x0:x1]
        gray,_=normalize_roi(raw_values[y0:y1,x0:x1],valid)
        bbox=[0,0,gray.shape[1],gray.shape[0]];local=True
    else:raise ValueError('Unknown frozen gray variant '+variant)
    assert gray is not None
    region={'image_position':item['image_position'],'bbox_xyxy_exclusive':bbox,
            'quality_status':'candidate','reject_reasons':[]}
    return gray,valid,region,local


def extract(gray,valid,region,size,item,local):
    crop=extract_regions(gray,valid,[region],image_size=size)[0]
    if local:crop['transform']['local_input_origin_xy']=item['bbox'][:2]
    check_array(crop['image'],crop['valid_mask'],size)
    return crop


def process(task):
    started=time.monotonic();row=task['row'];items=task['items'];case=row['case_id']
    directory=DATA/case;directory.mkdir(parents=True,exist_ok=True)
    result={'case_id':case,'split':row['split'],'identity':task['identity'],
            'parent_items_digest':p.digest(items),'entries':[],'training':False,'test_accessed':False}
    try:
        ds,pixels,receipt=read_pa_archive(p.NAS/row['input_archive_relpath'],
            expected_bytes=int(row['archive_bytes']),source_root=p.NAS)
        assert receipt['archive_sha256']==row['archive_sha256']
        assert receipt['native_pixel_sha256']==row['native_pixel_sha256']
        values,valid,pres=presentation_values(pixels,ds)
        variants={x['variant'] for x in items}
        original=normalize_roi(values,valid)[0] if variants&{'original','full_original','lower_original'} else None
        del values
        raw=raw_values=inverted=None
        if variants-{'original','full_original','lower_original'}:
            raw,raw_values,_=c.modality_gray(pixels,ds,valid,pres)
            if variants&{'inv_center_raw','inv_full_raw','lower_inv_raw'}:
                inverted=np.where(valid,1-raw,0).astype(np.float32)
            if 'modality_roi_samebox' not in variants:raw_values=None
        del pixels
        result.update(source_receipt=receipt,native_hw=list(valid.shape))
        for item in items:
            source=Path(item['source_result']);assert p.sha(source)==item['source_result_sha256']
            previous=json.loads(source.read_text())
            old=(next(x for x in previous['yolo'] if x['image_position']==item['image_position'] and x['status']=='saved')
                 if item['candidate_family']=='base' else saved(previous,item['variant'],item['image_position']))
            assert old['bbox']==item['bbox'] and old['kl']==item['kl']
            for field in ['gray01','valid_mask']:
                assert old[field+'_sha256']==item[field+'_sha256']==p.sha(item[field+'_npy'])
            gray,mask,region,local=native_input(item,original,raw,raw_values,inverted,valid)
            parent=extract(gray,mask,region,224,item,local)
            assert np.array_equal(parent['image'],np.load(item['gray01_npy'],allow_pickle=False)),(item['key'],'224 gray mismatch')
            assert np.array_equal(parent['valid_mask'],np.load(item['valid_mask_npy'],allow_pickle=False)),(item['key'],'224 mask mismatch')
            assert parent['transform']==old['transform'],(item['key'],'224 transform mismatch')
            crop=extract(gray,mask,region,448,item,local)
            image_path=directory/(item['clinical_side']+'_gray01.npy');mask_path=directory/(item['clinical_side']+'_valid.npy')
            p.npy_save(image_path,crop['image']);p.npy_save(mask_path,crop['valid_mask'])
            entry={**item,'shape':[1,448,448],'gray01_npy':str(image_path),'gray01_sha256':p.sha(image_path),
                   'valid_mask_npy':str(mask_path),'valid_mask_sha256':p.sha(mask_path),'transform':crop['transform'],
                   'parent_224_gray01_npy':item['gray01_npy'],'parent_224_gray01_sha256':item['gray01_sha256'],
                   'parent_224_valid_mask_sha256':item['valid_mask_sha256'],
                   'parent_candidate_result':item['source_result'],'parent_candidate_result_sha256':item['source_result_sha256'],
                   'native_reconstruction_224_exact':True,'output_resolution':448,
                   'source_is_native_pixels':True,'resized_from_224':False}
            result['entries'].append(entry)
        result['status']='completed'
    except Exception as exc:
        result.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    result['seconds']=time.monotonic()-started;p.save(directory/'result.json',result);gc.collect()
    return {k:result[k] for k in ['case_id','status','seconds']}


def verify_result(task,key,arrays=True):
    path=DATA/task['row']['case_id']/'result.json';r=json.loads(path.read_text())
    assert r['identity']==key and r['parent_items_digest']==p.digest(task['items']) and r['status']=='completed'
    assert {x['key'] for x in r['entries']}=={x['key'] for x in task['items']}
    old={x['key']:x for x in task['items']}
    for x in r['entries']:
        for f in ['key','bbox','split','clinical_side','image_position','kl','patient_id','candidate_family','variant']:
            assert x[f]==old[x['key']][f]
        assert x['native_reconstruction_224_exact'] and x['source_is_native_pixels'] and not x['resized_from_224']
        for f in ['gray01','valid_mask']:assert p.sha(x[f+'_npy'])==x[f+'_sha256']
        if arrays:check_array(np.load(x['gray01_npy'],allow_pickle=False),np.load(x['valid_mask_npy'],allow_pickle=False),448)
        t=x['transform'];assert t['image_size']==448 and t['source_crop_size_wh']==[x['bbox'][2]-x['bbox'][0],x['bbox'][3]-x['bbox'][1]]
        assert 0<=t['offset_x']<=448-t['resized_width'] and 0<=t['offset_y']<=448-t['resized_height']
    return r


def finalize(all_tasks,key):
    output=defaultdict(list);hashes=defaultdict(set);counts=Counter();result_digests={}
    for task in all_tasks:
        r=verify_result(task,key);path=DATA/r['case_id']/'result.json';sha=p.sha(path);result_digests[r['case_id']]=sha
        for x in r['entries']:
            entry={**x,'source_result':str(path),'source_result_sha256':sha}
            output[x['split']].append(entry);counts[x['split']]+=1;hashes[x['gray01_sha256']].add(x['split'])
    assert counts=={'train':34258,'val':4262} and not any(len(s)>1 for s in hashes.values())
    MANIFEST.mkdir(parents=True,exist_ok=False);files={}
    for split in ['train','val']:
        path=MANIFEST/(split+'_usable.jsonl')
        path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in output[split]))
        assert sum(1 for _ in path.open())==counts[split];files[path.name]=p.sha(path)
    excluded=MANIFEST/'excluded_private.json';excluded.write_bytes((PARENT/excluded.name).read_bytes());files[excluded.name]=p.sha(excluded)
    parent=json.loads((PARENT/'summary.json').read_text())
    report={**parent,'dataset':MANIFEST.name,'count_unit':'single_knee_448_float32_ROI',
            'output_shape':[1,448,448],'output_resolution':448,'phase':'completed','export448_identity':key,
            'parent_224_summary_sha256':p.sha(PARENT/'summary.json'),'parent_224_dataset':PARENT.name,
            'native_source_hashes_verified_cases':len(all_tasks),'native_reconstruction_224_exact_knees':sum(counts.values()),
            'all_selected_448_arrays_verified':True,'parent_selection_bbox_gray_labels_unchanged':True,
            'resized_from_224':False,'manifest_sha256':files,'selected_result_hashes_digest':p.digest(result_digests),
            'code_sha256':p.sha(Path(__file__)),'server_manifest_directory':str(MANIFEST),
            'counts':{split:{**parent['counts'][split],'excluded':parent['counts'][split].get('excluded',0)} for split in ['train','val']}}
    p.save(MANIFEST/'summary.json',report);p.save(OUT/'completion.json',report)
    print(json.dumps({'phase':'completed','train':counts['train'],'val':counts['val'],'shape':[1,448,448]},ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','pilot','full']);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ident=identity();key=ident['identity'];all_tasks=tasks();frozen=OUT/'identity.json'
        if a.stage=='prepare':
            assert not frozen.exists();p.save(frozen,ident)
            strata={}
            for task in all_tasks:
                for x in task['items']:strata.setdefault((x['split'],x['modality'],x['variant']),x['case_id'])
            pilot=sorted(set(strata.values()));p.save(OUT/'pilot_cases.json',pilot)
            p.save(OUT/'processing_authorization.json',{'user_request':'Why 224 x 224? I want 448 x 448',
                   'authorized_scope':'rebuild the confirmed usable list at 448 from the original images, with fixed boxes/gray/side; no new training or test','identity':key})
            print(json.dumps({'prepared':True,'cases':len(all_tasks),'pilot_cases':len(pilot),'identity':key}),flush=True);return
        assert json.loads(frozen.read_text())==ident
        if (OUT/'completion.json').exists():raise RuntimeError('448 export already completed; do not restart')
        chosen=set(json.loads((OUT/'pilot_cases.json').read_text()))
        if a.stage=='full':assert json.loads((OUT/'pilot_completion.json').read_text())['passed']
        selected=all_tasks if a.stage=='full' else [t for t in all_tasks if t['row']['case_id'] in chosen]
        done=[];pending=[];started=time.time()
        for task in selected:
            path=DATA/task['row']['case_id']/'result.json'
            if path.exists():
                if not a.resume:raise RuntimeError('Existing case needs explicit --resume')
                r=verify_result(task,key);done.append({'case_id':r['case_id'],'status':'completed','seconds':r['seconds']})
            else:pending.append({**task,'identity':key})
        def status(phase):
            value={'phase':phase,'stage':a.stage,'identity':key,'target_cases':len(selected),'completed_cases':len(done),
                   'failed_cases':sum(x['status']!='completed' for x in done),'elapsed_seconds':time.time()-started,'updated_unix':time.time()}
            p.save(OUT/'status.json',value);return value
        status('running')
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=initialize,max_tasks_per_child=50) as pool:
            iterator=iter(pending);futures=set();exhausted=False
            while futures or not exhausted:
                while not exhausted and len(futures)<2:
                    task=next(iterator,None)
                    if task is None:exhausted=True;break
                    futures.add(pool.submit(process,task))
                if not futures:break
                ready,_=wait(futures,return_when=FIRST_COMPLETED)
                for f in ready:
                    futures.remove(f);done.append(f.result())
                    if done[-1]['status']!='completed':print(json.dumps(done[-1]),flush=True)
                    if len(done)%10==0:status('running')
                    if len(done)%100==0:print(json.dumps(status('running')),flush=True)
        st=status('generated')
        if st['failed_cases']:
            status('failed');raise RuntimeError('Case failures require investigation; no final manifest written')
        if a.stage=='pilot':
            for task in selected:verify_result(task,key)
            receipt={**st,'passed':True,'phase':'pilot_verified','knees':sum(len(t['items']) for t in selected),
                     'all_reproduced224_exact':True,'mean_case_seconds':sum(x['seconds'] for x in done)/len(done)}
            p.save(OUT/'pilot_completion.json',receipt);print(json.dumps(receipt),flush=True)
        else:
            status('verifying_all448');finalize(all_tasks,key);status('completed')


if __name__=='__main__':main()
