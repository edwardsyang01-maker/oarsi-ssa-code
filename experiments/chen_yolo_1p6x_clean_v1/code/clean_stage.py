import argparse,fcntl,hashlib,json,multiprocessing as mp,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2,numpy as np
import audit_rulers as a
import proposal_v1 as p
import ruler_refinement as ref

CONFIG=a.ROOT/'code/clean_decisions_v1.json'
DATA=a.PROJECT/'data/processed/chen_yolo_1p6x_clean_v1'


def atomic_array(path,array):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp.npy')
    np.save(temp,array,allow_pickle=False);temp.replace(path)
    restored=np.load(path,allow_pickle=False)
    assert restored.dtype==array.dtype and np.array_equal(restored,array)
    return str(path),a.sha(path)


def overrides():
    config=json.loads(CONFIG.read_text());out={}
    mappings={'pilot':a.REVIEW/'p_private.json','q01':a.ROOT/'review_queue/q01/mapping_private.json'}
    for rule in config['manual_overrides']:
        rows=json.loads(mappings[rule['cohort']].read_text());row=next(r for r in rows if r['alias']==rule['alias'])
        out[row['item']['key']]={**rule,'raw_gray01_sha256':row['item']['gray01_sha256']}
    return out


def protection(mask,item):
    yy,xx=np.mgrid[:448,:448];b=a.detector_box(item,1.3)
    protected=(xx>=b[0])&(xx<=b[2])&(yy>=b[1])&(yy<=b[3])
    return int((mask&protected).sum())


def worker(task):
    proposal_file,stage_name,identity,manual=task
    cv2.setNumThreads(1);proposal_file=Path(proposal_file);proposal=json.loads(proposal_file.read_text())
    source_path=Path(proposal['source_result_path']);assert a.sha(source_path)==proposal['source_result_sha256']
    source=json.loads(source_path.read_text());entries=[];folder=DATA/stage_name/source['case_id']
    for item in source['entries']:
        entry=next(e for e in proposal['entries'] if e['key']==item['key'])
        assert a.sha(item['gray01_npy'])==item['gray01_sha256'] and a.sha(item['valid_mask_npy'])==item['valid_mask_sha256']
        g=np.load(item['gray01_npy'],allow_pickle=False);v=np.load(item['valid_mask_npy'],allow_pickle=False)
        assert g.shape==v.shape==(1,448,448) and g.dtype==np.float32 and v.dtype==bool
        assert np.isfinite(g).all() and g.min()>=0 and g.max()<=1 and (g[~v]==0).all()
        assert a.sha(entry['proposed_mask_npy'])==entry['proposed_mask_sha256']
        mask=np.load(entry['proposed_mask_npy'],allow_pickle=False)
        assert mask.dtype==bool and mask.shape==g.shape and int(mask.sum())==entry['metrics']['mask_pixels']
        reasons=list(entry['review_flags']);manual_rule=manual.get(item['key'])
        if manual_rule:
            assert manual_rule['raw_gray01_sha256']==item['gray01_sha256']
            if manual_rule['action']=='hold':reasons.append('manual_uncertain_fragment')
            else:
                assert manual_rule['action']=='mask_rectangle'
                x0,y0,x1,y1=manual_rule['rectangle_xyxy'];mask=np.zeros(g.shape,bool);mask[:,y0:y1,x0:x1]=True;mask&=v
                reasons=[] if protection(mask[0],item)==0 else ['manual_mask_touches_original_joint_roi']
        overlap=protection(mask[0],item)
        if overlap and not reasons:reasons.append('mask_touches_original_joint_roi')
        record={**item,'raw_gray01_npy':item['gray01_npy'],'raw_gray01_sha256':item['gray01_sha256'],
                'raw_valid_mask_npy':item['valid_mask_npy'],'raw_valid_mask_sha256':item['valid_mask_sha256'],
                'cleanup_stage':stage_name,'cleanup_identity':identity,'cleanup_review_reasons':reasons,
                'ruler_removal_applied':False,'engineering_input_ready':False,'clinical_quality_approved':False,
                'automated_full_image_audit':True,'manual_override':manual_rule,
                'proposed_mask_pixels':int(mask.sum()),'cleanup_mask_npy':None,'cleanup_mask_sha256':None,
                'unmasked_pixels_bit_exact':True,'postclean_high_confidence_rulers':[]}
        if reasons:
            record['cleanup_status']='held_unmodified'
            record['ruler_review']='quarantined_for_uncertainty_or_joint_protection'
        elif not mask.any():
            record['cleanup_status']='accepted_unchanged';record['engineering_input_ready']=True
            record['ruler_review']='no_actionable_ruler_detected_by_full_automatic_audit'
        else:
            assert overlap==0
            clean=g.copy();clean[mask]=0;clean_valid=v&~mask
            assert np.array_equal(clean[~mask],g[~mask]) and (clean[mask]==0).all() and (clean[~clean_valid]==0).all()
            assert np.array_equal(clean_valid[~mask],v[~mask])
            side=item['clinical_side']
            record['gray01_npy'],record['gray01_sha256']=atomic_array(folder/(side+'_gray01.npy'),clean)
            record['valid_mask_npy'],record['valid_mask_sha256']=atomic_array(folder/(side+'_valid.npy'),clean_valid)
            record['cleanup_mask_npy'],record['cleanup_mask_sha256']=atomic_array(folder/(side+'_ruler_mask.npy'),mask)
            residual=[]
            for line in ref.detect(clean[0],clean_valid[0],item):
                evidence=p.confidence(line,item,source['entries'])
                if evidence['high_confidence']:residual.append({**line,'evidence':evidence})
            record['postclean_high_confidence_rulers']=residual
            record['ruler_removal_applied']=True
            record['cleanup_status']='held_residual_candidate' if residual else 'accepted_cleaned'
            record['engineering_input_ready']=not residual
            record['ruler_review']='residual_candidate_quarantined' if residual else 'cleaned_and_rescreened'
            if residual:record['cleanup_review_reasons'].append('postclean_ruler_candidate')
        entries.append(record)
    result={'case_id':source['case_id'],'identity':identity,'proposal_result_path':str(proposal_file),
            'proposal_result_sha256':a.sha(proposal_file),'source_result_path':str(source_path),
            'source_result_sha256':a.sha(source_path),'native_hw':source['native_hw'],'entries':entries}
    a.save(a.ROOT/stage_name/'cases'/(source['case_id']+'.json'),result)
    return {'case_id':source['case_id'],**counts_for(entries)}


def counts_for(entries):
    c=Counter(knees=len(entries))
    for e in entries:
        c[e['cleanup_status']]+=1;c[e['split']+'_total']+=1
        c[e['split']+('_usable' if e['engineering_input_ready'] else '_held')]+=1
        c['changed_arrays']+=e['ruler_removal_applied']
    return dict(c)


def pilot_cases():
    groups={};all_paths=sorted((p.OUT/'cases').glob('*/result.json'))
    for path in all_paths:
        r=json.loads(path.read_text())
        for e in r['entries']:
            tags=[e['split'],'held' if e['review_flags'] else ('clean' if e['metrics']['mask_pixels'] else 'unchanged')]
            if any(l.get('projected_from_same_film') for l in e['lines']):tags.append('projected')
            if any(l['evidence'].get('dark_regular_channel') for l in e['lines']):tags.append('dark')
            for tag in tags:groups.setdefault(tag,[]).append(str(path))
    chosen=set()
    for tag,paths in sorted(groups.items()):
        chosen.update(sorted(set(paths),key=lambda s:hashlib.sha256((tag+s).encode()).hexdigest())[:8])
    for path in sorted(map(str,all_paths),key=lambda s:hashlib.sha256(s.encode()).hexdigest()):
        if len(chosen)>=64:break
        chosen.add(path)
    return sorted(chosen)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['pilot','full']);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    stage_name='clean_pilot_v1' if args.mode=='pilot' else 'clean_v1';root=a.ROOT/stage_name;root.mkdir(parents=True,exist_ok=True)
    lock=(root/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manual=overrides();identity={'code_sha256':a.sha(__file__),'config_sha256':a.sha(CONFIG),
                               'proposal_identity':json.loads((p.OUT/'identity.json').read_text()),
                               'manual_overrides_digest':hashlib.sha256(json.dumps(manual,sort_keys=True).encode()).hexdigest()}
    if args.mode=='full':
        pilot=json.loads((a.ROOT/'clean_pilot_v1/completion.json').read_text());assert pilot['all_processed_arrays_verified']
        identity['pilot_completion_sha256']=a.sha(a.ROOT/'clean_pilot_v1/completion.json')
        visual=json.loads((a.ROOT/'clean_pilot_v1/visual_review.json').read_text());assert visual['approved_to_run_same_policy']
        identity['pilot_visual_review_sha256']=a.sha(a.ROOT/'clean_pilot_v1/visual_review.json')
    identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    identity['identity']=identity_hash;frozen=root/'identity.json'
    if frozen.exists():assert args.resume and json.loads(frozen.read_text())==identity
    else:a.save(frozen,identity)
    assert not (root/'completion.json').exists(),'do not restart a completed cleaning'
    chosen=None
    if args.mode=='pilot':
        selection=root/'selection_private.json'
        if selection.exists():chosen=json.loads(selection.read_text())
        else:chosen=pilot_cases();a.save(selection,chosen)
        assert len(chosen)==64
    target=64 if chosen else 19264;done=set();counts=Counter();start=time.time()
    for file in (root/'cases').glob('*.json'):
        r=json.loads(file.read_text());assert r['identity']==identity_hash and a.sha(r['proposal_result_path'])==r['proposal_result_sha256']
        for e in r['entries']:
            assert a.sha(e['gray01_npy'])==e['gray01_sha256'] and a.sha(e['valid_mask_npy'])==e['valid_mask_sha256']
        done.add(r['case_id']);counts.update(counts_for(r['entries']))
    def status(phase):
        record={'phase':phase,'cases':len(done),'target_cases':target,**dict(counts),'elapsed_seconds':time.time()-start,'updated_unix':time.time()}
        a.save(root/'status.json',record);return record
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        while True:
            available=list(map(Path,chosen)) if chosen else sorted((p.OUT/'cases').glob('*/result.json'))
            tasks=[(str(path),stage_name,identity_hash,manual) for path in available if path.parent.name not in done]
            for r in pool.map(worker,tasks,chunksize=1):
                done.add(r.pop('case_id'));counts.update(r)
                if len(done)%25==0:status('cleaning_and_rescreening')
            if len(done)==target:
                if chosen or ((p.OUT/'completion.json').exists() and (a.ROOT/'outputs/raw_completion.json').exists()):
                    if not chosen:assert counts['knees']==38520 and counts['train_total']==34258 and counts['val_total']==4262
                    a.save(root/'completion.json',{**status('completed'),'identity':identity,'all_processed_arrays_verified':True,
                                                   'mask_outside_original_roi_for_all_changed':True,'unmasked_pixels_bit_exact_for_all':True,
                                                   'full_manual_review_completed':False,'clinical_quality_approved':False,
                                                   'final_manifests_published':False,'training_started':False})
                    break
            status('waiting_for_more_proposals');time.sleep(20)
    print(json.dumps(status('completed')),flush=True)


if __name__=='__main__':main()
