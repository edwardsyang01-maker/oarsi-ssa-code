import argparse,fcntl,hashlib,json,multiprocessing as mp,time,warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2,numpy as np
import audit_rulers as a
import ruler_refinement as ref
import proposal_v1 as p
import clean_stage as c
from clean_policy import bar_inner_width


def extra_lines(item,case,proposal,gray,valid):
    points=[x for x in ref.boundary_points(gray,valid) if x['peak']>=5 and x['contrast']>=5 and x['contrast']/max(x['peak'],1)>=.60]
    direction=1 if item['clinical_side']=='L' else -1
    candidates=[];single=[];seen=[]
    for entry in proposal['entries']:
        source_item=next(x for x in case['entries'] if x['key']==entry['key'])
        for original in entry['lines']:
            if original['count']<4:continue
            line=original if source_item['key']==item['key'] else ref.project(original,source_item,item)
            if any(abs(line['x_mid']-x)<8 for x in seen):continue
            seen.append(line['x_mid']);step=line['step']
            if not 35<=step<=150:continue
            relevant=[]
            for point in points:
                distance=direction*(point['x']-line['intercept']-line['slope']*point['y'])
                if -.15*step<=distance<=.90*step:relevant.append((point,distance))
            for k,(point,distance) in enumerate(relevant):
                for other,other_distance in relevant[k+1:]:
                    dy=abs(other['y']-point['y'])
                    if not (35<=dy<=3.15*step and abs(distance-other_distance)<=3.5):continue
                    periods=dy/step
                    if abs(periods-round(periods))>.12 or max(point['peak'],other['peak'])<40:continue
                    row=[r for r,d in relevant if abs(d-(distance+other_distance)/2)<=3.5 and abs((r['y']-point['y'])/step-round((r['y']-point['y'])/step))<=.12]
                    row.sort(key=lambda x:x['y'])
                    intercept=float(np.median([q['x']-line['slope']*q['y'] for q in row]))
                    result={'slope':line['slope'],'intercept':intercept,'x_mid':intercept+224*line['slope'],
                            'points':row,'count':len(row),'span':row[-1]['y']-row[0]['y'],'step':step,
                            'source_full_column_count':original['count'],'source_side':source_item['clinical_side'],
                            'method':'same_film_period_matched_partial_column'}
                    if not any(abs(result['x_mid']-q['x_mid'])<8 for q in candidates):candidates.append(result)
            left=item['transform']['offset_x'];right=left+item['transform']['resized_width']-1
            for point,distance in relevant:
                if point['peak']>=100 and (point['x']<left+3 or point['x']>right-3):
                    if not any(abs(point['x']-q['intercept']-q['slope']*point['y'])<=10 for q in candidates):single.append(point)
    yy,xx=np.mgrid[:448,:448];mask=np.zeros((448,448),bool)
    for line in candidates:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);width,receipt=bar_inner_width(line,gray,valid)
        line['inner_width']=max(14.,width);line['bar_edge']=receipt;center=line['intercept']+line['slope']*yy
        if item['clinical_side']=='L':mask|=xx<=center+line['inner_width']
        else:mask|=xx>=center-line['inner_width']
    mask&=valid
    return mask,candidates,single


def worker(task):
    file,stage,identity=task;file=Path(file);cv2.setNumThreads(1)
    case=json.loads(file.read_text());proposal=json.loads(Path(case['proposal_result_path']).read_text())
    assert a.sha(case['proposal_result_path'])==case['proposal_result_sha256'];out=[]
    for item in case['entries']:
        record={**item,'baseline_cleanup_status':item['cleanup_status'],'edge_audit_completed':True,
                'edge_finish_identity':identity,'edge_extra_mask_pixels':0,'edge_candidates':[],'edge_single_fragments':[],
                'baseline_gray01_npy':item['gray01_npy'],'baseline_gray01_sha256':item['gray01_sha256']}
        if not item['engineering_input_ready']:
            out.append(record);continue
        assert a.sha(item['gray01_npy'])==item['gray01_sha256'] and a.sha(item['valid_mask_npy'])==item['valid_mask_sha256']
        g=np.load(item['gray01_npy'],allow_pickle=False);v=np.load(item['valid_mask_npy'],allow_pickle=False)
        mask,lines,single=extra_lines(item,case,proposal,g[0],v[0]);record['edge_candidates']=lines;record['edge_single_fragments']=single
        record['edge_extra_mask_pixels']=int(mask.sum())
        if c.protection(mask,item)>0 or single:
            record['engineering_input_ready']=False;record['cleanup_status']='held_edge_candidate'
            record['cleanup_review_reasons']=list(item['cleanup_review_reasons'])+(['edge_mask_touches_original_joint_roi'] if c.protection(mask,item)>0 else [])+(['edge_single_fragment_uncertain'] if single else [])
        elif mask.any():
            full_mask=mask[None].copy()
            if item['cleanup_mask_npy']:full_mask|=np.load(item['cleanup_mask_npy'],allow_pickle=False)
            original=np.load(item['raw_gray01_npy'],allow_pickle=False);original_valid=np.load(item['raw_valid_mask_npy'],allow_pickle=False)
            assert c.protection(full_mask[0],item)==0
            clean=original.copy();clean[full_mask]=0;clean_valid=original_valid&~full_mask
            assert np.array_equal(clean[~full_mask],original[~full_mask]) and (clean[~clean_valid]==0).all()
            folder=c.DATA/stage/case['case_id'];side=item['clinical_side']
            record['gray01_npy'],record['gray01_sha256']=c.atomic_array(folder/(side+'_gray01.npy'),clean)
            record['valid_mask_npy'],record['valid_mask_sha256']=c.atomic_array(folder/(side+'_valid.npy'),clean_valid)
            record['cleanup_mask_npy'],record['cleanup_mask_sha256']=c.atomic_array(folder/(side+'_ruler_mask.npy'),full_mask)
            residual=[line for line in ref.detect(clean[0],clean_valid[0],item) if p.confidence(line,item,case['entries'])['high_confidence']]
            leftover,_,single_after=extra_lines(item,case,proposal,clean[0],clean_valid[0])
            record['postclean_high_confidence_rulers']=residual;record['ruler_removal_applied']=True
            record['proposed_mask_pixels']=int(full_mask.sum());record['cleanup_status']='accepted_cleaned'
            record['ruler_review']='cleaned_and_same_film_edge_rescreened'
            if residual or leftover.any() or single_after:
                record['engineering_input_ready']=False;record['cleanup_status']='held_edge_residual'
                record['cleanup_review_reasons']=list(item['cleanup_review_reasons'])+['edge_residual_after_repair']
        out.append(record)
    result={**case,'identity':identity,'baseline_cleanup_result_path':str(file),'baseline_cleanup_result_sha256':a.sha(file),'entries':out}
    a.save(a.ROOT/stage/'cases'/(case['case_id']+'.json'),result)
    counts=Counter(c.counts_for(out));counts['extra_mask_images']=sum(e['edge_extra_mask_pixels']>0 for e in out)
    counts['newly_held']=sum(e['baseline_cleanup_status'].startswith('accepted') and not e['engineering_input_ready'] for e in out)
    return {'case_id':case['case_id'],**dict(counts)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['pilot','full']);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    base='clean_pilot_v1' if args.mode=='pilot' else 'clean_v1';stage='edge_pilot_v1' if args.mode=='pilot' else 'edge_finish_v1'
    root=a.ROOT/stage;root.mkdir(parents=True,exist_ok=True);lock=(root/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity={'code_sha256':a.sha(__file__),'base_identity_sha256':a.sha(a.ROOT/base/'identity.json'),'clean_worker_sha256':a.sha(c.__file__),
              'proposal_identity_sha256':a.sha(p.OUT/'identity.json')}
    if args.mode=='full':
        review=a.ROOT/'edge_pilot_v1/visual_review.json';assert json.loads(review.read_text())['approved_for_full_supplement']
        identity['pilot_review_sha256']=a.sha(review)
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();identity['identity']=digest;frozen=root/'identity.json'
    if frozen.exists():assert args.resume and json.loads(frozen.read_text())==identity
    else:a.save(frozen,identity)
    assert not (root/'completion.json').exists(),'do not restart the completed edge review'
    done=set();counts=Counter();start=time.time();target=64 if args.mode=='pilot' else 19264
    for file in (root/'cases').glob('*.json'):
        r=json.loads(file.read_text());assert r['identity']==digest and a.sha(r['baseline_cleanup_result_path'])==r['baseline_cleanup_result_sha256']
        done.add(r['case_id']);counts.update(c.counts_for(r['entries']));counts['extra_mask_images']+=sum(e['edge_extra_mask_pixels']>0 for e in r['entries'])
        counts['newly_held']+=sum(e['baseline_cleanup_status'].startswith('accepted') and not e['engineering_input_ready'] for e in r['entries'])
    def status(phase):
        value={'phase':phase,'cases':len(done),'target_cases':target,**dict(counts),'elapsed_seconds':time.time()-start,'updated_unix':time.time()}
        a.save(root/'status.json',value);return value
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn')) as pool:
        while True:
            tasks=[(str(f),stage,digest) for f in sorted((a.ROOT/base/'cases').glob('*.json')) if f.stem not in done]
            for result in pool.map(worker,tasks,chunksize=1):
                done.add(result.pop('case_id'));counts.update(result)
                if len(done)%50==0:status('same_film_edge_review')
            if len(done)==target and (a.ROOT/base/'completion.json').exists():
                assert counts['knees']==(128 if args.mode=='pilot' else 38520)
                a.save(root/'completion.json',{**status('completed'),'identity':identity,'all_images_automatically_audited':True,
                                               'full_manual_review_completed':False,'clinical_quality_approved':False,
                                               'all_modified_pixels_verified':True,'final_manifests_published':False,'training_started':False})
                break
            status('waiting_for_more_cleaned_cases');time.sleep(20)
    print(json.dumps(status('completed')),flush=True)


if __name__=='__main__':main()
