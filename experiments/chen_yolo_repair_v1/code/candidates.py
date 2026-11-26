from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
BASE = PROJECT/'experiments/chen_yolo_fullcrop_v1'
PRIOR = PROJECT/'experiments/chen_yolo_roi_v2'
DATA = PROJECT/'data/processed/chen_yolo_repair_v1'
OUT = ROOT/'outputs'
sys.path.insert(0,str(BASE/'code'))
import pipeline as p

POLICY = {
    'schema':'chen-author-yolo-repair-candidates-v1',
    'author_weights_sha256':p.MODEL_SHA,'detector_score':.12,'nms_iou':.3,'expansion':1.3,
    'flip':False,'side_policy':{'image_left':'R','image_right':'L'},
    'original_gray':'frozen presentation + wholefilm p0.5/p99.5',
    'modality_gray':'same modality transform and display polarity, omit VOI/window, wholefilm p0.5/p99.5',
    'roi_gray':'same modality values/polarity, original bbox p0.5/p99.5, only a review candidate',
    'variants':['original','modality_samebox','modality_roi_samebox','center_modality','full_original','full_modality'],
    'selection_uses_kl_or_classifier_predictions':False,'training':False,
}


def identity():
    base_id=p.verify_identity()
    content={'policy':POLICY,'base_run_id':base_id,'code_sha256':p.sha(Path(__file__))}
    content['identity']=p.digest(content)
    return content


def initialize():
    p.initializer('repair-candidates-only')


def modality_gray(pixels,ds,valid,presentation):
    from drai.data.dicom_gray import apply_modality_lut,normalize_roi
    values=np.asarray(apply_modality_lut(pixels,ds),dtype=np.float64)
    if not np.isfinite(values).all():raise ValueError('Nonfinite modality values')
    if presentation['presentation_inverse_applied']:
        values=values[valid].min()+values[valid].max()-values
    gray,norm=normalize_roi(values,valid)
    return gray,values,{'policy':'modality_no_voi_global_p0.5_p99.5','normalization':norm,
                       'original_voi_method':presentation['voi_method'],
                       'presentation_inverse_applied':presentation['presentation_inverse_applied'],
                       'source_values_min':float(values[valid].min()),'source_values_max':float(values[valid].max())}


def full_fov_input(gray,ds):
    spacing=None
    for key in ['PixelSpacing','ImagerPixelSpacing']:
        value=getattr(ds,key,None)
        if value is not None and len(value)==2:
            v=list(map(float,value))
            if all(math.isfinite(x) and x>0 for x in v):spacing=v;break
    if spacing is None:raise ValueError('Missing reliable spacing')
    h,w=gray.shape
    ratio=min(320/(w*spacing[1]),256/(h*spacing[0]))
    nw=max(1,min(320,round(w*spacing[1]*ratio)))
    nh=max(1,min(256,round(h*spacing[0]*ratio)))
    px,py=(320-nw)//2,(256-nh)//2
    resized=Image.fromarray(np.rint(gray*255).astype(np.uint8)).resize((nw,nh),Image.Resampling.BILINEAR)
    image=np.zeros((256,320),dtype=np.uint8);image[py:py+nh,px:px+nw]=np.asarray(resized)
    return image,{'mode':'full_fov_physical_letterbox','native_hw':[h,w],
                  'scale_xy':[nw/w,nh/h],'pad_xy':[px,py],'content_xyxy':[px,py,px+nw,py+nh],
                  'spacing_mm':spacing,'flip':False}


def mapped_candidates(gray,ds,mode):
    import torch
    from chen_yolo_adapter import decode
    if mode=='center':small,transform=p.author_input(gray,ds)
    else:small,transform=full_fov_input(gray,ds)
    with torch.inference_mode():
        detections=decode(p.MODEL(p.tensor_from_u8(small)),threshold=.12,nms_threshold=.3)
    h,w=gray.shape;all_boxes=[]
    for detection in detections:
        if mode=='center':box,raw=p.backmap(detection['bbox'],transform,1.3)
        else:
            x0,y0,x1,y1=map(float,detection['bbox']);cx,cy=(x0+x1)/2,(y0+y1)/2
            px,py=transform['pad_xy'];sx,sy=transform['scale_xy'];bounds=transform['content_xyxy']
            if not (bounds[0]<=cx<bounds[2] and bounds[1]<=cy<bounds[3]):continue
            hw,hh=(x1-x0)*1.3/2,(y1-y0)*1.3/2
            raw=[(cx-hw-px)/sx,(cy-hh-py)/sy,(cx+hw-px)/sx,(cy+hh-py)/sy]
            box=[max(0,min(w,math.floor(raw[0]))),max(0,min(h,math.floor(raw[1]))),
                 max(0,min(w,math.ceil(raw[2]))),max(0,min(h,math.ceil(raw[3])))]
        if box[2]<=box[0] or box[3]<=box[1]:continue
        pos='image_left' if (box[0]+box[2])/2<w/2 else 'image_right'
        flags=[]
        if any(abs(a-b)>1.01 for a,b in zip(box,raw)):flags.append('expanded_box_clipped')
        if box[0]<w/2<box[2]:flags.append('crosses_image_midline')
        if mode=='center' and transform['padding_needed']:flags.append('detector_center_padding')
        if any(v<=0 for v in detection['bbox'][:2]) or detection['bbox'][2]>=319 or detection['bbox'][3]>=255:
            flags.append('detector_frame_edge')
        all_boxes.append({'image_position':pos,'bbox':box,'unclipped_bbox':raw,
                          'detector_bbox':detection['bbox'],'score':detection['score'],'flags':flags})
    selected=[]
    for pos in p.SIDES:
        boxes=[x for x in all_boxes if x['image_position']==pos]
        if boxes:
            item=dict(max(boxes,key=lambda x:x['score']));item['flags']=list(item['flags'])
            if len(boxes)>1:item['flags'].append('multiple_detections_same_side')
            selected.append(item)
    if len(selected)==2 and p.overlap(selected[0]['bbox'],selected[1]['bbox']):
        for x in selected:x['flags'].append('bilateral_boxes_overlap')
    return selected,{'transform':transform,'raw_detections':detections,'all_mapped':all_boxes}


def original_box_item(roi):
    keep={k:roi[k] for k in ['image_position','bbox','unclipped_bbox','detector_bbox','score'] if k in roi}
    gray_flags={'high_saturation_fraction','low_saturation_fraction','low_gray_dynamic_range','low_valid_fraction'}
    keep['flags']=[f for f in roi['flags'] if f not in gray_flags]
    return keep


def roi_normalized_variant(values,valid,item,directory,labels):
    from drai.data.dicom_gray import normalize_roi
    from downstream.oa_kl_unified_v1.geometry import extract_regions
    x0,y0,x1,y1=item['bbox'];local_valid=valid[y0:y1,x0:x1]
    local,norm=normalize_roi(values[y0:y1,x0:x1],local_valid)
    shifted={**item,'bbox':[0,0,local.shape[1],local.shape[0]]}
    output=p.write_roi(local,local_valid,shifted,directory,'modality_roi_samebox',labels)
    output['bbox']=item['bbox'];output['normalization']=norm
    output['transform']['local_input_origin_xy']=[x0,y0]
    output['normalization_source']='original modality values within native bbox'
    return output


def preview_panel(result,directory,preview_dir):
    width=6*232;canvas=Image.new('RGB',(width,520),'#101820');draw=ImageDraw.Draw(canvas)
    labels=['BASE','RAW same box','RAW ROI norm','RAW re-detect','FULL original','FULL raw']
    draw.text((8,4),result['case_id']+' | joint-only repair comparison | no KL labels',fill='white')
    for row,pos in enumerate(p.SIDES):
        y=24+row*248
        for col,variant in enumerate(POLICY['variants']):
            x=col*232;draw.text((x+3,y),labels[col]+' '+p.SIDES[pos],fill='#73dbe9')
            item=next((r for r in result['variants'].get(variant,[]) if r['image_position']==pos),None)
            if item is None or item['status']!='saved':
                draw.text((x+20,y+100),'MISSING',fill='#ffbf64');continue
            a=np.load(item['gray01_npy'],allow_pickle=False)[0]
            tile=Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8)).convert('RGB')
            canvas.paste(tile,(x+2,y+18))
    preview_dir.mkdir(parents=True,exist_ok=True)
    canvas.save(preview_dir/(result['case_id']+'.png'))


def process_case(task):
    from drai.data.dicom_gray import read_pa_archive,presentation_values,normalize_roi
    row=task['row'];phase=task['phase'];directory=DATA/phase/row['case_id'];directory.mkdir(parents=True,exist_ok=True)
    baseline_path=Path(task['baseline_path']);baseline=json.loads(baseline_path.read_text())
    result={**row,'repair_identity':task['identity'],'phase':phase,'baseline_result_sha256':p.sha(baseline_path),
            'variants':{'original':baseline['yolo']},'selected_for_training':False,
            'clinical_quality_approved':False,'test_accessed':False}
    started=time.monotonic()
    try:
        assert row['split'] in ['train','val'] and row['film_id']==baseline['film_id']
        ds,pixels,receipt=read_pa_archive(p.NAS/row['input_archive_relpath'],expected_bytes=int(row['archive_bytes']),source_root=p.NAS)
        if receipt['archive_sha256']!=row['archive_sha256'] or receipt['native_pixel_sha256']!=row['native_pixel_sha256']:
            raise RuntimeError('Source hash mismatch')
        values,valid,presentation=presentation_values(pixels,ds)
        original,normalization=normalize_roi(values,valid);del values
        alternative,raw_values,raw_receipt=modality_gray(pixels,ds,valid,presentation);del pixels
        result.update(source_receipt=receipt,native_hw=list(original.shape),original_presentation=presentation,
                      original_normalization=normalization,modality_gray_receipt=raw_receipt)
        result['variants']['modality_samebox']=[];result['variants']['modality_roi_samebox']=[]
        for old in baseline['yolo']:
            if old['status']!='saved':continue
            item=original_box_item(old)
            result['variants']['modality_samebox'].append(p.write_roi(alternative,valid,item,directory,'modality_samebox',row['labels']))
            result['variants']['modality_roi_samebox'].append(roi_normalized_variant(raw_values,valid,item,directory,row['labels']))
        for variant,gray,mode in [('center_modality',alternative,'center'),('full_original',original,'full'),('full_modality',alternative,'full')]:
            boxes,evidence=mapped_candidates(gray,ds,mode)
            result['variants'][variant]=[p.write_roi(gray,valid,item,directory,variant,row['labels']) for item in boxes]
            result[variant+'_detection']=evidence
        result['status']='candidates_generated'
        preview_panel(result,directory,ROOT/'preview_server_only'/phase)
    except Exception as exc:
        result.update(status='case_error',error=str(exc),traceback=traceback.format_exc())
    result['seconds']=time.monotonic()-started
    p.save(directory/'result.json',result)
    return {'case_id':row['case_id'],'status':result['status'],'seconds':result['seconds'],
            'counts':{k:sum(x['status']=='saved' for x in v) for k,v in result['variants'].items()}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['pilot','full']);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();ident=identity();OUT.mkdir(parents=True,exist_ok=True)
    identity_path=OUT/(args.phase+'_identity.json')
    if identity_path.exists() and json.loads(identity_path.read_text())!=ident:raise RuntimeError('Repair code changed; use a new revision')
    p.save(identity_path,ident)
    rows=json.loads((p.OUT/'selection_private.json').read_text())
    tasks=[]
    if args.phase=='pilot':
        for row in rows:
            if row.get('prior_case_id','').startswith('C'):
                tasks.append({'row':row,'phase':'pilot','identity':ident['identity'],
                              'baseline_path':str(PROJECT/'data/processed/chen_yolo_roi_v2'/row['prior_case_id']/'result.json')})
        assert len(tasks)==48 and all(t['row']['split']=='train' for t in tasks)
    else:
        selected=json.loads((OUT/'full_selection_private.json').read_text())
        chosen=set(selected['case_ids'])
        if selected['repair_identity']!=ident['identity']:raise RuntimeError('Selection identity mismatch')
        for row in rows:
            if row['case_id'] in chosen:
                tasks.append({'row':row,'phase':'full','identity':ident['identity'],
                              'baseline_path':str(p.DATA/row['case_id']/'result.json')})
        assert len(tasks)==len(chosen)
    if any((DATA/args.phase/t['row']['case_id']/'result.json').exists() for t in tasks):
        raise RuntimeError('Existing repair results require explicit reviewed resume; do not overwrite')
    p.save(OUT/(args.phase+'_tasks_private.json'),tasks)
    started=time.time();done=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        futures=[pool.submit(process_case,t) for t in tasks]
        for future in as_completed(futures):
            record=future.result();done.append(record)
            p.save(OUT/(args.phase+'_status.json'),{'phase':'running','completed':len(done),'total':len(tasks),'started_unix':started,'last':record})
            print(json.dumps(record),flush=True)
    p.save(OUT/(args.phase+'_status.json'),{'phase':'completed','completed':len(done),'total':len(tasks),'elapsed_seconds':time.time()-started,
                                          'errors':sum(r['status']=='case_error' for r in done),'repair_identity':ident['identity']})
    p.save(OUT/(args.phase+'_result_index_private.json'),done)


if __name__=='__main__':main()
