from pathlib import Path
import json,warnings
import cv2
import numpy as np
import audit_rulers as a
import ruler_refinement as ref
from clean_policy import bar_inner_width

OUT=a.ROOT/'proposals_v1'


def confidence(line,item,entries):
    context=ref.pair_context(line,item,entries)
    points=line['points'];peaks=np.array([p['peak'] for p in points]);contrasts=np.array([p['contrast'] for p in points])
    strong=(peaks>=75)&(contrasts>=30)
    relative=float(np.median(contrasts/np.maximum(peaks,1)))
    gaps=np.diff([p['y'] for p in points]);step=line['step']
    occupied=(len(points)-1)/max(round(line['span']/step),1)
    regular=len(points)>=3 and line['span']>=115 and line['residual']<=3.5
    trusted=regular and int(strong.sum())>=3 and relative>=.45 and occupied>=.50 and context['central_between_knees']
    if len(points)==3:trusted=trusted and (bool(np.all(np.abs(gaps/step-1)<=.12)) or float(np.median(peaks))>=180)
    dark_regular=regular and len(points)>=4 and relative>=.80 and occupied>=.60 and 55<=step<=115 and context['central_between_knees']
    t=item['transform'];left=t['offset_x'];right=left+t['resized_width']-1
    bright_partial=regular and int(strong.sum())>=3 and float(np.median(peaks))>=100 and relative>=.60 and occupied>=.25 and context['central_between_knees'] and (max(p['x'] for p in points)<left+16 or min(p['x'] for p in points)>right-16)
    trusted=trusted or dark_regular or bright_partial
    return {'high_confidence':bool(trusted),'dark_regular_channel':bool(dark_regular),'bright_partial_channel':bool(bright_partial),'strong_points':int(strong.sum()),'relative_contrast':relative,
            'period_occupancy':float(occupied),'pair_context':context}


def fitted_candidates(item,initial,entries,gray,valid):
    lines=ref.detect(gray,valid,item)+initial['lines'];out=[]
    for line in lines:
        rec={**line,'evidence':confidence(line,item,entries)}
        if not any(abs(rec['x_mid']-q['x_mid'])<1 and rec['count']==q['count'] and abs(rec['span']-q['span'])<1 for q in out):out.append(rec)
    return out


def evaluate(source,initial):
    items=source['entries'];arrays={};candidates={};seeds=[]
    for item in items:
        g=np.load(item['gray01_npy'],allow_pickle=False)[0];v=np.load(item['valid_mask_npy'],allow_pickle=False)[0]
        arrays[item['key']]=(g,v);ini=next(x for x in initial['entries'] if x['key']==item['key'])
        lines=fitted_candidates(item,ini,items,g,v);candidates[item['key']]=lines
        for line in lines:
            if not line['evidence']['high_confidence']:continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning);width,receipt=bar_inner_width(line,g,v)
            seeds.append((item,{**line,'inner_width':max(14.,width),'bar_edge':receipt}))
    yy,xx=np.mgrid[:448,:448];results=[]
    for item in items:
        g,v=arrays[item['key']];lines=[]
        for source_item,seed in seeds:
            if source_item['key']==item['key']:line={**seed,'projected_from_same_film':False}
            else:
                assert source_item['case_id']==item['case_id']
                line=ref.project(seed,source_item,item);line['inner_width']=seed['inner_width']*item['transform']['scale_x']/source_item['transform']['scale_x']+2
            endpoints=[line['intercept'],line['intercept']+447*line['slope']]
            if max(endpoints)+line['inner_width']<0 or min(endpoints)-line['inner_width']>447:continue
            near=next((j for j,q in enumerate(lines) if abs(q['x_mid']-line['x_mid'])<8),None)
            score=(line['count'],line['span'],-line['residual'])
            if near is None:lines.append(line)
            elif score>(lines[near]['count'],lines[near]['span'],-lines[near]['residual']):lines[near]=line
        mask=np.zeros((448,448),bool)
        for line in lines:
            center=line['intercept']+line['slope']*yy
            # This ROI keeps the original left/right orientation; the ruler lies on the side between the two knees.
            if item['clinical_side']=='L':mask|=xx<=center+line['inner_width']
            else:mask|=xx>=center-line['inner_width']
        mask&=v
        parent=a.detector_box(item,1.3);core=a.detector_box(item,1.05)
        parent_area=(xx>=parent[0])&(xx<=parent[2])&(yy>=parent[1])&(yy<=parent[3])
        core_area=(xx>=core[0])&(xx<=core[2])&(yy>=core[1])&(yy<=core[3])
        metrics={'mask_pixels':int(mask.sum()),'mask_fraction':float(mask.mean()),
                 'parent_1p3_overlap_pixels':int((mask&parent_area).sum()),'core_overlap_pixels':int((mask&core_area).sum())}
        risks=[]
        if metrics['parent_1p3_overlap_pixels']:risks.append('mask_touches_original_joint_roi')
        if metrics['core_overlap_pixels']:risks.append('mask_touches_detector_core')
        if metrics['mask_fraction']>.25:risks.append('large_mask')
        unresolved=[ln for ln in candidates[item['key']] if ln['evidence']['pair_context']['central_between_knees'] and
                    ln['evidence']['strong_points']>=2 and not ln['evidence']['high_confidence']]
        unresolved=[ln for ln in unresolved if not any(abs(ln['x_mid']-q['x_mid'])<=max(q['inner_width'],12) for q in lines)]
        if unresolved:risks.append('unresolved_marker_candidate')
        results.append({'key':item['key'],'split':item['split'],'side':item['clinical_side'],'lines':lines,
                        'candidate_lines':candidates[item['key']],'unresolved_lines':unresolved,'metrics':metrics,
                        'review_flags':risks,'mask':mask,'ruler_cleaning_applied':False})
    return results


def worker(path):
    cv2.setNumThreads(1);path=Path(path);source=json.loads(path.read_text())
    audit_path=a.ROOT/'audit_v2/cases'/(source['case_id']+'.json');initial=json.loads(audit_path.read_text())
    assert initial['source_result_sha256']==a.sha(path)
    entries=evaluate(source,initial);directory=OUT/'cases'/source['case_id'];directory.mkdir(parents=True,exist_ok=True)
    for entry in entries:
        mask=entry.pop('mask');file=directory/(entry['side']+'_proposed_mask.npy');tmp=file.with_suffix('.tmp.npy')
        np.save(tmp,mask[None],allow_pickle=False);tmp.replace(file)
        entry['proposed_mask_npy']=str(file);entry['proposed_mask_sha256']=a.sha(file)
    result={'case_id':source['case_id'],'source_result_path':str(path),'source_result_sha256':a.sha(path),
            'audit_sha256':a.sha(audit_path),'entries':entries}
    a.save(directory/'result.json',result)
    return {'case_id':source['case_id'],'knees':len(entries),'proposed_clean':sum(e['metrics']['mask_pixels']>0 for e in entries),
            'review':sum(bool(e['review_flags']) for e in entries),'safe_proposed':sum(e['metrics']['mask_pixels']>0 and not e['review_flags'] for e in entries)}
