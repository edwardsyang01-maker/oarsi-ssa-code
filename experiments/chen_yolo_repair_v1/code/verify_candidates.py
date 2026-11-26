import argparse
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np
import candidates as c
import extra_candidates as e
p=c.p


def roi_check(x,r,row):
    assert x['clinical_side']==p.SIDES[x['image_position']]
    assert x['kl']==row['labels'][x['clinical_side']]
    if x['status']!='saved':return False
    for f in ['gray01','valid_mask']:assert p.sha(x[f+'_npy'])==x[f+'_sha256']
    a=np.load(x['gray01_npy'],allow_pickle=False);v=np.load(x['valid_mask_npy'],allow_pickle=False)
    assert a.shape==v.shape==(1,224,224) and a.dtype==np.float32 and v.dtype==bool
    assert v.any() and np.isfinite(a).all() and 0<=float(a.min())<=float(a.max())<=1
    x0,y0,x1,y1=x['bbox'];h,w=r['native_hw'];assert 0<=x0<x1<=w and 0<=y0<y1<=h
    assert ('image_left' if (x0+x1)/2<w/2 else 'image_right')==x['image_position']
    t=x['transform'];assert t['image_size']==224
    assert t['source_crop_size_wh']==[x1-x0,y1-y0]
    assert math.isclose(t['scale_x'],t['resized_width']/(x1-x0),abs_tol=1e-12)
    assert math.isclose(t['scale_y'],t['resized_height']/(y1-y0),abs_tol=1e-12)
    if 'local_input_origin_xy' in t:
        assert t['local_input_origin_xy']==[x0,y0]
        assert t['source_bbox_xyxy_exclusive']==[0,0,x1-x0,y1-y0]
    else:assert t['source_bbox_xyxy_exclusive']==x['bbox']
    assert 0<=t['offset_x']<=224-t['resized_width'] and 0<=t['offset_y']<=224-t['resized_height']
    return True


def mapping_check(r,variant):
    evidence=r.get(variant+'_detection')
    if evidence is None:return 0
    t=evidence['transform'];origin=evidence.get('source_field_origin_xy',[0,0])
    if variant.startswith('lower_'):
        assert origin==[0,round(r['native_hw'][0]*.45)]
        assert t['native_hw']==[r['native_hw'][0]-origin[1],r['native_hw'][1]]
    checked=0
    for item in r['variants'][variant]:
        if item['status']!='saved':continue
        if t.get('mode')!='full_fov_physical_letterbox':
            box,raw=p.backmap(item['detector_bbox'],t,1.3)
        else:
            x0,y0,x1,y1=item['detector_bbox'];cx,cy=(x0+x1)/2,(y0+y1)/2
            px,py=t['pad_xy'];sx,sy=t['scale_xy'];hh=(y1-y0)*1.3/2;hw=(x1-x0)*1.3/2
            raw=[(cx-hw-px)/sx,(cy-hh-py)/sy,(cx+hw-px)/sx,(cy+hh-py)/sy]
            h,w=t['native_hw'];box=[max(0,min(w,math.floor(raw[0]))),max(0,min(h,math.floor(raw[1]))),max(0,min(w,math.ceil(raw[2]))),max(0,min(h,math.ceil(raw[3])))]
        for i in [0,2]:box[i]+=origin[0];raw[i]+=origin[0]
        for i in [1,3]:box[i]+=origin[1];raw[i]+=origin[1]
        assert box==item['bbox'] and np.allclose(raw,item['unclipped_bbox'],rtol=0,atol=1e-8)
        assert any(item['detector_bbox']==det['bbox'] and abs(item['score']-det['score'])<1e-10 for det in evidence['raw_detections'])
        checked+=1
    return checked


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);a=ap.parse_args();assert '/' not in a.name
    ident=c.identity();extra=e.identity();rows={r['case_id']:r for r in json.loads((p.OUT/'selection_private.json').read_text())}
    counts=Counter();no_voi=0;pilot_equal=0;mapping=0;files={}
    for family in ['full','extra']:
        for path in sorted((c.DATA/family).glob('*/result.json')):
            r=json.loads(path.read_text());case=r['case_id'];row=rows[case]
            assert r['status']=='candidates_generated' and r['repair_identity']==ident['identity']
            assert row['split'] in ['train','val'] and not r['test_accessed']
            for field,value in row.items():assert r[field]==value,(case,field)
            assert r['baseline_result_sha256']==p.sha(p.DATA/case/'result.json')
            for f in ['archive_sha256','native_pixel_sha256']:assert r['source_receipt'][f]==row[f]
            base=json.loads((p.DATA/case/'result.json').read_text());assert r['native_hw']==base['native_hw']
            if family=='full':assert r['variants']['original']==base['yolo']
            else:
                assert r['extra_identity']==extra['identity'] and r['parent_repair_result_sha256']==p.sha(c.DATA/'full'/case/'result.json')
            for variant,xs in r['variants'].items():
                assert len({x['image_position'] for x in xs})==len(xs)
                for x in xs:counts[family+'_saved_rois']+=roi_check(x,r,row)
                mapping+=mapping_check(r,variant)
            if family=='full' and r['original_presentation']['voi_method']=='none':
                for x in r['variants']['original']:
                    if x['status']=='saved':
                        y=next(y for y in r['variants']['modality_samebox'] if y['image_position']==x['image_position'])
                        assert x['gray01_sha256']==y['gray01_sha256'];no_voi+=1
            pilot=c.DATA/'pilot'/case/'result.json'
            if family=='full' and pilot.exists():
                previous=json.loads(pilot.read_text())
                for variant,xs in previous['variants'].items():
                    current=r['variants'][variant];assert len(xs)==len(current)
                    for x,y in zip(xs,current):
                        for f in ['status','bbox','clinical_side','kl']:assert x[f]==y[f]
                        if x['status']=='saved':
                            for f in ['gray01_sha256','valid_mask_sha256']:assert x[f]==y[f]
                            pilot_equal+=1
            counts[family+'_cases']+=1;files[family+'/'+case]=p.sha(path)
    report={'phase':'verified','repair_identity':ident['identity'],'extra_identity':extra['identity'],
        'counts':dict(counts),'mapping_rois_checked':mapping,'no_voi_samebox_equal_rois':no_voi,
        'pilot_full_equal_rois':pilot_equal,'source_result_hashes_digest':p.digest(files),
        'source_labels_and_position_verified':True,'clinical_quality_approved':False,
        'code_sha256':p.sha(Path(__file__)),'test_accessed':False,'training':False}
    dest=c.OUT/(a.name+'.json');assert not dest.exists();p.save(dest,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
