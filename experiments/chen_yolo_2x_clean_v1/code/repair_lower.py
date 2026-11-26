from collections import Counter
import json
import math
from pathlib import Path
import time

import numpy as np

import crop_full as c

ROOT=c.ROOT/'repair_lower_v1'
DATA=c.PROJECT/'data/processed/chen_yolo_2x_clean_v1/raw_repaired_lower'
ORIGINAL_DIRECT_BOX=c.pilot.direct_box


def source_entry(item):
    source=Path(item['parent_candidate_result']);assert c.e.p.sha(source)==item['parent_candidate_result_sha256']
    r=json.loads(source.read_text())
    entry=c.e.saved(r,item['variant'],item['image_position'])
    assert entry['bbox']==item['bbox']
    return r,entry


def lower_geometry(item,native_hw):
    if not item['variant'].startswith('lower_'):
        return ORIGINAL_DIRECT_BOX(item,native_hw)
    r,entry=source_entry(item);h,w=native_hw
    origin=r[item['variant']+'_detection']['source_field_origin_xy']
    assert origin==[0,round(h*.45)] and r['native_hw']==list(native_hw)
    x0,y0,x1,y1=map(float,entry['unclipped_bbox'])
    old=[max(0,min(w,math.floor(x0))),max(origin[1],min(h,math.floor(y0))),
         max(0,min(w,math.ceil(x1))),max(origin[1],min(h,math.ceil(y1)))]
    assert old==item['bbox'],'the old lower-half window crop bounds cannot be reproduced exactly'
    cx,cy=(x0+x1)/2,(y0+y1)/2;dw,dh=(x1-x0)/1.3,(y1-y0)/1.3
    requested=[cx-dw,cy-dh,cx+dw,cy+dh]
    box=[max(0,min(w,math.floor(requested[0]))),max(0,min(h,math.floor(requested[1]))),
         max(0,min(w,math.ceil(requested[2]))),max(0,min(h,math.ceil(requested[3])))]
    assert box[0]<=old[0] and box[1]<=old[1] and box[2]>=old[2] and box[3]>=old[3]
    clipped=requested[0]<0 or requested[1]<0 or requested[2]>w or requested[3]>h
    return box,clipped,{'unclipped_1p3_bbox':entry['unclipped_bbox'],'requested_2p0_bbox':requested,
                       'detector_native_size_wh':[dw,dh],'parent_result_sha256':c.e.p.sha(item['parent_candidate_result']),
                       'old_detection_window_origin_xy':origin,'old_window_bbox_reproduced':True,
                       'new_crop_uses_full_native_field':True,'repair_policy':'lower_window_origin_verified_v1'}


def main():
    ROOT.mkdir(parents=True,exist_ok=True);assert not (ROOT/'completion.json').exists()
    identity={'base_identity':c.identity(),'repair_code_sha256':c.e.p.sha(__file__),
              'policy':'old box validation also considers the lower-half detection window; the 2x output still uses the full native film and is not cut again by that window'}
    identity['identity']=c.e.p.digest(identity);c.save(ROOT/'identity.json',identity)
    all_tasks=c.tasks();selected=[];lower_items=0
    for t in all_tasks:
        needed=False
        for item in t['items']:
            if not item['variant'].startswith('lower_'):continue
            lower_items+=1;r,e=source_entry(item);h,w=r['native_hw'];b=e['unclipped_bbox']
            normal=[max(0,min(w,math.floor(b[0]))),max(0,min(h,math.floor(b[1]))),
                    max(0,min(w,math.ceil(b[2]))),max(0,min(h,math.ceil(b[3])))]
            lower_geometry(item,r['native_hw'])
            if normal!=item['bbox']:needed=True
        if needed:selected.append(t)
    c.save(ROOT/'selection_private.json',selected)
    c.DATA=DATA;c.pilot.direct_box=lower_geometry;c.initialize();start=time.time()
    for t in selected:
        t['identity']=identity['identity'];r=c.process(t)
        assert r['status']=='completed','independent repair failed; technical trace kept'
        c.verify_result(t,identity['identity'])
    report={'phase':'completed','cases':len(selected),'knees':sum(len(t['items']) for t in selected),
            'all_lower_items_checked':lower_items,'identity':identity['identity'],
            'old448_exact_reproduction':True,'all_new_arrays_verified':True,
            'base_code_and_outputs_unchanged':True,'elapsed_seconds':time.time()-start}
    c.save(ROOT/'completion.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
