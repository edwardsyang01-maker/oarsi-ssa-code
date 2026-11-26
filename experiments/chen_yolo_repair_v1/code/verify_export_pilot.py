from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image,ImageDraw
import candidates as c
p=c.p


def safe_tile(roi,native_hw):
    a=np.load(roi['gray01_npy'],allow_pickle=False)[0].copy()
    t=roi['transform'];x0,y0,_,_=roi['bbox'];h,w=native_hw
    xx=x0+(np.arange(224)-t['offset_x']+.5)/t['scale_x']
    yy=y0+(np.arange(224)-t['offset_y']+.5)/t['scale_y']
    keep=(yy[:,None]>=.20*h)&(yy[:,None]<.88*h)&(xx[None,:]>=.03*w)&(xx[None,:]<.97*w)
    a[~keep]=0
    return Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8)).convert('RGB')


def main():
    state=json.loads((c.OUT/'pilot_status.json').read_text())
    assert state['phase']=='completed' and state['completed']==48 and state['errors']==0
    identity=c.identity();assert state['repair_identity']==identity['identity']
    auth=json.loads((p.OUT/'preview_transfer_authorization_20261111.json').read_text())
    assert auth['user_answer']=='joint-anomaly review and spot-check previews for this round are allowed'
    tasks=json.loads((c.OUT/'pilot_tasks_private.json').read_text());counts=defaultdict(Counter)
    export=c.ROOT/'preview_server_only/pilot_export';export.mkdir(parents=True,exist_ok=True)
    public=[];checked=0;no_voi_equal=0
    for task in sorted(tasks,key=lambda t:t['row']['prior_case_id']):
        row=task['row'];r=json.loads((c.DATA/'pilot'/row['case_id']/'result.json').read_text())
        assert r['status']=='candidates_generated' and r['repair_identity']==identity['identity']
        assert r['source_receipt']['archive_sha256']==row['archive_sha256']
        assert r['source_receipt']['native_pixel_sha256']==row['native_pixel_sha256']
        assert r['baseline_result_sha256']==p.sha(Path(task['baseline_path']))
        for variant,items in r['variants'].items():
            for x in items:
                counts[variant]['detected']+=1
                assert x['clinical_side']==p.SIDES[x['image_position']]
                assert x['kl']==row['labels'][x['clinical_side']]
                if x['status']!='saved':counts[variant]['extract_errors']+=1;continue
                for field in ['gray01','valid_mask']:assert p.sha(x[field+'_npy'])==x[field+'_sha256']
                a=np.load(x['gray01_npy'],allow_pickle=False);v=np.load(x['valid_mask_npy'],allow_pickle=False)
                assert a.shape==v.shape==(1,224,224) and a.dtype==np.float32 and v.dtype==bool
                assert v.any() and np.isfinite(a).all() and a.min()>=0 and a.max()<=1
                x0,y0,x1,y1=x['bbox'];h,w=r['native_hw'];assert 0<=x0<x1<=w and 0<=y0<y1<=h
                counts[variant]['saved']+=1;counts[variant]['flagged']+=bool(x['flags']);checked+=1
        if r['original_presentation']['voi_method']=='none':
            for original in r['variants']['original']:
                if original['status']!='saved':continue
                changed=next(x for x in r['variants']['modality_samebox'] if x['image_position']==original['image_position'])
                assert changed['gray01_sha256']==original['gray01_sha256']
                no_voi_equal+=1
        canvas=Image.new('RGB',(1392,520),'#101820');draw=ImageDraw.Draw(canvas)
        draw.text((6,4),row['prior_case_id']+' | '+row['case_id']+' | joint-only comparisons; perimeter redacted',fill='white')
        names=['BASE','RAW same box','RAW ROI norm','RAW re-detect','FOV original','FOV raw']
        for yi,pos in enumerate(p.SIDES):
            y=24+248*yi
            for xi,variant in enumerate(c.POLICY['variants']):
                x=232*xi;draw.text((x+3,y),names[xi]+' '+p.SIDES[pos],fill='#73dbe9')
                roi=next((item for item in r['variants'][variant] if item['image_position']==pos and item['status']=='saved'),None)
                if roi:canvas.paste(safe_tile(roi,r['native_hw']),(x+2,y+18))
                else:draw.text((x+18,y+100),'MISSING',fill='#ffbf64')
        name=row['prior_case_id']+'_'+row['case_id']+'.png';canvas.save(export/name)
        public.append({'case_id':row['case_id'],'pilot_code':row['prior_case_id'],'preview':name,'sha256':p.sha(export/name),
                       'variants':{v:[{'image_position':x['image_position'],'status':x['status'],'flags':x['flags']} for x in items]
                                   for v,items in r['variants'].items()}})
    p.save(export/'public_index.json',public)
    for start in range(0,len(public),4):
        sheet=Image.new('RGB',(1392,2080),'#101820')
        for j,item in enumerate(public[start:start+4]):
            with Image.open(export/item['preview']) as im:sheet.paste(im,(0,j*520))
        sheet.save(export/f'contact_{start//4+1:02d}.png')
    bundle=c.ROOT/'preview_server_only/pilot_joint_previews.tar.gz'
    with tarfile.open(bundle,'w:gz') as archive:
        for path in sorted(export.iterdir()):archive.add(path,arcname=path.name,recursive=False)
    receipt={'repair_identity':identity['identity'],'created_utc':datetime.now(timezone.utc).isoformat(),
             'cases_checked':48,'roi_arrays_checked':checked,'no_voi_samebox_byte_identical_rois':no_voi_equal,
             'variant_counts':{k:dict(v) for k,v in counts.items()},'clinical_quality_approved':False,
             'anatomical_visual_review_completed':False,'source_hashes_verified':True,
             'position_side_labels_verified':True,'model_retraining':False,
             'preview_scope':'joint-only training pilot48; peripheral source areas redacted',
             'contains_patient_identity_or_kl_labels':False,'original_dicom_included':False,
             'preview_files':len(list(export.glob('*.png'))),'bundle_sha256':p.sha(bundle),'bundle_bytes':bundle.stat().st_size,
             'export_code_sha256':p.sha(Path(__file__))}
    p.save(c.OUT/'pilot_technical_verification.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
