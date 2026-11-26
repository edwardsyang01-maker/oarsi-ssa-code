from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
OUT = ROOT / 'outputs'
ASSETS = ROOT / 'assets'
NAS = Path('/path/to/nas')
sys.path[:0] = [str(PROJECT), str(PROJECT/'src'), str(ROOT/'code/vendor')]
BASE = PROJECT/'outputs/downstream/oa_kl_unified_v1'
CACHE = PROJECT/'data/processed/oa_kl_unified_v1/native_all_20261025_r3'
TRAIN = PROJECT/'outputs/downstream/oa_kl_wholefilm_v2/preparation_20261025/wholefilm_train.csv'
POSITIONS = ['image_left', 'image_right']


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4<<20), b''): h.update(b)
    return h.hexdigest()


def rows(path):
    with Path(path).open() as f:
        for line in f: yield json.loads(line)


def select():
    target = OUT/'selection_private.json'
    if target.exists(): return json.loads(target.read_text())
    with TRAIN.open() as f: train = {r['sample_id']: r for r in csv.DictReader(f)}
    eligible = {r['film_id'] for r in rows(BASE/'label_candidates_20261025_r2/candidates.jsonl') if r['split']=='train'}
    excluded = {r['film_id'] for r in rows(BASE/'label_candidates_20261025_r2/excluded.jsonl') if r['split']=='train' and r['reason']=='unequal_kl_without_header_mapping'}
    counts = Counter(r['film_id'] for r in rows(BASE/'cohort_matrix_20261026_r1/roi_all.jsonl') if r['split']=='train')
    groups = [('old_two',24), ('old_one',8), ('old_zero',8), ('new_position',8)]
    chosen=[]; patients=set(); availability={}
    for group, number in groups:
        pool=[]
        for fid,r in train.items():
            belongs = fid in excluded if group=='new_position' else fid in eligible and counts[fid]=={'old_two':2,'old_one':1,'old_zero':0}[group]
            if belongs: pool.append(r)
        availability[group]=len(pool)
        cells=defaultdict(list)
        for r in pool: cells[(r['visit'],r['dicom_modality_first_instance'])].append(r)
        for cell in cells.values(): cell.sort(key=lambda r:hashlib.sha256(('chen-crop-pilot-v1|'+r['sample_id']).encode()).hexdigest())
        selected=[]
        while len(selected)<number:
            advanced=False
            for key in sorted(cells):
                while cells[key] and cells[key][0]['patient_id'] in patients: cells[key].pop(0)
                if cells[key] and len(selected)<number:
                    r=cells[key].pop(0); patients.add(r['patient_id']); selected.append(r); advanced=True
            if not advanced: raise RuntimeError(f'Insufficient unique patients for {group}')
        for r in selected:
            chosen.append({'case_id':f'C{len(chosen)+1:02d}','group':group,'film_id':r['sample_id'],
                **{k:r[k] for k in ['patient_id','visit','split','dicom_modality_first_instance','input_archive_relpath','archive_bytes','archive_sha256','native_pixel_sha256']}})
    manifest={'selection_seed':'chen-crop-pilot-v1','train_only':True,'unique_patients':len(patients),'availability_films':availability,'sampling':'failure-enriched; round-robin visit/modality; one film per patient; no population success estimate',
        'source_train_sha256':sha(TRAIN),'geometry_sha256':sha(PROJECT/'downstream/oa_kl_unified_v1/geometry.py'),'cases':chosen}
    save(target,manifest)
    return manifest


def author_input(gray, ds):
    spacing=None; source=None
    for key in ['PixelSpacing','ImagerPixelSpacing']:
        value=getattr(ds,key,None)
        if value is not None and len(value)==2:
            v=list(map(float,value))
            if all(math.isfinite(x) and x>0 for x in v): spacing=v;source=key;break
    if spacing is None: raise RuntimeError('No reliable pixel spacing for author geometry')
    h,w=gray.shape; nh,nw=round(h*spacing[0]/.14),round(w*spacing[1]/.14)
    if not (100<=nh<=10000 and 100<=nw<=10000): raise RuntimeError('Unexpected resampled size')
    sx,sy=nw/w,nh/h; ox=(nw-2560)//2;oy=(nh-2048)//2
    high=Image.fromarray(np.rint(gray*255).astype('uint8')).resize((nw,nh),Image.Resampling.BILINEAR)
    center=high.crop((ox,oy,ox+2560,oy+2048))
    small=np.array(center.resize((320,256),Image.Resampling.BILINEAR))
    return small,{'native_hw':[h,w],'spacing_mm':spacing,'spacing_source':source,'resampled_hw':[nh,nw],
        'resample_xy':[sx,sy],'center_origin_xy':[ox,oy],'center_hw':[2048,2560],'detector_hw':[256,320],
        'padding_needed':nw<2560 or nh<2048,'gray_policy':'frozen project presentation + p0.5/p99.5; uint8 for detector only',
        'flip':False,'model_normalize_mean':.431,'model_normalize_std':.0872}


def preview_crop(gray, box, size=224):
    x0,y0,x1,y1=map(int,box)
    image=Image.fromarray(np.rint(np.clip(gray[y0:y1,x0:x1],0,1)*255).astype('uint8')).convert('RGB')
    image.thumbnail((size,size),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(size,size),'black');canvas.paste(image,((size-image.width)//2,(size-image.height)//2))
    return canvas


def prepare():
    from drai.data.dicom_gray import read_pa_archive,presentation_values,normalize_roi
    from downstream.oa_kl_unified_v1.geometry import locate_knees,extract_regions
    selection=select()
    for index,r in enumerate(selection['cases']):
        d=OUT/'cases'/r['case_id'];d.mkdir(parents=True,exist_ok=True); path=d/'prepared.json'
        if path.exists(): continue
        record={k:r[k] for k in ['case_id','group','visit','dicom_modality_first_instance']}
        try:
            ds,pixels,receipt=read_pa_archive(NAS/r['input_archive_relpath'],expected_bytes=int(r['archive_bytes']),source_root=NAS)
            if receipt['archive_sha256']!=r['archive_sha256'] or receipt['native_pixel_sha256']!=r['native_pixel_sha256']:raise RuntimeError('Source fingerprint differs from frozen train manifest')
            values,valid,presentation=presentation_values(pixels,ds)
            gray,normalization=normalize_roi(values,valid)
            del values,pixels
            loc=locate_knees(gray,valid,layout='bilateral'); old=[]; rejected=[]
            for region in loc['regions']:
                if region['quality_status']!='candidate':rejected.append(region);continue
                try:
                    crop=extract_regions(gray,valid,[region],image_size=448)[0]
                    item={'image_position':crop['image_position'],'bbox':crop['bbox_xyxy_exclusive']}
                    old.append(item);preview_crop(gray,item['bbox']).save(d/f"old_{item['image_position']}.png")
                except Exception as e: rejected.append({'image_position':region['image_position'],'extract_error':str(e)})
            stable=hashlib.sha256(('film-selection-v1|'+r['film_id']).encode()).hexdigest()
            cache=CACHE/'films'/stable/'case.json'
            cached_match=None
            if cache.exists():
                cached=json.loads(cache.read_text())
                a={x['image_position']:x['bbox_xyxy_exclusive'] for x in cached['knees'] if x['status']=='success'}
                b={x['image_position']:x['bbox'] for x in old}
                cached_match=a==b
                if not cached_match: raise RuntimeError(f'Existing cache disagreement: old={a}, rerun={b}')
            small,transform=author_input(gray,ds)
            np.save(d/'gray.npy',gray,allow_pickle=False);np.save(d/'detector_input.npy',small,allow_pickle=False)
            Image.fromarray(small).save(d/'detector_input.png')
            record.update(status='prepared',native_hw=list(gray.shape),old=old,old_rejected=rejected,old_cache_match=cached_match,transform=transform,
                source_hash_verified=True,normalization=normalization,presentation= presentation)
        except Exception as e:
            record.update(status='prepare_error',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
        save(path,record);gc.collect()
        save(OUT/'status.json',{'phase':'prepare','finished':index+1,'total':len(selection['cases']),'last':r['case_id'],'status':record['status']})
        print(r['case_id'],record['status'],len(record.get('old',[])),flush=True)
    save(OUT/'status.json',{'phase':'prepared','total':len(selection['cases'])})


def tensor_from_u8(small, divisor=1.):
    import torch
    a=np.repeat(small[None,:,:],3,axis=0).astype(np.float32)/divisor
    return torch.from_numpy((a-.431)/.0872).unsqueeze(0)


def backmap(box, transform, expansion=1.3):
    sx,sy=transform['resample_xy'];ox,oy=transform['center_origin_xy'];h,w=transform['native_hw']
    x0,y0,x1,y1=map(float,box);cx=(x0+x1)/2;cy=(y0+y1)/2
    hw=(x1-x0)*expansion/2;hh=(y1-y0)*expansion/2
    raw=[((cx-hw)*8+ox)/sx,((cy-hh)*8+oy)/sy,((cx+hw)*8+ox)/sx,((cy+hh)*8+oy)/sy]
    clipped=[max(0,min(w,math.floor(raw[0]))),max(0,min(h,math.floor(raw[1]))),max(0,min(w,math.ceil(raw[2]))),max(0,min(h,math.ceil(raw[3])))]
    return clipped,raw


def validate_controls(model):
    import torch,h5py
    from chen_yolo_adapter import decode,bbox_iou
    result=[]
    for path in sorted(ASSETS.glob('author_train_control_*.h5')):
        with h5py.File(path) as f:
            image=f['images'][:].astype(np.float32)
            # deepdish stores small lists in HDF5 group attributes; read numbers only, no pickle.
            gt=np.array([[float(f['gt_boxes'][f'i{i}'].attrs[f'i{j}']) for j in range(4)] for i in range(2)])
        if image.shape!=(256,320,3) or image.min()<0 or image.max()>255:raise RuntimeError('Unexpected author control image')
        a=image.transpose(2,0,1)
        with torch.inference_mode():
            pred=decode(model(torch.from_numpy((a-.431)/.0872).unsqueeze(0)))
            incorrectly_scaled=decode(model(torch.from_numpy((a/255.-.431)/.0872).unsqueeze(0)))
        ious=[max([bbox_iou(box,x['bbox']) for x in pred],default=0.) for box in gt]
        result.append({'file':path.name,'status':'checked','image_hw_channels':list(image.shape),'pixel_range':[float(image.min()),float(image.max())],
            'input_policy':'literal author float32 H5 + Normalize, no division by 255','predicted_boxes':pred,'gt_boxes':gt.tolist(),'best_iou_per_gt':ious,
            'incorrect_divide_255_detection_count':len(incorrectly_scaled)})
    if len(result)!=3 or any(len(r['predicted_boxes'])!=2 or min(r['best_iou_per_gt'])<.5 for r in result):
        save(OUT/'adapter_control_failure.json',result)
        raise RuntimeError('Author control compatibility check failed; do not run study cases')
    return result


def infer():
    import torch
    from chen_yolo_adapter import load_detector,validate_author_forward,decode,bbox_iou
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model,load_report=load_detector(ASSETS/'chen_yolov2_1wd.pth')
    case_dirs=sorted((OUT/'cases').iterdir())
    first=next(d for d in case_dirs if (d/'detector_input.npy').exists())
    forward=validate_author_forward(model,ASSETS/'author_source',tensor_from_u8(np.load(first/'detector_input.npy')))
    controls=validate_controls(model)
    synthetic={'resample_xy':[2.,2.],'center_origin_xy':[120,80],'native_hw':[2000,2000]}
    mapped,_=backmap([10,20,50,60],synthetic,expansion=1.)
    assert mapped==[100,120,260,280]
    validation={'load':load_report,'author_forward':forward,'coordinate_synthetic_passed':True,'author_train_controls':controls,'model_sha256':sha(ASSETS/'chen_yolov2_1wd.pth'),
        'input_policy':'author float32 0..255, no /255; normalize mean .431/std .0872; confirmed against author train controls before pilot inference'}
    save(OUT/'adapter_validation.json',validation)
    for index,d in enumerate(case_dirs):
        target=d/'result.json'
        if target.exists():continue
        p=json.loads((d/'prepared.json').read_text())
        if p['status']!='prepared': save(target,p);continue
        try:
            started=time.monotonic()
            p['transform']['model_input_range_before_normalize']='0..255 float32 (author literal pipeline)'
            p['transform']['model_input_divisor']=1.
            with torch.inference_mode(): raw=decode(model(tensor_from_u8(np.load(d/'detector_input.npy'))))
            h,w=p['native_hw'];gray=np.load(d/'gray.npy',mmap_mode='r');mapped=[]
            for item in raw:
                box,unclipped=backmap(item['bbox'],p['transform'])
                if box[2]<=box[0] or box[3]<=box[1]:continue
                pos='image_left' if (box[0]+box[2])/2<w/2 else 'image_right'
                mapped.append({'image_position':pos,'bbox':box,'unclipped_bbox':unclipped,'detector_bbox':item['bbox'],'score':item['score'],'clipped':any(abs(a-b)>1.01 for a,b in zip(box,unclipped))})
            selected=[]
            for pos in POSITIONS:
                candidates=[v for v in mapped if v['image_position']==pos]
                if candidates:
                    best=max(candidates,key=lambda v:v['score']);selected.append(best)
                    preview_crop(gray,best['bbox']).save(d/f'yolo_{pos}.png')
            agreement=[]
            for old in p['old']:
                match=next((v for v in selected if v['image_position']==old['image_position']),None)
                if match:
                    ob,nb=old['bbox'],match['bbox'];oa=(ob[2]-ob[0])*(ob[3]-ob[1]);na=(nb[2]-nb[0])*(nb[3]-nb[1])
                    agreement.append({'position':old['image_position'],'iou_between_methods':bbox_iou(ob,nb),'new_to_old_area':na/oa})
            p.update(status='inferred',yolo=selected,raw_detections=mapped,raw_count=len(raw),duplicates=len(mapped)-len(selected),agreement=agreement,infer_seconds=time.monotonic()-started)
        except Exception as e:p.update(status='infer_error',error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
        save(target,p);gc.collect()
        save(OUT/'status.json',{'phase':'infer','finished':index+1,'total':len(case_dirs),'last':d.name,'status':p['status']})
        print(d.name,p['status'],len(p.get('old',[])),len(p.get('yolo',[])),flush=True)
    report()


def font(size):
    p=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(p),size) if p.exists() else ImageFont.load_default()


def panel(d, result):
    canvas=Image.new('RGB',(1392,340),'#101820');draw=ImageDraw.Draw(canvas)
    group=result['group'];draw.text((14,8),f"{d.name} | {group} | {result['dicom_modality_first_instance']} | {result['visit']}",font=font(18),fill='white')
    if result['status']!='inferred':draw.text((20,90),result['status'],font=font(22),fill='#ffb454');return canvas
    gray=np.load(d/'gray.npy',mmap_mode='r');h,w=gray.shape
    image=Image.fromarray(np.rint(gray*255).astype('uint8')).convert('RGB')
    # The peripheral mask affects only the manual preview; the detector and box statistics use all native pixels.
    mask=ImageDraw.Draw(image);mask.rectangle((0,0,w,round(h*.20)),fill='black');mask.rectangle((0,round(h*.88),w,h),fill='black')
    mask.rectangle((0,0,round(w*.05),h),fill='black');mask.rectangle((round(w*.95),0,w,h),fill='black')
    scale=min(390/w,270/h);view=image.resize((round(w*scale),round(h*scale)),Image.Resampling.BILINEAR)
    x=10+(390-view.width)//2;y=55+(270-view.height)//2;canvas.paste(view,(x,y))
    for key,color in [('old','#ffaa33'),('yolo','#00e1dc')]:
        for item in result[key]:
            a,b,c,e=item['bbox'];draw.rectangle((x+a*scale,y+b*scale,x+c*scale,y+e*scale),outline=color,width=2)
    draw.text((14,33),'Orange: current | Cyan: YOLO x1.3',font=font(14),fill='white')
    for i,(method,pos) in enumerate([(m,p) for m in ['old','yolo'] for p in POSITIONS]):
        x=418+i*240;name=f'{method}_{pos}.png';f=d/name
        title=f"{'Current' if method=='old' else 'YOLO'} | {'image L / R knee' if pos=='image_left' else 'image R / L knee'}"
        draw.text((x,43),title,font=font(13),fill='#ffaa33' if method=='old' else '#00e1dc')
        if f.exists(): canvas.paste(Image.open(f).convert('RGB'),(x,69))
        else: draw.text((x+40,160),'NO CROP',font=font(18),fill='#ffb454')
    return canvas


def report():
    public=OUT/'review';public.mkdir(exist_ok=True);all_results=[];panels=[]
    for d in sorted((OUT/'cases').iterdir()):
        f=d/'result.json'
        if not f.exists():continue
        r=json.loads(f.read_text());all_results.append(r)
        im=panel(d,r);im.save(public/f'{d.name}.png');panels.append(im)
    for start in range(0,len(panels),4):
        page=Image.new('RGB',(1392,340*len(panels[start:start+4])))
        for i,im in enumerate(panels[start:start+4]):page.paste(im,(0,i*340))
        page.save(public/f'contact_{start//4+1:02d}.jpg',quality=92)
    summary={'total':len(all_results),'status_counts':dict(Counter(r['status'] for r in all_results)),'groups':{},'note':'Failure-enriched train-only pilot; no manual ground-truth; IoU measures agreement, not accuracy.'}
    for group in ['old_two','old_one','old_zero','new_position']:
        rr=[r for r in all_results if r['group']==group];ok=[r for r in rr if r['status']=='inferred']
        summary['groups'][group]={'selected':len(rr),'readable':len(ok),'old_knees':sum(len(r['old']) for r in ok),'yolo_knees':sum(len(r['yolo']) for r in ok),'old_two_films':sum(len(r['old'])==2 for r in ok),'yolo_two_films':sum(len(r['yolo'])==2 for r in ok),'yolo_raw_count_histogram':dict(Counter(r['raw_count'] for r in ok))}
    agreements=[a for r in all_results for a in r.get('agreement',[])]
    summary['agreement']={'n':len(agreements),'median_iou':float(np.median([a['iou_between_methods'] for a in agreements])) if agreements else None,'median_yolo_to_old_area':float(np.median([a['new_to_old_area'] for a in agreements])) if agreements else None}
    summary['cached_old_bbox_matches']=sum(r.get('old_cache_match') is True for r in all_results)
    summary['padded_detector_inputs']=sum(r.get('transform',{}).get('padding_needed',False) for r in all_results)
    save(OUT/'summary.json',summary);save(public/'summary.json',summary)
    with (public/'comparison.csv').open('w',newline='') as f:
        fields=['case_id','group','visit','modality','status','old_count','yolo_count','raw_count','duplicates','median_agreement_iou']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in all_results:writer.writerow({'case_id':r['case_id'],'group':r['group'],'visit':r['visit'],'modality':r['dicom_modality_first_instance'],'status':r['status'],'old_count':len(r.get('old',[])),'yolo_count':len(r.get('yolo',[])),'raw_count':r.get('raw_count'),'duplicates':r.get('duplicates'),'median_agreement_iou':float(np.median([a['iou_between_methods'] for a in r.get('agreement',[])])) if r.get('agreement') else ''})
    tiles='\n'.join(f'<section data-group="{r["group"]}"><h2>{r["case_id"]} · {r["group"]}</h2><img loading="lazy" src="{r["case_id"]}.png"></section>' for r in all_results)
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Knee crop comparison</title><style>body{margin:24px;background:#101820;color:#eff4f8;font:16px system-ui}main{max-width:1450px;margin:auto}img{width:100%;border:1px solid #35454e}section{margin:28px 0}h2{font-size:18px}p{line-height:1.8}button{padding:8px 18px;margin:5px;background:#263d49;color:white;border:1px solid #5a7582;cursor:pointer}pre{white-space:pre-wrap;background:#1b2a34;padding:20px}</style><main><h1>Existing geometric crop vs. author YOLOv2: 48 training films</h1><p>Orange: existing geometric method; cyan: author YOLOv2 boxes with width and height expanded 1.3x. The four single-knee images are existing image-left, existing image-right, YOLO image-left and YOLO image-right. Displays keep the aspect ratio. The periphery is masked; inference uses the full original pixels.</p><p>One film per patient, training set only; old-method failures were deliberately enriched, so this batch cannot estimate the overall success rate. Two boxes do not mean both knees are located correctly; box IoU only shows agreement between methods. This batch was not used to tune thresholds or train models.</p><p>Native gray uses the frozen processing of this project. Run as the author: 0.14 mm/px, center 2048x2560, input 256x320, mean 0.431/std 0.0872; default score>=0.12, NMS=0.3. Films smaller than the center window are zero-padded and recorded. This is not a pixel-exact reproduction of the author full original gray chain.</p><div>FILTERS</div><pre>SUMMARY</pre>TILES</main><script>function filter(g){document.querySelectorAll('section').forEach(x=>x.hidden=g!=='all'&&x.dataset.group!==g)}</script></html>'''
    buttons=''.join(f'<button onclick="filter(\'{g}\')">{label}</button>' for g,label in [('all','All'),('old_two','old method: both knees'),('old_one','old method: one side'),('old_zero','old method: both failed'),('new_position','added by position split')])
    (public/'index.html').write_text(page.replace('FILTERS',buttons).replace('SUMMARY',html.escape(json.dumps(summary,ensure_ascii=False,indent=2))).replace('TILES',tiles))
    save(OUT/'status.json',{'phase':'completed','total':len(all_results),'summary':summary})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','infer','report']);a=p.parse_args()
    {'prepare':prepare,'infer':infer,'report':report}[a.phase]()
