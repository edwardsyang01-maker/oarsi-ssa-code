import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import audit_rulers as a


def safe_image(item, native_hw):
    gray=np.load(item['gray01_npy'],allow_pickle=False)[0]
    valid=np.load(item['valid_mask_npy'],allow_pickle=False)[0]
    t=item['transform']; box=item['bbox']; h,w=native_hw
    xx=box[0]+(np.arange(448)-t['offset_x']+.5)/t['scale_x']-.5
    yy=box[1]+(np.arange(448)-t['offset_y']+.5)/t['scale_y']-.5
    safe=((xx>=.03*w)&(xx<.97*w))[None,:]&((yy>=.20*h)&(yy<.88*h))[:,None]
    b=item.get('parent_1p3_bbox',item['bbox'])
    safe|=((xx>=b[0])&(xx<b[2]))[None,:]&((yy>=b[1])&(yy<b[3]))[:,None]
    pixels=np.rint(gray*255).astype(np.uint8)
    pixels[~safe]=0
    return gray,valid,safe,pixels


def load_pilot():
    chosen=json.loads((a.ROOT/'outputs/pilot_cases.json').read_text())
    data=a.PROJECT/'data/processed/chen_yolo_2x_clean_v1/raw'
    records=[]
    for case in chosen:
        r=json.loads((data/case/'result.json').read_text())
        assert r['status']=='completed'
        for item in r['entries']:
            records.append({'item':item,'native_hw':r['native_hw']})
    return records


def make(records, directory, prefix, overlay=True):
    directory.mkdir(parents=True,exist_ok=False)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    pages=[]; public=[]; private=[]; flags=Counter(); images=[]
    for index,r in enumerate(records):
        item=r['item']; alias=f'{prefix}{index+1:04d}'
        g,v,safe,pix=safe_image(item,r['native_hw'])
        d=a.detect(g,v)
        im=Image.fromarray(pix).convert('RGB')
        if overlay:
            draw=ImageDraw.Draw(im)
            for ln in d['lines']:
                color='#40ff70' if ln['confidence']=='regular_chain' else '#ffad35'
                for p in ln['points']:
                    x,y=p['x'],p['y']
                    if safe[max(0,min(447,round(y))),max(0,min(447,round(x)))]:
                        draw.ellipse([x-8,y-8,x+8,y+8],outline=color,width=1)
        public.append({'alias':alias,'split':item['split'],'side':item['clinical_side'],
                       'lines':len(d['lines']),'edge_blobs':len(d['edge_blobs']),
                       'preview_hidden_pixels':int((v&~safe).sum()),'flags':item['selected_flags']})
        private.append({'alias':alias,'key':item['key'],'item':item,'native_hw':r['native_hw'],'detection':d})
        flags['with_line']+=bool(d['lines']);flags['with_edge_blob']+=bool(d['edge_blobs'])
        images.append(im)
    for start in range(0,len(records),16):
        page=Image.new('RGB',(1824,1960),'#171b22');draw=ImageDraw.Draw(page)
        for j,im in enumerate(images[start:start+16]):
            p=public[start+j];x=8+(j%4)*456;y=8+(j//4)*488
            text=f"{p['alias']} {p['split']} {p['side']} lines={p['lines']} edge={p['edge_blobs']}"
            draw.text((x,y),text,font=font,fill='white');page.paste(im,(x,y+30))
        name=f'{prefix.lower()}_{start//16+1:03d}.jpg'
        page.save(directory/name,quality=93,subsampling=0)
        pages.append({'file':name,'sha256':a.sha(directory/name),'aliases':[p['alias'] for p in public[start:start+16]]})
    a.save(directory/'public_index.json',{'records':public,'pages':pages,'preview_only':True})
    a.save(directory.parent/(prefix.lower()+'_private.json'),private)
    archive=directory.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(directory.iterdir()):z.write(p,p.name)
    receipt={'count':len(records),'pages':len(pages),'candidate_counts':dict(flags),'bundle':str(archive),'sha256':a.sha(archive),
             'patient_identity_and_kl_exported':False,'raw_dicom_exported':False}
    a.save(directory.parent/(prefix.lower()+'_receipt.json'),receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    cv2.setNumThreads(1)
    make(load_pilot(),a.REVIEW/'pilot_preview','P')
