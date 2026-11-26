import argparse
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

import audit_rulers as a
from review_gallery import safe_image


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);args=p.parse_args()
    root=a.ROOT/'review_queue'/args.batch;records=json.loads((root/'mapping_private.json').read_text())
    triage=json.loads((a.ROOT/'code'/(args.batch+'_triage.json')).read_text())
    selected=sorted(set(triage['ruler_or_partial_for_detail']+triage['false_mask_proposals_for_detail']+triage['coverage_for_detail']))
    out=root/'detail';out.mkdir(exist_ok=False);pages=[];public=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    for start in range(0,len(selected),16):
        canvas=Image.new('RGB',(1824,1920),'#171b22');draw=ImageDraw.Draw(canvas)
        for j,n in enumerate(selected[start:start+16]):
            r=records[n-1];assert r['alias'].endswith(f'-{n:04d}')
            _,_,_,pix=safe_image(r['item'],r['native_hw']);x=(j%4)*456;y=(j//4)*480
            draw.text((x+4,y+2),r['alias']+' '+r['item']['clinical_side'],font=font,fill='white')
            canvas.paste(Image.fromarray(pix).convert('RGB'),(x+4,y+27))
            public.append({'alias':r['alias'],'side':r['item']['clinical_side']})
        name=f'detail_{start//16+1:02d}.jpg';canvas.save(out/name,quality=96,subsampling=0)
        pages.append({'file':name,'sha256':a.sha(out/name),'aliases':[records[n-1]['alias'] for n in selected[start:start+16]]})
    a.save(out/'public_index.json',{'records':public,'pages':pages})
    bundle=root/'detail.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):z.write(f,f.name)
    print(json.dumps({'batch':args.batch,'count':len(selected),'pages':len(pages),'sha256':a.sha(bundle)}))


if __name__=='__main__':main()
