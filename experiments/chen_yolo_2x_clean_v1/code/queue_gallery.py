import argparse
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

import audit_rulers as a
from review_gallery import safe_image
from run_audit_v2 import source_paths,AUDIT
from clean_policy import propose


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True);parser.add_argument('--limit',type=int,default=768);args=parser.parse_args()
    cv2.setNumThreads(1)
    directory=a.ROOT/'review_queue'/args.batch;public=directory/'public';public.mkdir(parents=True,exist_ok=False)
    done=set()
    for p in (a.ROOT/'review_queue').glob('*/mapping_private.json'):
        done.update(r['key'] for r in json.loads(p.read_text()))
    paths=source_paths();records=[]
    for path in sorted((AUDIT/'cases').glob('*.json')):
        audit=json.loads(path.read_text());source=None
        for ent in audit['entries']:
            if ent['key'] in done:continue
            flags=list(ent['review_flags'])
            if not ent['likely_ruler_count'] and ent['edge_candidates']:flags.append('unexplained_edge_marker')
            if not flags:continue
            if source is None:
                source=json.loads(paths[audit['case_id']].read_text());assert a.sha(paths[audit['case_id']])==audit['source_result_sha256']
            item=next(i for i in source['entries'] if i['key']==ent['key'])
            records.append({'key':ent['key'],'item':item,'native_hw':source['native_hw'],'audit':ent,'flags':flags})
            if len(records)>=args.limit:break
        if len(records)>=args.limit:break
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    pub=[];pages=[]
    for start in range(0,len(records),32):
        page=Image.new('RGB',(1792,1016),'#171b22');draw=ImageDraw.Draw(page)
        for j,r in enumerate(records[start:start+32]):
            index=start+j;alias=f'{args.batch.upper()}-{index+1:04d}';r['alias']=alias
            g,v,safe,pix=safe_image(r['item'],r['native_hw']);mask,met=propose(r['audit'],r['item'],g,v)
            r['mask_proposal']=met
            im=np.stack([pix]*3,axis=-1)
            border=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)&safe
            im[border]=[255,90,40]
            tile=Image.fromarray(im).resize((224,224),Image.Resampling.LANCZOS)
            x=(j%8)*224;y=(j//8)*254
            draw.text((x+2,y+1),alias+' '+r['item']['clinical_side'],font=font,fill='white')
            draw.text((x+2,y+15),','.join(r['flags'])[:27],font=font,fill='#e6bd8c');page.paste(tile,(x,y+30))
            pub.append({'alias':alias,'split':r['item']['split'],'side':r['item']['clinical_side'],'flags':r['flags'],
                        'mask_pixels':met['mask_pixels'],'core_overlap':met['core_overlap_pixels']})
        name=f'page_{start//32+1:03d}.jpg';page.save(public/name,quality=95,subsampling=0)
        pages.append({'file':name,'aliases':[r['alias'] for r in records[start:start+32]],'sha256':a.sha(public/name)})
    a.save(directory/'mapping_private.json',records);a.save(public/'public_index.json',{'records':pub,'pages':pages})
    bundle=directory/'preview.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(public.iterdir()):z.write(p,p.name)
    receipt={'batch':args.batch,'records':len(records),'pages':len(pages),'sha256':a.sha(bundle),'private_identity_and_kl_exported':False}
    a.save(directory/'receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
