import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

import audit_rulers as a
from review_gallery import safe_image


def main():
    cv2.setNumThreads(1)
    records=json.loads((a.REVIEW/'pilot_detector_v2_private.json').read_text())
    truth=json.loads((a.ROOT/'code/pilot_visual_labels.json').read_text())
    chosen=sorted(set(truth['ruler_visible']+truth['uncertain_partial_marker']+truth['privacy_preview_obscures_joint_context']+truth['anatomical_quality_attention']))
    directory=a.REVIEW/'pilot_mask_compare';directory.mkdir(exist_ok=False)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    public=[];private=[];panels=[]
    for n in chosen:
        r=records[n-1];item=r['item'];g,v,safe,pix=safe_image(item,r['native_hw'])
        lines=[ln for ln in r['detection']['lines'] if a.classify_line(ln,item)['likely_ruler']]
        mask,met=a.ruler_mask(lines,item,v)
        if n in [38,165]:
            mask[:]=False;met['manual_detail_pending']=True
        after=pix.copy();after[mask]=0
        before=Image.fromarray(pix).convert('RGB');ov=np.array(before)
        border=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)&safe
        ov[border]=[255,100,60]
        panel=Image.new('RGB',(912,480),'#171b22');draw=ImageDraw.Draw(panel)
        label=r['alias']+' '+item['split']+' '+item['clinical_side']+f' | proposal={int(mask.sum())} protected={met["protected_overlap_pixels"]}'
        draw.text((4,4),label,font=font,fill='white');panel.paste(Image.fromarray(ov),(4,28));panel.paste(Image.fromarray(after).convert('RGB'),(460,28))
        panels.append(panel)
        public.append({'alias':r['alias'],'split':item['split'],'side':item['clinical_side'],'mask_metrics':met,'preview_hidden_pixels':int((v&~safe).sum())})
        private.append({'alias':r['alias'],'key':item['key'],'lines':lines,'mask_metrics':met})
    pages=[]
    for start in range(0,len(panels),8):
        canvas=Image.new('RGB',(1824,1920),'#171b22')
        for j,p in enumerate(panels[start:start+8]):canvas.paste(p,((j%2)*912,(j//2)*480))
        name=f'comparison_{start//8+1:02d}.jpg';canvas.save(directory/name,quality=95,subsampling=0)
        pages.append({'file':name,'aliases':[x['alias'] for x in public[start:start+8]],'sha256':a.sha(directory/name)})
    a.save(directory/'public_index.json',{'records':public,'pages':pages})
    a.save(a.REVIEW/'pilot_mask_proposals_private.json',private)
    bundle=directory.with_suffix('.zip')
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(directory.iterdir()):z.write(p,p.name)
    print(json.dumps({'samples':len(panels),'pages':len(pages),'sha256':a.sha(bundle),'overlap_cases':[r['alias'] for r in public if r['mask_metrics']['protected_overlap_pixels']]}))


if __name__=='__main__':main()
