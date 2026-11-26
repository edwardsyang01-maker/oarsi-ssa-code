import argparse,json,zipfile
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import audit_rulers as a
from review_gallery import safe_image


def reconstructed_mask(entry,valid):
    yy,xx=np.mgrid[:448,:448];mask=np.zeros((448,448),bool)
    for line in entry['lines']:
        center=line['intercept']+line['slope']*yy
        if entry['side']=='L':mask|=xx<=center+line['inner_width']
        else:mask|=xx>=center-line['inner_width']
    mask&=valid;assert int(mask.sum())==entry['metrics']['mask_pixels']
    return mask


def main():
    arg=argparse.ArgumentParser();arg.add_argument('--cohort',required=True);arg.add_argument('--name',required=True);arg.add_argument('--indices');arg.add_argument('--version',default='v1');args=arg.parse_args()
    root=a.ROOT/('proposal_validation' if args.version=='v1' else 'proposal_validation_'+args.version);records=json.loads((root/(args.cohort+'_private.json')).read_text())
    if args.indices:selected=sorted(set(map(int,args.indices.split(','))))
    else:
        if args.cohort=='pilot':truth=json.loads((a.ROOT/'code/pilot_visual_labels.json').read_text())['ruler_visible']+[60,148]
        else:truth=json.loads((a.ROOT/'code/q01_visual_review.json').read_text())['confirmed_ruler']+[25,65,101,118,152,188,200,231,250,263,296,346,357,364,377]
        selected=sorted(set(truth+[n for n,r in enumerate(records,1) if r['proposal']['review_flags']]))
    out=root/args.name;out.mkdir(exist_ok=False);panels=[];public=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for n in selected:
        r=records[n-1];e=r['proposal'];g,v,safe,pix=safe_image(r['item'],r['native_hw']);mask=reconstructed_mask(e,v)
        before=np.repeat(pix[:,:,None],3,axis=2);border=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)&safe
        before[border]=[255,100,60];after=pix.copy();after[mask]=0
        panel=Image.new('RGB',(912,480),'#171b22');draw=ImageDraw.Draw(panel)
        label=r['alias']+' '+e['side']+f" | mask={e['metrics']['mask_pixels']} oldROI={e['metrics']['parent_1p3_overlap_pixels']} flags={len(e['review_flags'])}"
        draw.text((4,3),label,font=font,fill='white');panel.paste(Image.fromarray(before),(4,28));panel.paste(Image.fromarray(after).convert('RGB'),(460,28))
        panels.append(panel);public.append({'alias':r['alias'],'metrics':e['metrics'],'review_flags':e['review_flags']})
    pages=[]
    for start in range(0,len(panels),8):
        canvas=Image.new('RGB',(1824,1920),'#171b22')
        for j,panel in enumerate(panels[start:start+8]):canvas.paste(panel,((j%2)*912,(j//2)*480))
        name=f'compare_{start//8+1:02d}.png';canvas.save(out/name)
        pages.append({'file':name,'sha256':a.sha(out/name),'aliases':[r['alias'] for r in public[start:start+8]]})
    a.save(out/'public_index.json',{'records':public,'pages':pages,'proposal_only':True})
    bundle=root/(args.name+'.zip')
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):z.write(f,f.name)
    print(json.dumps({'count':len(panels),'pages':len(pages),'sha256':a.sha(bundle)}),flush=True)


if __name__=='__main__':main()
