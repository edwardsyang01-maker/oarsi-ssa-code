import argparse,json,zipfile
from collections import Counter
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import audit_rulers as a
from review_gallery import safe_image


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='clean_pilot_v1');parser.add_argument('--name',default='actual_comparison');parser.add_argument('--limit',type=int,default=0);parser.add_argument('--indices');args=parser.parse_args()
    root=a.ROOT/args.stage;assert (root/'completion.json').exists()
    records=[]
    for path in sorted((root/'cases').glob('*.json')):
        case=json.loads(path.read_text())
        for item in case['entries']:records.append({'item':item,'native_hw':case['native_hw'],'result_path':str(path)})
    if args.limit:records=records[:args.limit]
    out=root/args.name;out.mkdir(exist_ok=False);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    public=[];private=[];panels=[];counts=Counter()
    for n,r in enumerate(records,1):
        if args.indices and n not in set(map(int,args.indices.split(','))):continue
        item=r['item'];alias=f'CP{n:04d}';status=item['cleanup_status'];counts[status]+=1
        old={**item,'gray01_npy':item['raw_gray01_npy'],'valid_mask_npy':item['raw_valid_mask_npy']}
        _,v,safe,before=safe_image(old,r['native_hw']);_,_,_,after=safe_image(item,r['native_hw'])
        rgb=np.repeat(before[:,:,None],3,axis=2)
        if item['cleanup_mask_npy']:
            mask=np.load(item['cleanup_mask_npy'],allow_pickle=False)[0]
            border=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool)&safe
            rgb[border]=[255,110,50]
        panel=Image.new('RGB',(912,480),'#171b22');draw=ImageDraw.Draw(panel)
        state={'accepted_cleaned':'CLEAN','accepted_unchanged':'KEEP','held_unmodified':'HOLD (unchanged)','held_residual_candidate':'HOLD (residual)'}[status]
        draw.text((4,3),f"{alias} {item['split']} {item['clinical_side']} | {state} | before / after",font=font,fill='white')
        panel.paste(Image.fromarray(rgb),(4,28));panel.paste(Image.fromarray(after).convert('RGB'),(460,28));panels.append(panel)
        public.append({'alias':alias,'split':item['split'],'side':item['clinical_side'],'status':status,'mask_pixels':item['proposed_mask_pixels'],'reasons':item['cleanup_review_reasons']})
        private.append({'alias':alias,'key':item['key'],**r})
    pages=[]
    for start in range(0,len(panels),8):
        canvas=Image.new('RGB',(1824,1920),'#171b22')
        for j,panel in enumerate(panels[start:start+8]):canvas.paste(panel,((j%2)*912,(j//2)*480))
        name=f'actual_{start//8+1:02d}.png';canvas.save(out/name);pages.append({'file':name,'sha256':a.sha(out/name),'aliases':[r['alias'] for r in public[start:start+8]]})
    a.save(out/'public_index.json',{'records':public,'pages':pages,'actual_arrays':True})
    a.save(root/(args.name+'_private.json'),private);bundle=root/(args.name+'.zip')
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.iterdir()):z.write(f,f.name)
    receipt={'images':len(public),'pages':len(pages),'counts':dict(counts),'sha256':a.sha(bundle),'patient_identity_and_kl_exported':False,'raw_dicom_exported':False}
    a.save(root/(args.name+'_receipt.json'),receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
