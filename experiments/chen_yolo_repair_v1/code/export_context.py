import argparse
import json
from pathlib import Path
import tarfile
import numpy as np
from PIL import Image,ImageDraw
import candidates as c
from export_review import individual,EXPORT
p=c.p


def context(case,directory):
    from drai.data.dicom_gray import read_pa_archive,presentation_values,normalize_roi
    r=json.loads((c.DATA/'full'/case/'result.json').read_text())
    ds,pixels,receipt=read_pa_archive(p.NAS/r['input_archive_relpath'],expected_bytes=int(r['archive_bytes']),source_root=p.NAS)
    assert receipt['archive_sha256']==r['archive_sha256'] and receipt['native_pixel_sha256']==r['native_pixel_sha256']
    values,valid,pres=presentation_values(pixels,ds)
    original,_=normalize_roi(values,valid);del values
    raw,values,_=c.modality_gray(pixels,ds,valid,pres);del pixels,values
    h,w=original.shape;bounds=[round(.03*w),round(.20*h),round(.97*w),round(.88*h)]
    canvas=Image.new('RGB',(1392,968),'#101820');draw=ImageDraw.Draw(canvas)
    x0,y0,x1,y1=bounds;scales=[]
    for i,(label,gray) in enumerate([('ORIGINAL joint-band',original),('RAW modality joint-band',raw)]):
        image=Image.fromarray(np.rint(np.clip(gray[y0:y1,x0:x1],0,1)*255).astype(np.uint8)).convert('RGB')
        image.thumbnail((680,414),Image.Resampling.BILINEAR)
        x=i*696+(696-image.width)//2;y=30+(414-image.height)//2
        canvas.paste(image,(x,y));draw.text((i*696+8,5),case+' '+label,fill='white')
        mx=x+(w/2-x0)*image.width/(x1-x0)
        draw.line((mx,y,mx,y+image.height),fill='#41d0d0',width=1)
        scales.append({'offset_xy':[x,y],'size_wh':list(image.size),'native_band_xyxy':bounds})
    with Image.open(EXPORT/'individual'/(case+'.png')) as panel:canvas.paste(panel,(0,448))
    dest=directory/(case+'.png');canvas.save(dest)
    return {'case_id':case,'preview':dest.name,'sha256':p.sha(dest),'native_hw':[h,w],
            'display_transforms':scales,'result_sha256':p.sha(c.DATA/'full'/case/'result.json')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('cases',nargs='+');a=parser.parse_args()
    assert '/' not in a.name and all(x.startswith(('T','V')) and x[1:].isdigit() for x in a.cases)
    auth=json.loads((p.OUT/'preview_transfer_authorization_20261111.json').read_text());assert auth['user_answer']=='joint-anomaly review and spot-check previews for this round are allowed'
    directory=EXPORT/a.name;directory.mkdir(parents=True,exist_ok=False)
    individual(a.cases);records=[]
    for case in a.cases:
        records.append(context(case,directory));print('exported',case,flush=True)
    p.save(directory/'public_index.json',{'records':records,'no_patient_identity_or_kl':True,'code_sha256':p.sha(Path(__file__))})
    bundle=EXPORT/(a.name+'.tar.gz')
    with tarfile.open(bundle,'w:gz') as archive:
        for f in sorted(directory.iterdir()):archive.add(f,arcname=f.name,recursive=False)
    receipt={'name':a.name,'cases':len(records),'bundle_sha256':p.sha(bundle),'bundle_bytes':bundle.stat().st_size}
    p.save(c.OUT/(a.name+'_export_receipt.json'),receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
