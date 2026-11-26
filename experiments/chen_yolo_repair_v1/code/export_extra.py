import argparse
import json
from pathlib import Path
import tarfile
import numpy as np
from PIL import Image
import extra_candidates as e
import export_review as r
c=e.c;p=c.p


def tile(roi,native_hw):
    a=np.load(roi['gray01_npy'],allow_pickle=False)[0].copy();t=roi['transform']
    y=roi['bbox'][1]+(np.arange(224)-t['offset_y']+.5)/t['scale_y']
    a[y<.2*native_hw[0],:]=0
    return Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8)).convert('RGB')


def propose(result,pos,pilot):
    pres=result['original_presentation']
    inverse_group=pres['photometric_interpretation']=='MONOCHROME2' and pres['presentation_inverse_applied']
    order=['inv_center_raw','inv_full_raw','lower_inv_raw'] if inverse_group else ['lower_raw','lower_original']
    ranked=[]
    for preference,v in enumerate(order):
        roi=r.saved(result,v,pos)
        if roi:
            flags=set(roi['flags'])-r.IGNORE
            ranked.append(((len(flags-r.GRAY),len(flags&r.GRAY),preference),v,roi))
    if not ranked:return None,None,'no appropriate extra candidate; continue review'
    _,v,roi=min(ranked,key=lambda x:x[0])
    return v,roi,'extra fixed-model candidate; intensity inversion is not a spatial flip; visual decision required'


def main():
    a=argparse.ArgumentParser();a.add_argument('--selection',required=True);a.add_argument('--name',required=True);args=a.parse_args()
    selection=Path(args.selection);completion=json.loads((c.OUT/(selection.stem+'_completion.json')).read_text())
    assert completion['phase']=='completed' and completion['case_errors']==0
    ident=e.identity();assert completion['extra_identity']==ident['identity']
    records=[]
    for case in json.loads(selection.read_text()):
        path=c.DATA/'extra'/case/'result.json';rec=json.loads(path.read_text());assert rec['extra_identity']==ident['identity']
        parent=json.loads((c.DATA/'full'/case/'result.json').read_text());rec['variants']['original']=parent['variants']['original']
        records.append((path,rec))
    r.propose=propose;r.safe_tile=tile
    r.ABBR.update(inv_center_raw='INV-center',inv_full_raw='INV-full',lower_original='LOW-base',lower_raw='LOW-raw',lower_inv_raw='LOW-inv')
    r.render_batch(args.name,records,{'entries':{}})
    directory=r.EXPORT/args.name;index=json.loads((directory/'public_index.json').read_text())
    index.update(extra_identity=ident['identity'],extra_export_code_sha256=p.sha(Path(__file__)),
                 preview_policy='true ROI; original source top 20% masked; no lateral/bottom preview-only mask')
    p.save(directory/'public_index.json',index)
    bundle=r.EXPORT/(args.name+'.tar.gz')
    with tarfile.open(bundle,'w:gz') as archive:
        for f in sorted(directory.iterdir()):archive.add(f,arcname=f.name,recursive=False)
    receipt=json.loads((c.OUT/(args.name+'_export_receipt.json')).read_text());receipt.update(extra_identity=ident['identity'],
           bundle_sha256=p.sha(bundle),bundle_bytes=bundle.stat().st_size,public_index_sha256=p.sha(directory/'public_index.json'))
    p.save(c.OUT/(args.name+'_export_receipt.json'),receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
