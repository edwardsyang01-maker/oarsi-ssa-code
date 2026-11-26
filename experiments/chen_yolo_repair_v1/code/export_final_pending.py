import argparse
import json
from pathlib import Path
import tarfile
import numpy as np
from PIL import Image, ImageDraw
import candidates as c
from export_review import saved
p=c.p
FOCUSED={'T008682_L','T008682_R','T014352_L','T014352_R','V000925_L','V000925_R'}
VARIANTS={'full':['original','modality_samebox','modality_roi_samebox','center_modality','full_original','full_modality'],
          'extra':['inv_center_raw','inv_full_raw','lower_original','lower_raw','lower_inv_raw']}


def tile(roi,hw,focused):
    a=np.load(roi['gray01_npy'],allow_pickle=False)[0].copy();t=roi['transform'];b=roi['bbox']
    x=b[0]+(np.arange(224)-t['offset_x']+.5)/t['scale_x']
    y=b[1]+(np.arange(224)-t['offset_y']+.5)/t['scale_y']
    if focused:
        a[(y<b[1]+.12*(b[3]-b[1]))|(y>b[3]-.12*(b[3]-b[1])),:]=0
        a[:,(x<b[0]+.08*(b[2]-b[0]))|(x>b[2]-.08*(b[2]-b[0]))]=0
    else:
        a[y<.2*hw[0],:]=0
    return Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8)).convert('RGB')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ledger',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    auth=json.loads((p.OUT/'preview_transfer_authorization_20261111.json').read_text())
    assert auth['user_answer']=='joint-anomaly review and spot-check previews for this round are allowed' and '/' not in a.name
    ledger=Path(a.ledger);keys=sorted(set(json.loads(ledger.read_text())['unresolved_known']))
    directory=c.ROOT/'preview_server_only'/a.name;directory.mkdir(parents=True,exist_ok=False)
    records=[];pages=[]
    for family,variants in VARIANTS.items():
        for start in range(0,len(keys),5):
            canvas=Image.new('RGB',(1392,256*min(5,len(keys)-start)),'#101820');draw=ImageDraw.Draw(canvas)
            page=f'{family}_{start//5+1:03d}.png'
            for row,key in enumerate(keys[start:start+5]):
                case,side=key.split('_');pos={'R':'image_left','L':'image_right'}[side]
                source=c.DATA/family/case/'result.json'
                result=json.loads(source.read_text()) if source.exists() else None
                item={'key':key,'page':page,'candidate_family':family,'result_sha256':p.sha(source) if result else None,
                      'preview_policy':'inner_joint_window' if key in FOCUSED else 'source_top20_mask','candidates':[]}
                for col,variant in enumerate(variants):
                    roi=saved(result,variant,pos) if result else None;x=col*232;y=row*256
                    draw.text((x+2,y+2),key+' '+variant,fill='#73dbe9')
                    if roi:
                        canvas.paste(tile(roi,result['native_hw'],key in FOCUSED),(x+2,y+28))
                        item['candidates'].append({'variant':variant,'bbox':roi['bbox'],
                            'gray01_sha256':roi['gray01_sha256'],'valid_mask_sha256':roi['valid_mask_sha256']})
                    else:draw.text((x+25,y+115),'MISSING',fill='#ffbf64')
                records.append(item)
            canvas.save(directory/page);pages.append({'page':page,'sha256':p.sha(directory/page),'keys':keys[start:start+5]})
    p.save(directory/'public_index.json',{'name':a.name,'records':records,'pages':pages,'ledger_sha256':p.sha(ledger),
        'code_sha256':p.sha(Path(__file__)),'repair_identity':c.identity()['identity'],
        'preview_only_mask':True,'contains_patient_ids_or_kl':False})
    bundle=directory.with_suffix('.tar.gz')
    with tarfile.open(bundle,'w:gz') as tar:
        for f in sorted(directory.iterdir()):tar.add(f,arcname=f.name,recursive=False)
    receipt={'name':a.name,'knees':len(keys),'pages':len(pages),'bundle_sha256':p.sha(bundle),'bundle_bytes':bundle.stat().st_size}
    p.save(c.OUT/(a.name+'_export_receipt.json'),receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
