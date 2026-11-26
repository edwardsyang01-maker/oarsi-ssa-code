import argparse
from collections import Counter
import json
from pathlib import Path
import tarfile

from PIL import Image, ImageDraw
import candidates as c
from verify_export_pilot import safe_tile
p=c.p
EXPORT=c.ROOT/'preview_server_only/review_batches'
GRAY={'high_saturation_fraction','low_saturation_fraction','low_gray_dynamic_range'}
IGNORE={'detector_center_padding'}
ABBR={'original':'BASE','modality_samebox':'RAW-box','modality_roi_samebox':'RAW-ROI',
      'center_modality':'RAW-center','full_original':'FOV-base','full_modality':'FOV-raw'}


def saved(result,variant,pos):
    return next((x for x in result['variants'].get(variant,[]) if x['image_position']==pos and x['status']=='saved'),None)


def propose(result,pos,pilot):
    code=result.get('prior_case_id','');side=p.SIDES[pos]
    if code in pilot['entries']:
        variant=pilot['entries'][code][side]['variant']
        roi=saved(result,variant,pos)
        if roi is None:raise RuntimeError('Reviewed pilot selection absent in full candidate')
        return variant,roi,'previous pilot engineering visual decision; full byte comparison required'
    original=saved(result,'original',pos)
    if original:
        flags=set(original['flags'])-IGNORE
        if not flags:return 'original',original,'retain original; no non-padding flags'
        if not flags-GRAY:
            if flags&{'high_saturation_fraction','low_gray_dynamic_range'}:
                candidate=saved(result,'modality_roi_samebox',pos)
                if candidate:return 'modality_roi_samebox',candidate,'proposed same-box raw gray normalization; requires visual review'
            return 'original',original,'gray flag requires visual review; original geometry retained'
    ranked=[]
    for preference,variant in enumerate(['original','full_original','center_modality','full_modality']):
        roi=saved(result,variant,pos)
        if roi is None:continue
        flags=set(roi['flags'])-IGNORE
        ranked.append(((len(flags-GRAY),len(flags&GRAY),preference),variant,roi))
    if not ranked:return None,None,'no saved fixed-model candidate; investigate before exclusion'
    _,variant,roi=min(ranked,key=lambda x:x[0])
    return variant,roi,'proposed fixed-model geometry alternative; requires visual review'


def render_batch(name,records,pilot):
    directory=EXPORT/name
    directory.mkdir(parents=True,exist_ok=False)
    public=[];private=[]
    for path,result in records:
        for pos,side in p.SIDES.items():
            variant,roi,reason=propose(result,pos,pilot)
            key=result['case_id']+'_'+side
            original=saved(result,'original',pos)
            record={'key':key,'case_id':result['case_id'],'side':side,'image_position':pos,
                    'variant':variant,'reason':reason,'result_sha256':p.sha(path),
                    'original_missing':original is None,'original_flags':original['flags'] if original else [],
                    'selected_flags':roi['flags'] if roi else [],'gray01_sha256':roi['gray01_sha256'] if roi else None,
                    'valid_mask_sha256':roi['valid_mask_sha256'] if roi else None,
                    'bbox':roi['bbox'] if roi else None,'native_hw':result.get('native_hw'),
                    'review_status':'pending','clinical_quality_approved':False}
            public.append(record);private.append({**record,'result_path':str(path),'roi':roi,'source_result':result})
    pages=[]
    for start in range(0,len(public),64):
        canvas=Image.new('RGB',(8*232,8*256),'#101820');draw=ImageDraw.Draw(canvas)
        items=private[start:start+64]
        for j,item in enumerate(items):
            x=(j%8)*232;y=(j//8)*256
            roi=item['roi'];label=item['key']+' '+ABBR.get(item['variant'],'MISSING')
            draw.text((x+3,y+1),label,fill='#73dbe9' if roi else '#ffbf64')
            draw.text((x+3,y+14),' | '.join(item['selected_flags'])[:32],fill='#ffe9a6')
            if roi:canvas.paste(safe_tile(roi,item['native_hw']),(x+2,y+30))
            else:draw.text((x+25,y+120),'NO CANDIDATE',fill='#ffbf64')
        filename=f'page_{start//64+1:03d}.png';canvas.save(directory/filename)
        pages.append({'page':filename,'sha256':p.sha(directory/filename),'keys':[x['key'] for x in items]})
    index={'batch':name,'repair_identity':c.identity()['identity'],'records':public,'pages':pages,
           'selection_uses_kl':False,'contains_patient_ids_or_labels':False,'source_code_sha256':p.sha(Path(__file__))}
    p.save(directory/'public_index.json',index)
    p.save(c.OUT/(name+'_review_private.json'),private)
    bundle=EXPORT/(name+'.tar.gz')
    with tarfile.open(bundle,'w:gz') as archive:
        for f in sorted(directory.iterdir()):archive.add(f,arcname=f.name,recursive=False)
    receipt={'batch':name,'cases':len(records),'knees':len(public),'pages':len(pages),
             'missing_proposals':sum(x['variant'] is None for x in public),
             'variants':dict(Counter(x['variant'] for x in public)),'bundle_sha256':p.sha(bundle),
             'bundle_bytes':bundle.stat().st_size,'public_index_sha256':p.sha(directory/'public_index.json')}
    p.save(c.OUT/(name+'_export_receipt.json'),receipt);print(json.dumps(receipt),flush=True)


def individual(cases):
    directory=EXPORT/'individual';directory.mkdir(parents=True,exist_ok=True)
    result=[]
    for case in cases:
        path=c.DATA/'full'/case/'result.json';r=json.loads(path.read_text())
        canvas=Image.new('RGB',(1392,520),'#101820');draw=ImageDraw.Draw(canvas)
        draw.text((6,4),case+' | joint comparison; source perimeter redacted',fill='white')
        for yi,pos in enumerate(p.SIDES):
            y=24+248*yi
            for xi,variant in enumerate(c.POLICY['variants']):
                x=232*xi;draw.text((x+3,y),ABBR[variant]+' '+p.SIDES[pos],fill='#73dbe9')
                roi=saved(r,variant,pos)
                if roi:canvas.paste(safe_tile(roi,r['native_hw']),(x+2,y+18))
                else:draw.text((x+18,y+100),'MISSING',fill='#ffbf64')
        dest=directory/(case+'.png');canvas.save(dest);result.append({'case_id':case,'sha256':p.sha(dest)})
    print(json.dumps(result),flush=True)


def main():
    args=argparse.ArgumentParser();args.add_argument('--batch');args.add_argument('--max-cases',type=int,default=320)
    args.add_argument('--individual',nargs='*');arg=args.parse_args()
    auth=json.loads((p.OUT/'preview_transfer_authorization_20261111.json').read_text())
    assert auth['user_answer']=='joint-anomaly review and spot-check previews for this round are allowed'
    if arg.individual:individual(arg.individual);return
    assert arg.batch and '/' not in arg.batch
    seen=set()
    for path in EXPORT.glob('*/public_index.json'):
        seen.update(x['case_id'] for x in json.loads(path.read_text())['records'])
    pilot=json.loads((c.OUT/'pilot_ai_review.json').read_text());assert pilot['complete']
    records=[]
    for path in sorted((c.DATA/'full').glob('*/result.json')):
        if path.parent.name in seen:continue
        r=json.loads(path.read_text())
        if r['status']!='candidates_generated':continue
        records.append((path,r))
        if len(records)>=arg.max_cases:break
    if not records:raise RuntimeError('No new repair candidates to export')
    render_batch(arg.batch,records,pilot)


if __name__=='__main__':main()
