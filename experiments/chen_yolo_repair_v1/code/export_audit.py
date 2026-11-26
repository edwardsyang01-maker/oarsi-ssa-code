from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import tarfile
from PIL import Image, ImageDraw
import candidates as c
from export_pending import tile
from run_supplement_v2 import reasons
p = c.p


def main():
    auth = json.loads((p.OUT/'preview_transfer_authorization_20261111.json').read_text())
    assert auth['user_answer'] == 'joint-anomaly review and spot-check previews for this round are allowed'
    assert json.loads((p.OUT/'completion.json').read_text())['phase'] == 'completed'
    groups = defaultdict(list)
    for source in sorted(p.DATA.glob('*/result.json')):
        case = source.parent.name
        if (c.DATA/'full'/case/'result.json').exists():
            continue
        r = json.loads(source.read_text())
        rois = [x for x in r.get('yolo', []) if x['status'] == 'saved']
        if len(rois) != 2 or reasons(r):
            continue
        flags = {f for x in rois for f in x['flags']}
        if flags - {'detector_center_padding'}:
            continue
        stratum = '|'.join([r['split'], r['presentation']['modality'], 'padding' if flags else 'routine'])
        rank = hashlib.sha256(('chen-engineering-audit-20261111|' + case).encode()).hexdigest()
        groups[stratum].append((rank, source, r, rois))
    selected = [(s, x) for s, xs in sorted(groups.items()) for x in sorted(xs, key=lambda x:x[0])[:16]]
    directory = c.ROOT/'preview_server_only/audit01'
    directory.mkdir(parents=True, exist_ok=False)
    records = []; private = []
    for stratum, (_, source, r, rois) in selected:
        for roi in sorted(rois, key=lambda x:x['image_position']):
            key = r['case_id'] + '_' + roi['clinical_side']
            item = {'key':key, 'case_id':r['case_id'], 'stratum':stratum,
                    'variant':'original', 'source_result_sha256':p.sha(source),
                    'gray01_sha256':roi['gray01_sha256'], 'valid_mask_sha256':roi['valid_mask_sha256'],
                    'flags':roi['flags'], 'review_status':'pending'}
            records.append(item); private.append((roi, r['native_hw']))
    pages = []
    for start in range(0, len(records), 64):
        canvas = Image.new('RGB', (1856,2048), '#101820'); draw = ImageDraw.Draw(canvas)
        for j, item in enumerate(records[start:start+64]):
            x=(j%8)*232; y=(j//8)*256
            draw.text((x+2,y+2), item['key']+' BASE', fill='#73dbe9')
            draw.text((x+2,y+15), item['stratum'], fill='#ffe9a6')
            roi, hw = private[start+j]
            canvas.paste(tile(roi,hw,1), (x+2,y+30))
        name=f'page_{start//64+1:03d}.png'; canvas.save(directory/name)
        pages.append({'page':name,'sha256':p.sha(directory/name),'keys':[x['key'] for x in records[start:start+64]]})
    index={'name':'audit01','records':records,'pages':pages,
           'stratum_available':{s:len(xs) for s,xs in groups.items()},
           'stratum_selected':dict(Counter(s for s,_ in selected)),
           'uses_kl':False,'contains_patient_ids_or_kl':False,'code_sha256':p.sha(Path(__file__)),
           'clinical_quality_approved':False}
    p.save(directory/'public_index.json',index)
    bundle=directory.with_suffix('.tar.gz')
    with tarfile.open(bundle,'w:gz') as tar:
        for file in sorted(directory.iterdir()):tar.add(file,arcname=file.name,recursive=False)
    receipt={'cases':len(selected),'knees':len(records),'pages':len(pages),'sha256':p.sha(bundle)}
    p.save(c.OUT/'audit01_export_receipt.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
