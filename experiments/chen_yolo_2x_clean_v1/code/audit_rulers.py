from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math

import cv2
import numpy as np

PROJECT = Path('/path/to/drai')
ROOT = PROJECT / 'experiments/chen_yolo_2x_clean_v1'
REVIEW = ROOT / 'review_v1'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def blobs(gray, valid):
    a = np.rint(gray * 255).astype(np.uint8)
    opening = cv2.morphologyEx(a, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
    top = cv2.subtract(a, opening)
    masks = [(a >= level) & valid for level in [25, 50, 85, 128, 166, 204, 235]]
    masks += [(top >= level) & (a >= 25) & valid for level in [12, 25, 55, 85]]
    found = []
    for binary in masks:
        n, lab, stats, centres = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            edge = x == 0 or x + w == 448
            if not ((2 if edge else 4) <= area <= 300 and (1 if edge else 2) <= w <= 24 and 2 <= h <= 24):
                continue
            if not ((0.10 if edge else 0.48) <= w / h <= 2.1 and area / (w * h) >= 0.42):
                continue
            cx, cy = centres[i]
            patch = a[max(0,y-5):min(448,y+h+5),max(0,x-5):min(448,x+w+5)]
            peak = float(a[y:y+h,x:x+w][lab[y:y+h,x:x+w] == i].mean())
            bg = float(np.percentile(patch, 25))
            peripheral = cx < 110 or cx > 338
            if peak - bg < (12 if peripheral else 35) or peak < (25 if peripheral else 130):
                continue
            candidate = {'x': float(cx), 'y': float(cy), 'w': int(w), 'h': int(h),
                         'area': int(area), 'peak': peak, 'contrast': peak-bg,
                         'bbox': [int(x),int(y),int(x+w),int(y+h)]}
            near = next((j for j,b in enumerate(found) if math.hypot(cx-b['x'],cy-b['y']) < 5), None)
            if near is None:
                found.append(candidate)
            elif candidate['contrast'] > found[near]['contrast']:
                found[near] = candidate
    return sorted(found, key=lambda b:(b['y'],b['x']))


def ruler_lines(points):
    hypotheses = []
    for i,a in enumerate(points):
        for b in points[i+1:]:
            dy = b['y'] - a['y']
            if not (22 <= dy <= 200) or abs(b['x']-a['x']) > 0.13*dy + 3:
                continue
            slope = (b['x']-a['x']) / dy
            intercept = a['x'] - slope*a['y']
            members = [p for p in points if abs(p['x'] - (intercept+slope*p['y'])) <= 4.0]
            if len(members) < 2:
                continue
            ys = np.array([p['y'] for p in members])
            gaps = np.diff(ys)
            useful = gaps[gaps >= 22]
            if not len(useful):
                continue
            step = float(np.median(useful))
            if not (22 <= step <= 180):
                continue
            chain = []
            for start in members:
                bins = {}
                for p in members:
                    slot=round((p['y']-start['y'])/step)
                    if abs((p['y']-start['y'])/step-slot)>=.13:continue
                    if slot not in bins or p['contrast']>bins[slot]['contrast']:bins[slot]=p
                trial=[bins[k] for k in sorted(bins)]
                if len(trial)>len(chain): chain=trial
            if len(chain)<2: continue
            xmid = intercept+slope*224
            span = chain[-1]['y'] - chain[0]['y']
            if len(chain)==2 and not (xmid<105 or xmid>343): continue
            sizes = np.array([max(p['w'],p['h']) for p in chain])
            if float(sizes.max()/sizes.min()) > 3.2: continue
            fit = np.polyfit([p['y'] for p in chain],[p['x'] for p in chain],1)
            residual = max(abs(p['x']-(float(fit[1])+float(fit[0])*p['y'])) for p in chain)
            if residual>4.5: continue
            high = len(chain)>=3 and span>=55 and residual<=3.5
            hypothesis = {'slope':float(fit[0]),'intercept':float(fit[1]),'x_mid':float(fit[1]+fit[0]*224),
                          'count':len(chain),'span':float(span),'step':step,'residual':float(residual),
                          'confidence':'regular_chain' if high else 'review_chain','points':chain}
            near = next((j for j,h in enumerate(hypotheses) if abs(h['x_mid']-hypothesis['x_mid'])<12), None)
            score = (high,len(chain),-residual)
            if near is None: hypotheses.append(hypothesis)
            else:
                h = hypotheses[near]
                if score>(h['confidence']=='regular_chain',h['count'],-h['residual']):hypotheses[near]=hypothesis
    return sorted(hypotheses,key=lambda h:h['x_mid'])


def detect(gray, valid):
    assert gray.shape==valid.shape==(448,448)
    assert np.isfinite(gray).all() and gray.min()>=0 and gray.max()<=1
    points = blobs(gray,valid)
    lines = ruler_lines(points)
    strong = ruler_lines([p for p in points if p['contrast']>=35 and p['peak']>=100])
    for line in strong:
        if not any(abs(line['x_mid']-q['x_mid'])<1 and line['count']==q['count'] and abs(line['span']-q['span'])<1 for q in lines):
            lines.append(line)
    lines.sort(key=lambda q:q['x_mid'])
    v = gray[valid]
    return {'lines':lines,'blob_count':len(points),'edge_blobs':[b for b in points if b['x']<48 or b['x']>400],
            'gray_metrics':{'valid_fraction':float(valid.mean()),'black_fraction':float((v<=0.01).mean()),
                            'white_fraction':float((v>=0.99).mean()),'p01':float(np.quantile(v,.01)),
                            'p99':float(np.quantile(v,.99))}}


def proposed_stripe(line, valid):
    yy,xx=np.mgrid[:448,:448]
    radius = max(max(p['w'],p['h'])/2 for p in line['points']) + 5
    return (np.abs(xx-(line['intercept']+line['slope']*yy)) <= radius) & valid


def detector_box(item, factor=1.0):
    old=item['detection_geometry']['unclipped_1p3_bbox']
    cx,cy=(old[0]+old[2])/2,(old[1]+old[3])/2
    w,h=(old[2]-old[0])/1.3*factor,(old[3]-old[1])/1.3*factor
    t=item['transform'];b=item['bbox']
    return [(cx-w/2-b[0]+.5)*t['scale_x']+t['offset_x']-.5,
            (cy-h/2-b[1]+.5)*t['scale_y']+t['offset_y']-.5,
            (cx+w/2-b[0]+.5)*t['scale_x']+t['offset_x']-.5,
            (cy+h/2-b[1]+.5)*t['scale_y']+t['offset_y']-.5]


def classify_line(line,item):
    core=detector_box(item)
    x0=line['intercept'];x1=x0+447*line['slope']
    diameter=[max(p['w'],p['h']) for p in line['points']]
    radius=max(diameter)/2
    outside=(min(x0,x1)-radius>core[2]+3) or (max(x0,x1)+radius<core[0]-3)
    expected_side=(line['x_mid']>core[2] if item['clinical_side']=='R' else line['x_mid']<core[0])
    steps=np.diff([p['y'] for p in line['points']])
    step=float(np.median(steps))
    uniform=all(abs(g/step-round(g/step))<=.10 for g in steps)
    size_ratio=max(diameter)/min(diameter)
    relative_contrast=float(np.median([p['contrast']/max(p['peak'],1) for p in line['points']]))
    regular=(line['count']>=3 and line['span']>=115 and 35<=step<=130 and line['residual']<=3.5
             and uniform and size_ratio<=3.3 and relative_contrast>=.35)
    return {'regular':bool(regular),'outside_detector_core':bool(outside),'expected_medial_side':bool(expected_side),
            'likely_ruler':bool(regular and outside),'diameter_ratio':float(size_ratio),'median_step':step,
            'core_box_448':core,'relative_contrast':relative_contrast}


def ruler_mask(lines,item,valid):
    yy,xx=np.mgrid[:448,:448]
    mask=np.zeros((448,448),dtype=bool)
    core=detector_box(item,1.05)
    for line in lines:
        width=max(max(p['w'],p['h'])/2 for p in line['points'])+8
        center=line['intercept']+line['slope']*yy
        if line['x_mid']<core[0]:mask|=(xx<=center+width)
        elif line['x_mid']>core[2]:mask|=(xx>=center-width)
        else:raise ValueError('ruler line lies in the joint core; a local mask must be specified manually')
    mask&=valid
    protected=(xx>=core[0])&(xx<=core[2])&(yy>=core[1])&(yy<=core[3])
    return mask,{'protected_overlap_pixels':int((mask&protected).sum()),'mask_pixels':int(mask.sum()),
                 'mask_fraction':float(mask.mean()),'core_box_448':core}


def coverage_metrics(item,valid):
    core=detector_box(item)
    x0,y0,x1,y1=core
    inside=(x0>=0 and y0>=0 and x1<=447 and y1<=447)
    core_valid=valid[max(0,math.floor(y0)):min(448,math.ceil(y1)+1),max(0,math.floor(x0)):min(448,math.ceil(x1)+1)]
    old=item['parent_1p3_bbox'];new=item['bbox']
    containment=(new[0]<=old[0] and new[1]<=old[1] and new[2]>=old[2] and new[3]>=old[3])
    return {'core_inside_output':bool(inside),'core_valid_fraction':float(core_valid.mean()) if core_valid.size else 0.0,
            'contains_entire_parent_crop':bool(containment),'core_box_448':core,
            'geometry_flags':item['selected_flags'],'parent_flags':item['parent_selected_flags']}


def main():
    cv2.setNumThreads(1)
    reference=json.loads((PROJECT/'experiments/chen_yolo_expand2x_v1/outputs/selection_private.json').read_text())
    outputs=[]
    for r in reference:
        g=np.load(r['gray01_npy'],allow_pickle=False)[0]
        v=np.load(r['valid_mask_npy'],allow_pickle=False)[0]
        d=detect(g,v)
        alias=r.get('sample',r.get('public_alias',''))
        if not alias: alias=Path(r['gray01_npy']).parent.name
        outputs.append({'sample':alias,'result':d})
    save(REVIEW/'pilot_detection_private.json',outputs)
    for i,r in enumerate(outputs):
        print(json.dumps({'sample_number':i+1,'blobs':r['result']['blob_count'],
                          'lines':[{k:h[k] for k in ['x_mid','count','step','confidence']} for h in r['result']['lines']],
                          'edge_blobs':len(r['result']['edge_blobs'])}))


if __name__=='__main__':main()
