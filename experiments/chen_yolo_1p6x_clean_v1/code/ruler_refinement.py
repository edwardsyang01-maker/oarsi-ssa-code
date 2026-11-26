import math
import cv2
import numpy as np

import audit_rulers as a


def periodic_lines(points):
    pts=[p for p in points if (p['x']<150 or p['x']>298) and p['contrast']/max(p['peak'],1)>=.35]
    best=[]
    for i,p in enumerate(pts):
        for q in pts[i+1:]:
            dy=q['y']-p['y']
            if dy<30 or dy>300:continue
            slope=(q['x']-p['x'])/dy
            if abs(slope)>.14:continue
            intercept=p['x']-slope*p['y']
            members=[r for r in pts if abs(r['x']-intercept-slope*r['y'])<=4.0]
            if len(members)<3:continue
            for k in [1,2,3]:
                step=dy/k
                if not 35<=step<=130:continue
                bins={}
                for r in members:
                    phase=(r['y']-p['y'])/step;slot=round(phase)
                    if abs(phase-slot)>.09:continue
                    score=(r['contrast']/max(r['peak'],1),-abs(phase-slot))
                    if slot not in bins or score>bins[slot][0]:bins[slot]=(score,r)
                chain=[bins[z][1] for z in sorted(bins)]
                if len(chain)<3 or chain[-1]['y']-chain[0]['y']<110:continue
                fit=np.polyfit([r['y'] for r in chain],[r['x'] for r in chain],1)
                residual=max(abs(r['x']-fit[1]-fit[0]*r['y']) for r in chain)
                if residual>3.5:continue
                sizes=np.array([max(r['w'],r['h']) for r in chain]);ratio=float(np.quantile(sizes,.75)/max(np.quantile(sizes,.25),1))
                if ratio>2.5:continue
                out={'slope':float(fit[0]),'intercept':float(fit[1]),'x_mid':float(fit[1]+224*fit[0]),
                     'count':len(chain),'span':float(chain[-1]['y']-chain[0]['y']),'step':float(step),
                     'residual':float(residual),'points':chain,'confidence':'refined_periodic_chain',
                     'robust_diameter_ratio':ratio}
                near=next((j for j,b in enumerate(best) if abs(b['x_mid']-out['x_mid'])<10),None)
                score=(len(chain),out['span'],-residual)
                if near is None:best.append(out)
                elif score>(best[near]['count'],best[near]['span'],-best[near]['residual']):best[near]=out
    return sorted(best,key=lambda l:l['x_mid'])


def boundary_points(gray,valid):
    from scipy.ndimage import median_filter
    local=median_filter(gray,size=(25,1),mode='nearest')
    contrast=gray-local
    edge=np.zeros(gray.shape,bool);edge[:,:96]=True;edge[:,-96:]=True
    binary=(edge & valid & (contrast>=.03) & (contrast/np.maximum(gray,.01)>=.25)).astype('uint8')
    count,labels,stats,centers=cv2.connectedComponentsWithStats(binary,8)
    result=[]
    for k in range(1,count):
        x,y,w,h,area=map(int,stats[k])
        if not (1<=area<=160 and 1<=w<=16 and 1<=h<=18):continue
        chosen=labels[y:y+h,x:x+w]==k
        peak=float(gray[y:y+h,x:x+w][chosen].max()*255)
        diff=float(contrast[y:y+h,x:x+w][chosen].max()*255)
        result.append({'x':float(centers[k][0]),'y':float(centers[k][1]),'w':w,'h':h,
                       'area':area,'peak':peak,'contrast':diff,'bbox':[x,y,x+w,y+h],
                       'edge_partial_channel':True})
    return result


def detect(gray,valid,item):
    points=a.blobs(gray,valid)
    for p in boundary_points(gray,valid):
        if not any((p['x']-q['x'])**2+(p['y']-q['y'])**2<16 for q in points):points.append(p)
    points.sort(key=lambda p:p['y'])
    lines=periodic_lines(points)
    for line in periodic_lines([p for p in points if p['peak']>=75 and p['contrast']>=30]):
        if not any(abs(line['x_mid']-q['x_mid'])<1 and line['count']==q['count'] and abs(line['span']-q['span'])<1 for q in lines):lines.append(line)
    core=a.detector_box(item)
    for line in lines:
        radius=max(max(p['w'],p['h'])/2 for p in line['points'])
        x0=line['intercept'];x1=x0+447*line['slope']
        outside=min(x0,x1)-radius>core[2]+3 or max(x0,x1)+radius<core[0]-3
        line['classification']={'regular':True,'likely_ruler':bool(outside),'outside_detector_core':bool(outside),
                                'expected_medial_side':bool(line['x_mid']>core[2] if item['clinical_side']=='R' else line['x_mid']<core[0]),
                                'core_box_448':core,'method':'period_and_edge_refinement_v1'}
    return lines


def native_point(item,x,y):
    t=item['transform'];b=item['bbox']
    return b[0]+(x-t['offset_x']+.5)/t['scale_x']-.5,b[1]+(y-t['offset_y']+.5)/t['scale_y']-.5


def image_point(item,x,y):
    t=item['transform'];b=item['bbox']
    return (x-b[0]+.5)*t['scale_x']+t['offset_x']-.5,(y-b[1]+.5)*t['scale_y']+t['offset_y']-.5


def pair_context(line,item,entries):
    centers=[]
    for r in entries:
        b=r['detection_geometry']['unclipped_1p3_bbox'];centers.append(((b[0]+b[2])/2,(b[1]+b[3])/2))
    if len(centers)!=2:return {'available':False,'central_between_knees':False}
    middle_x=float(np.mean([c[0] for c in centers]));middle_y=float(np.mean([c[1] for c in centers]));distance=abs(centers[1][0]-centers[0][0])
    _,yp=image_point(item,middle_x,middle_y)
    nx,_=native_point(item,line['intercept']+line['slope']*yp,yp)
    offset=(nx-middle_x)/distance
    return {'available':True,'center_offset_relative_knee_separation':float(offset),'central_between_knees':bool(abs(offset)<=.20)}


def project(line,source,target):
    points=[];sx=target['transform']['scale_x']/source['transform']['scale_x'];sy=target['transform']['scale_y']/source['transform']['scale_y']
    for p in line['points']:
        nx,ny=native_point(source,p['x'],p['y']);x,y=image_point(target,nx,ny)
        points.append({**p,'x':x,'y':y,'w':max(1,p['w']*sx),'h':max(1,p['h']*sy)})
    fit=np.polyfit([p['y'] for p in points],[p['x'] for p in points],1)
    return {**line,'points':points,'slope':float(fit[0]),'intercept':float(fit[1]),'x_mid':float(fit[1]+224*fit[0]),
            'step':line['step']*sy,'span':line['span']*sy,'projected_from_same_film':True,'source_side':source['clinical_side']}
