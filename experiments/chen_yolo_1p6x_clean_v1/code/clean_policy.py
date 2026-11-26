import cv2
import numpy as np

import audit_rulers as a


def bar_inner_width(line,gray,valid):
    direction=1 if line['x_mid']<224 else -1
    rows=np.arange(30,418)
    center=line['intercept']+line['slope']*rows
    values=[];present=[]
    for distance in range(61):
        cols=np.rint(center+direction*distance).astype(int)
        ok=(cols>=0)&(cols<448)
        values.append(np.where(ok,gray[rows,np.clip(cols,0,447)],np.nan))
        present.append(ok & valid[rows,np.clip(cols,0,447)])
    values=np.array(values);present=np.array(present)
    profile=np.nanmedian(values,axis=1)
    changes=[]
    for d in range(6,48):
        delta=np.nanmedian(values[d-2:d],axis=0)-np.nanmedian(values[d+1:d+3],axis=0)
        ok=np.isfinite(delta)
        support=float((delta[ok]>.025).mean()) if ok.any() else 0
        strength=float(np.median(delta[ok])) if ok.any() else 0
        if support>=.60 and strength>.025:changes.append((strength,d,support))
    basic=max(max(p['w'],p['h'])/2 for p in line['points'])+8
    selected=max(changes,default=None)
    width=max(basic,selected[1]+4) if selected else basic
    return float(width),{'basic_width':basic,'selected_bar_edge':selected,'width':width,
                         'profile_finite':bool(np.isfinite(profile).all())}


def propose(audit,item,gray,valid):
    yy,xx=np.mgrid[:448,:448];mask=np.zeros((448,448),bool);receipts=[]
    lines=[ln for ln in audit['lines'] if ln['classification']['likely_ruler']]
    for ln in lines:
        width,receipt=bar_inner_width(ln,gray,valid)
        center=ln['intercept']+ln['slope']*yy
        if ln['x_mid']<224:mask|=xx<=center+width
        else:mask|=xx>=center-width
        receipts.append({'x_mid':ln['x_mid'],'slope':ln['slope'],'intercept':ln['intercept'],**receipt})
    mask&=valid
    core=a.detector_box(item,1.05)
    protected=(xx>=core[0])&(xx<=core[2])&(yy>=core[1])&(yy<=core[3])
    original=a.detector_box(item,1.3)
    original_area=(xx>=original[0])&(xx<=original[2])&(yy>=original[1])&(yy<=original[3])
    metrics={'mask_pixels':int(mask.sum()),'mask_fraction':float(mask.mean()),
             'core_overlap_pixels':int((mask&protected).sum()),'parent_1p3_overlap_pixels':int((mask&original_area).sum()),
             'bar_edges':receipts}
    return mask,metrics
