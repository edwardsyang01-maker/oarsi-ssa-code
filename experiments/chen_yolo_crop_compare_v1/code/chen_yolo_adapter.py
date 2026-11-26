from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2 if same_padding else 0)
        self.relu = nn.LeakyReLU(.1, inplace=True) if relu else None
    def forward(self, x):
        x = self.conv(x)
        return self.relu(x) if self.relu is not None else x


class Conv2d_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2 if same_padding else 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=.01)
        self.relu = nn.LeakyReLU(.1, inplace=True) if relu else None
    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.relu(x) if self.relu is not None else x


def _make_layers(in_channels, cfg):
    layers = []
    if cfg and isinstance(cfg[0], list):
        for sub in cfg:
            layer, in_channels = _make_layers(in_channels, sub); layers.append(layer)
    else:
        for item in cfg:
            if item == 'M': layers.append(nn.MaxPool2d(2, 2))
            else:
                out_channels, kernel = item
                layers.append(Conv2d_BatchNorm(in_channels, out_channels, kernel, same_padding=True)); in_channels = out_channels
    return nn.Sequential(*layers), in_channels


def match_tensor(x, shape):
    h, w = x.shape[2:]
    dh, dw = shape[0]-h, shape[1]-w
    if dw >= 0: x = F.pad(x, (dw//2, dw-dw//2, 0, 0), mode='reflect')
    else: x = x[:, :, :, (-dw)//2:(-dw)//2+shape[1]]
    if dh >= 0: x = F.pad(x, (0, 0, dh//2, dh-dh//2), mode='reflect')
    else: x = x[:, :, (-dh)//2:(-dh)//2+shape[0], :]
    return x


class Reorg(nn.Module):
    def __init__(self, stride=2): super().__init__(); self.stride = stride
    def forward(self, x, small_size):
        x = match_tensor(x, (2*small_size[0], 2*small_size[1]))
        b,c,h,w=x.shape; s=self.stride
        x=x.contiguous().view(b,c,h//s,s,w//s,s).transpose(3,4).contiguous()
        x=x.view(b,c,h//s*w//s,s*s).transpose(2,3).contiguous()
        x=x.view(b,c,s*s,h//s,w//s).transpose(1,2).contiguous()
        return x.view(b,s*s*c,h//s,w//s)


class Darknet19(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('device_id', torch.zeros(1, dtype=torch.int32))
        cfg=[[(32,3)],['M',(64,3)],['M',(128,3),(64,1),(128,3)],['M',(256,3),(128,1),(256,3)],['M',(512,3),(256,1),(512,3),(256,1),(512,3)],['M',(1024,3),(512,1),(1024,3),(512,1),(1024,3)],[(1024,3),(1024,3)],[(1024,3)]]
        self.conv1s,c1=_make_layers(3,cfg[:5]); self.conv2,c2=_make_layers(c1,cfg[5]); self.conv3,c3=_make_layers(c2,cfg[6])
        self.reorg=Reorg(2); self.conv4,c4=_make_layers(c1*4+c3,cfg[7]); self.conv5=Conv2d(c4,10,1,relu=False)
    def forward(self,x):
        if self.training: raise RuntimeError('Inference-only adapter')
        a=self.conv1s(x); b=self.conv3(self.conv2(a)); out=self.conv5(self.conv4(torch.cat([self.reorg(a,b.shape[2:]),b],1)))
        b,c,h,w=out.shape
        out=out.permute(0,2,3,1).contiguous().view(b,h*w,1,10)
        boxes=torch.cat([out[:,:,:,:2].sigmoid(),out[:,:,:,2:4].exp()],3)
        iou=out[:,:,:,4:5].sigmoid(); prob=out[:,:,:,5:].softmax(-1)
        return boxes,iou,prob


def load_detector(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not all(isinstance(v,torch.Tensor) for v in state.values()): raise RuntimeError('Expected tensor state_dict')
    model=Darknet19().eval(); report=model.load_state_dict(state,strict=True)
    for p in model.parameters():p.requires_grad_(False)
    return model,{'state_entries':len(state),'missing':list(report.missing_keys),'unexpected':list(report.unexpected_keys),'out_channels':int(state['conv5.conv.weight'].shape[0])}


def bbox_iou(a,b,inclusive=False):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float); extra=1. if inclusive else 0.
    wh=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])+extra)
    inter=float(np.prod(wh)); aa=float(np.prod(np.maximum(0,a[2:]-a[:2]+extra)));bb=float(np.prod(np.maximum(0,b[2:]-b[:2]+extra)))
    return inter/max(aa+bb-inter,1e-12)


def decode(pred, shape=(256,320), threshold=.12, nms_threshold=.3):
    boxes,iou,prob=[v.detach().cpu().numpy() for v in pred]
    boxes=boxes.reshape(-1,4).astype(np.float64); grid_h=shape[0]//32;grid_w=shape[1]//32
    rows,cols=np.divmod(np.arange(len(boxes)),grid_w)
    cx=(boxes[:,0]+cols)*(shape[1]/grid_w);cy=(boxes[:,1]+rows)*(shape[0]/grid_h)
    hw=boxes[:,2]*67.05/2;hh=boxes[:,3]*65.61/2
    xyxy=np.stack([cx-hw,cy-hh,cx+hw,cy+hh],axis=1).astype(np.int64)
    scores=iou.reshape(-1)*prob.reshape(len(boxes),-1).max(axis=1)
    inds=np.flatnonzero(scores>=threshold);order=inds[np.argsort(scores[inds])[::-1]];keep=[]
    while len(order):
        first=int(order[0]);keep.append(first)
        order=np.array([j for j in order[1:] if bbox_iou(xyxy[first],xyxy[j],True)<=nms_threshold],dtype=int)
    out=[]
    for i in keep:
        box=xyxy[i].copy();box[[0,2]]=np.clip(box[[0,2]],0,shape[1]-1);box[[1,3]]=np.clip(box[[1,3]],0,shape[0]-1)
        out.append({'bbox':box.tolist(),'score':float(scores[i])})
    return out


def validate_author_forward(model, source_root, tensor):
    path=Path(source_root)/'DetJoint/yolo_v2/darknet.py';tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Darknet19')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    fn.name='author_forward';module=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(module)
    ns={'torch':torch,'F':F,'nn':nn};exec(compile(module,str(path),'exec'),ns)
    model.cfg=SimpleNamespace(num_anchors=1,num_classes=5)
    with torch.inference_mode():
        expected=ns['author_forward'](model,tensor);actual=model(tensor)
    errors=[float((a-b).abs().max()) for a,b in zip(actual,expected)]
    if max(errors)>1e-6:raise RuntimeError('Author forward equivalence failed')
    return {'method':'reviewed_author_forward_same_layers_and_weights','max_abs_errors':errors,'passed':True}
