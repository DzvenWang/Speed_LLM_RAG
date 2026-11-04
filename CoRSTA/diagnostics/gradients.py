"""Read-only gradient conflict probe on a small deterministic TRAIN subset.

This does not train, perform optim.step(), or write checkpoints.
"""
from __future__ import annotations
import math
import torch
from corsta.v7.losses import cfrm_auxiliary

GROUPS=('stla_tail','stla_head','c1','c2','bridge_fusion')

def groups(model):
    all_layers=list(model.student.base.attn_layers_st)
    result={
        'stla_tail':list(p for m in all_layers[-model.student.tail_layers:] for p in m.parameters() if p.requires_grad),
        'stla_head':list(p for p in model.student.base.output_proj.parameters() if p.requires_grad),
        'c1':[p for p in model.c1.parameters() if p.requires_grad],
        'c2':[p for p in model.c2.parameters() if p.requires_grad],
        'bridge_fusion':[p for name,p in model.named_parameters() if p.requires_grad and name.startswith(('support.','bridge1.','bridge2.','fusion.'))],
    }
    return {k:v for k,v in result.items() if v}

def _dot(g,h):
    return sum(float((x.detach().float()*y.detach().float()).sum()) for x,y in zip(g,h) if x is not None and y is not None)

def _norm(g):return math.sqrt(sum(float(x.detach().float().square().sum()) for x in g if x is not None))

def one_batch(model,b,mean,std):
    """Four unweighted objectives; use unweighted gradient norms to inspect scales."""
    o=model(b);valid=b['future_valid'].float();den=valid.sum().clamp_min(1)
    target=b['y_raw'].float();pred=o['refined'].float()*std+mean
    refined=(((pred-target).abs()*valid).sum()/den+0.05*torch.sqrt(((pred-target).square()*valid).sum()/den+1e-8))
    base=((o['backbone'].float()*std+mean-target).abs()*valid).sum()/den
    anchor=((o['backbone'].float()-b['cached_base'].float()).abs()*std*valid).sum()/den
    aux=cfrm_auxiliary(o['legacy'],b,mean,std,wrong_pair=('cf_wrong_increment' in o['legacy']))
    objectives={'refined':refined,'base':base,'anchor':anchor,'c1_aux':aux}
    ps=groups(model);all_params=[];labels=[]
    for k,v in ps.items():labels.append((k,len(all_params),len(v)));all_params+=v
    gradient={}
    for i,(key,loss) in enumerate(objectives.items()):
        gradient[key]=torch.autograd.grad(loss,all_params,allow_unused=True,retain_graph=(i!=len(objectives)-1))
    result={}
    for name,start,length in labels:
        d={};sl=slice(start,start+length)
        for k,g in gradient.items():d[f'{k}_grad_norm']=_norm(g[sl])
        for a,c in (('refined','base'),('refined','anchor'),('refined','c1_aux')):
            ga=gradient[a][sl];gb=gradient[c][sl];na=_norm(ga);nb=_norm(gb)
            d[f'cos_{a}_{c}']=_dot(ga,gb)/(na*nb) if na>1e-12 and nb>1e-12 else None
        result[name]=d
    result['losses']={k:float(v.detach()) for k,v in objectives.items()}
    return result

class GradientAudit:
    def __init__(self):self.samples=[]
    def add(self,x):self.samples.append(x)
    def to_dict(self):
        if not self.samples:return {'num_batches':0}
        keys=set().union(*(set(a) for a in self.samples));out={}
        for k in keys:
            names=set().union(*(set(s.get(k,{})) for s in self.samples))
            out[k]={name:sum(float(s[k][name]) for s in self.samples if k in s and s[k].get(name) is not None)/
                max(1,sum(1 for s in self.samples if k in s and s[k].get(name) is not None))
                for name in names}
        return {'num_batches':len(self.samples),'mean_metrics':out,
                'per_batch':self.samples,
                'note':'Diagnostic-only gradients at fixed checkpoints; no optimizer steps.'}
