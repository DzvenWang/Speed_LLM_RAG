from __future__ import annotations
import numpy as np


class MetricAccumulator:
    def __init__(self,horizon):
        self.H=int(horizon); self.s={k:{'abs':0.0,'sq':0.0,'n':0} for k in ['overall','masked','support']}
        self.h_abs=np.zeros(self.H,np.float64); self.h_n=np.zeros(self.H,np.int64)
    def update(self,pred,y,valid,direct,support,row_mask=None):
        p=np.asarray(pred,np.float32); yy=np.asarray(y,np.float32); vv=np.asarray(valid,bool); d=np.asarray(direct,bool); s=np.asarray(support,bool)
        if row_mask is not None:
            r=np.asarray(row_mask,bool); p=p[r]; yy=yy[r]; vv=vv[r]; d=d[r]; s=s[r]
        if len(p)==0: return
        err=p-yy; ae=np.abs(err); sq=err*err; dm=d[:,None,:]; sm=s[:,None,:]
        for name,m in [('overall',vv),('masked',vv&dm),('support',vv&sm)]:
            finite=m&np.isfinite(ae)
            self.s[name]['abs']+=float(ae[finite].sum(dtype=np.float64)); self.s[name]['sq']+=float(sq[finite].sum(dtype=np.float64)); self.s[name]['n']+=int(finite.sum())
        for h in range(min(self.H,p.shape[1])):
            m=vv[:,h]&d&np.isfinite(ae[:,h]); self.h_abs[h]+=float(ae[:,h][m].sum(dtype=np.float64)); self.h_n[h]+=int(m.sum())
    def summary(self):
        out={}
        for name,z in self.s.items():
            if z['n']<=0:
                out[f'{name}_mae']=float('nan'); out[f'{name}_rmse']=float('nan'); out[f'{name}_count']=0
            else:
                n=z['n']; out[f'{name}_mae']=z['abs']/n; out[f'{name}_rmse']=float(np.sqrt(z['sq']/n)); out[f'{name}_count']=int(n)
        out['horizon_masked_mae']=[float(self.h_abs[h]/self.h_n[h]) if self.h_n[h] else float('nan') for h in range(self.H)]
        return out


def improvement_pct(base,model):
    if not (np.isfinite(base) and np.isfinite(model)) or base==0: return float('nan')
    return float((base-model)/base*100.0)


def per_query_masked_mae(pred,y,valid,direct):
    p=np.asarray(pred,np.float32); yy=np.asarray(y,np.float32); vv=np.asarray(valid,bool); d=np.asarray(direct,bool)[:,None,:]
    ae=np.abs(p-yy); m=vv&d&np.isfinite(ae); num=np.where(m,ae,0.0).sum(axis=(1,2),dtype=np.float64); den=m.sum(axis=(1,2))
    return np.divide(num,den,out=np.full(len(p),np.nan,np.float64),where=den>0)


def latent_query_stats(h0,h1,direct=None,support=None):
    a=np.asarray(h0,np.float32); b=np.asarray(h1,np.float32)
    if a.ndim==4 and a.shape[1]==1: a=a[:,0]
    if b.ndim==4 and b.shape[1]==1: b=b[:,0]
    diff=b-a
    rel=np.linalg.norm(diff.reshape(len(a),-1),axis=1)/(np.linalg.norm(b.reshape(len(b),-1),axis=1)+1e-8)
    aa=a.reshape(len(a),-1); bb=b.reshape(len(b),-1); cos=(aa*bb).sum(1)/(np.linalg.norm(aa,axis=1)*np.linalg.norm(bb,axis=1)+1e-8)
    out={'rel_l2':rel,'cosine':cos}
    for name,mask in [('direct',direct),('support',support)]:
        if mask is None: continue
        m=np.asarray(mask,bool); vals=[]
        for i in range(len(a)):
            if not np.any(m[i]): vals.append(np.nan); continue
            da=diff[i,m[i]]; rb=b[i,m[i]]
            vals.append(float(np.linalg.norm(da)/(np.linalg.norm(rb)+1e-8)))
        out[f'{name}_rel_l2']=np.asarray(vals,np.float64)
    return out
