from __future__ import annotations
import numpy as np


class MetricAccumulator:
    def __init__(self, horizon):
        self.H = int(horizon)
        self.s = {k: {"abs":0.0,"sq":0.0,"n":0} for k in ["overall","masked","support"]}
        self.h_abs = np.zeros(self.H, np.float64); self.h_n = np.zeros(self.H, np.int64)
    def update(self, pred, y, valid, direct, support):
        p=np.asarray(pred,np.float32); yy=np.asarray(y,np.float32); vv=np.asarray(valid,bool)
        d=np.asarray(direct,bool); s=np.asarray(support,bool)
        if p.ndim==2: p=p[None]
        if yy.ndim==2: yy=yy[None]
        if vv.ndim==2: vv=vv[None]
        if d.ndim==1: d=d[None]
        if s.ndim==1: s=s[None]
        err=p-yy; ae=np.abs(err); sq=err*err; dm=d[:,None,:]; sm=s[:,None,:]
        for name,m in [("overall",vv),("masked",vv&dm),("support",vv&sm)]:
            finite=m&np.isfinite(ae)
            self.s[name]["abs"]+=float(ae[finite].sum(dtype=np.float64)); self.s[name]["sq"]+=float(sq[finite].sum(dtype=np.float64)); self.s[name]["n"]+=int(finite.sum())
        for h in range(min(self.H,p.shape[1])):
            m=vv[:,h]&d&np.isfinite(ae[:,h]); self.h_abs[h]+=float(ae[:,h][m].sum(dtype=np.float64)); self.h_n[h]+=int(m.sum())
    def summary(self):
        out={}
        for name,z in self.s.items():
            if z["n"]<=0:
                out[f"{name}_mae"]=float("nan"); out[f"{name}_rmse"]=float("nan"); out[f"{name}_count"]=0
            else:
                n=z["n"]; out[f"{name}_mae"]=z["abs"]/n; out[f"{name}_rmse"]=float(np.sqrt(z["sq"]/n)); out[f"{name}_count"]=int(n)
        out["horizon_masked_mae"]=[float(self.h_abs[h]/self.h_n[h]) if self.h_n[h] else float("nan") for h in range(self.H)]
        return out


def improvement_pct(base, model):
    if not (np.isfinite(base) and np.isfinite(model)) or base == 0: return float("nan")
    return float((base-model)/base*100.0)


def query_masked_mae(pred,y,valid,direct):
    p=np.asarray(pred,np.float32); yy=np.asarray(y,np.float32); vv=np.asarray(valid,bool); d=np.asarray(direct,bool)[None,:]
    ae=np.abs(p-yy); m=vv&d&np.isfinite(ae)
    return float(ae[m].mean()) if np.any(m) else float("nan")
