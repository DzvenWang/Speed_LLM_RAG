#!/usr/bin/env python3
"""Evaluate forecasting predictions with all-node, masked and support-dependent metrics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from supportbench.dataset import SupportBench
from supportbench.utils import ensure_dir, write_json, log


def _mae(p,y,m):
    m=np.asarray(m,bool)&np.isfinite(p)&np.isfinite(y)
    return float(np.mean(np.abs(p[m]-y[m]))) if np.any(m) else None

def _rmse(p,y,m):
    m=np.asarray(m,bool)&np.isfinite(p)&np.isfinite(y)
    return float(np.sqrt(np.mean((p[m]-y[m])**2))) if np.any(m) else None


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); ap.add_argument('--subgraph',default='TraffiDent_Net_06'); ap.add_argument('--predictions',required=True); ap.add_argument('--protocol',default=None); ap.add_argument('--output',default=None)
    args=ap.parse_args(); b=SupportBench(args.subgraph,config=args.config); z=np.load(args.predictions,allow_pickle=False)
    vids=list(map(str,z['view_id'])); pred=np.asarray(z['pred'],np.float32)
    if args.protocol:
        wanted=list(map(str,pd.read_csv(args.protocol)['view_id'])); pos={v:i for i,v in enumerate(vids)}; idx=[pos[v] for v in wanted if v in pos]; vids=[vids[i] for i in idx]; pred=pred[idx]
    rows=[]
    for i,vid in enumerate(vids):
        s=b.sample(vid); y=np.asarray(s.future_speed,np.float32); p=np.asarray(pred[i],np.float32)
        if p.shape!=y.shape: raise ValueError(f'{vid}: pred {p.shape} != {y.shape}')
        valid=np.asarray(s.raw_valid_future,bool); direct=np.asarray(s.artificial_mask,bool).any(axis=0); cm=b.predictive_context_metrics(s); support=np.asarray(cm['support_dependent_target_mask'],bool)
        allm=valid; dm=valid & direct[None,:]; sm=valid & support[None,:]
        r={'view_id':vid,'family':str(s.view_spec['family']),'severity':float(s.view_spec['severity']),'distribution':str(s.view_spec.get('distribution','id')),
           'MAE':_mae(p,y,allm),'RMSE':_rmse(p,y,allm),'directly_masked_MAE':_mae(p,y,dm),'support_dependent_MAE':_mae(p,y,sm),
           'support_dependent_fraction':float(support.mean()),
           'predictive_context_loss':float(cm['network_mean_context_loss']),
           'network_mean_context_loss':float(cm['network_mean_context_loss']),
           'legacy_context_loss_quantile':float(cm['context_loss_quantile_value'])}
        for h in [3,6,12]:
            if h<=b.P:
                r[f'H{h}_MAE']=_mae(p[h-1:h],y[h-1:h],valid[h-1:h])
                r[f'H{h}_support_dependent_MAE']=_mae(p[h-1:h],y[h-1:h],sm[h-1:h])
        rows.append(r)
    df=pd.DataFrame(rows); out=Path(args.output) if args.output else Path(args.predictions).with_name(Path(args.predictions).stem+'__metrics.csv'); ensure_dir(out.parent); df.to_csv(out,index=False)
    numeric=[c for c in df.columns if c not in {'view_id','family','distribution'} and pd.api.types.is_numeric_dtype(df[c])]
    summary={'queries':int(len(df)),'overall':{c:float(pd.to_numeric(df[c],errors='coerce').mean()) for c in numeric},'by_family':{}}
    for fam,g in df.groupby('family'):
        summary['by_family'][str(fam)]={c:float(pd.to_numeric(g[c],errors='coerce').mean()) for c in numeric}
    write_json(out.with_suffix('.summary.json'),summary); log(f'wrote {out}; queries={len(df)}')
if __name__=='__main__': main()
