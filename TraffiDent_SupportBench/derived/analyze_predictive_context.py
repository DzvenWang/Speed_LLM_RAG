#!/usr/bin/env python3
"""Compare general vs predictive structured masks using Train-only dependency context.

This diagnostic is future-free.  It never reads query future values or model
predictions; it measures how much Train-defined predictive-neighbour context is
removed by each fixed sensing layout on matched query windows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from supportbench.config import load_config
from supportbench.dataset import SupportBench
from supportbench.dependency import dependency_context_loss
from supportbench.sensing import materialize_mask
from supportbench.utils import ensure_dir, log


def _rows(bench: SupportBench, split: str, family: str, severity: float, budget: str) -> pd.DataFrame:
    legal=set(map(int,bench.window_starts(split,budget)))
    d=bench.views[(bench.views.split==split)&(bench.views.family==family)&(bench.views.distribution=='id')&np.isclose(bench.views.severity.astype(float),float(severity))&(bench.views.instance.astype(int)==0)].copy()
    return d[d.window_start.astype(int).isin(legal)].reset_index(drop=True)


def _loss(bench: SupportBench, row: pd.Series) -> float:
    spec=row.to_dict(); s=int(spec['window_start'])
    valid=np.asarray(bench.valid[s:s+bench.H,:,bench.speed_idx],dtype=bool)
    mask=materialize_mask(spec,bench.H,bench.N,bench.sg_dir,valid)
    return float(np.mean(dependency_context_loss(mask,bench.dependency_ids,bench.dependency_weights)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default=None)
    ap.add_argument('--subgraph',default='TraffiDent_Net_06')
    ap.add_argument('--split',default='test',choices=['val','test'])
    ap.add_argument('--budget',default='development',choices=['development','full'])
    ap.add_argument('--severity',type=float,default=None)
    ap.add_argument('--max-views',type=int,default=0)
    ap.add_argument('--output',default=None)
    args=ap.parse_args()

    cfg=load_config(args.config); bench=SupportBench(args.subgraph,config=args.config)
    sev=float(args.severity if args.severity is not None else cfg['sensing']['main_severity'])
    pairs=[('regional_outage','predictive_regional_outage'),('directional_outage','predictive_directional_outage')]
    rows=[]
    for general,predictive in pairs:
        gd=_rows(bench,args.split,general,sev,args.budget)
        pdv=_rows(bench,args.split,predictive,sev,args.budget)
        gm={int(r.window_start):r for _,r in gd.iterrows()}; pm={int(r.window_start):r for _,r in pdv.iterrows()}
        common=sorted(set(gm)&set(pm))
        if args.max_views>0 and len(common)>args.max_views:
            pos=np.linspace(0,len(common)-1,num=args.max_views,dtype=int); common=[common[int(i)] for i in pos]
        gl=[]; pl=[]
        for i,ws in enumerate(common):
            gl.append(_loss(bench,gm[ws])); pl.append(_loss(bench,pm[ws]))
            if (i+1)%500==0: log(f'{general} vs {predictive}: {i+1:,}/{len(common):,}')
        gmean=float(np.mean(gl)) if gl else np.nan; pmean=float(np.mean(pl)) if pl else np.nan
        ratio=float(pmean/max(gmean,1e-12)) if np.isfinite(gmean) and np.isfinite(pmean) else np.nan
        rows.append({'general_family':general,'predictive_family':predictive,'severity':sev,'split':args.split,'budget':args.budget,'n_matched_views':len(common),'general_context_loss_mean':gmean,'predictive_context_loss_mean':pmean,'predictive_to_general_ratio':ratio,'absolute_gain':pmean-gmean if np.isfinite(gmean) and np.isfinite(pmean) else np.nan})
    out=pd.DataFrame(rows)
    if args.output:
        path=Path(args.output)
    else:
        path=Path(cfg['project']['derived_root'])/args.subgraph/'diagnostics'/f'predictive_context_{args.split}_{args.budget}.csv'
    ensure_dir(path.parent); out.to_csv(path,index=False)
    print(out.to_string(index=False))
    log(f'wrote {path}')


if __name__=='__main__':
    main()
