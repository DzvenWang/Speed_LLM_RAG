#!/usr/bin/env python3
"""Select one global support-fusion alpha on Validation only.

Primary rule: similarity_top1. The same fixed alpha is then used for every
candidate/query in Train/Test utility generation; no per-candidate oracle tuning.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from supportbench.config import load_config
from supportbench.dataset import SupportBench
from supportbench.utils import ensure_dir, log


def mae(a,b): return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); ap.add_argument('--subgraph',default='TraffiDent_Net_06')
    ap.add_argument('--candidate-pool',required=True); ap.add_argument('--query-predictions',required=True); ap.add_argument('--memory-predictions',required=True)
    ap.add_argument('--grid',nargs='*',type=float,default=None); ap.add_argument('--output',default=None)
    args=ap.parse_args(); cfg=load_config(args.config); b=SupportBench(args.subgraph,config=args.config)
    pool=np.load(args.candidate_pool,allow_pickle=False); qp=np.load(args.query_predictions,allow_pickle=False); mp=np.load(args.memory_predictions,allow_pickle=False)
    pmap={str(v):i for i,v in enumerate(qp['view_id'])}; mmap={int(e):i for i,e in enumerate(mp['episode_id'])}
    splits={str(b.get_view_spec(str(v))['split']) for v in pool['view_id']}
    if splits != {'val'}: raise ValueError(f'Alpha selection must use Validation-only candidate pool; found splits={sorted(splits)}')
    grid=list(map(float,args.grid if args.grid else cfg['support_utility']['alpha_grid']))
    results=[]
    for alpha in grid:
        errs=[]; used=0
        for q,vid in enumerate(pool['view_id']):
            vid=str(vid)
            if vid not in pmap: raise KeyError(f'{vid} missing query prediction')
            scores=pool['retrieval_score'][q]; ids=pool['candidate_episode_id'][q]
            valid=np.flatnonzero((ids>=0)&np.isfinite(scores))
            if not len(valid): continue
            k=int(valid[np.argmax(scores[valid])]); eid=int(ids[k])
            if eid not in mmap: raise KeyError(f'memory episode {eid} missing prediction')
            sample=b.sample(vid); y=sample.future_speed; base=np.asarray(qp['pred'][pmap[vid]],np.float32)
            cy=np.asarray(b.memory_episode(eid)['future_speed'],np.float32); cres=cy-np.asarray(mp['pred'][mmap[eid]],np.float32)
            errs.append(mae(base+alpha*cres,y)); used+=1
        results.append({'alpha':float(alpha),'mean_similarity_top1_mae':float(np.mean(errs)) if errs else None,'queries':int(used)})
    finite=[r for r in results if r['mean_similarity_top1_mae'] is not None]
    if not finite: raise RuntimeError('No valid Validation queries for alpha selection')
    best=min(finite,key=lambda r:r['mean_similarity_top1_mae'])
    out=Path(args.output) if args.output else Path(args.candidate_pool).with_name(Path(args.candidate_pool).stem+'__selected_alpha.json'); ensure_dir(out.parent)
    obj={'mode':'residual','selection_split':'val','selection_rule':'similarity_top1','selected_alpha':float(best['alpha']),'grid_results':results,
         'candidate_pool':str(Path(args.candidate_pool).resolve()),'query_predictions':str(Path(args.query_predictions).resolve()),'memory_predictions':str(Path(args.memory_predictions).resolve())}
    out.write_text(json.dumps(obj,indent=2),encoding='utf-8'); log(f"selected alpha={best['alpha']:.3f}; val similarity-top1 MAE={best['mean_similarity_top1_mae']:.5f}; wrote {out}")
if __name__=='__main__': main()
