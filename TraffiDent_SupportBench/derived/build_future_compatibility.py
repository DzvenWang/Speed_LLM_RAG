#!/usr/bin/env python3
"""Compute model-independent historical-future compatibility for a fixed candidate pool."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from supportbench.config import load_config
from supportbench.dataset import SupportBench
from supportbench.utils import ensure_dir, log


def normalize_per_node(y: np.ndarray, eps=1e-6) -> np.ndarray:
    y=np.asarray(y,np.float32); mu=y.mean(axis=0,keepdims=True); sd=y.std(axis=0,keepdims=True)
    return (y-mu)/np.maximum(sd,eps)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); ap.add_argument('--subgraph',default='TraffiDent_Net_06'); ap.add_argument('--candidate-pool',required=True); ap.add_argument('--output',default=None)
    args=ap.parse_args(); cfg=load_config(args.config); b=SupportBench(args.subgraph,config=args.config); pool=np.load(args.candidate_pool,allow_pickle=False)
    vids=pool['view_id']; ids=pool['candidate_episode_id']; Q,K=ids.shape
    dist=np.full((Q,K),np.nan,np.float32); trend_match=np.zeros((Q,K),np.float32)
    for q in range(Q):
        s=b.sample(str(vids[q])); yq=normalize_per_node(s.future_speed)
        qtrend=np.sign(s.future_speed[-1].mean()-s.future_speed[0].mean())
        for k in range(K):
            eid=int(ids[q,k]);
            if eid<0: continue
            yc=b.memory_episode(eid)['future_speed']; yn=normalize_per_node(yc)
            dist[q,k]=float(np.mean(np.abs(yq-yn)))
            ctrend=np.sign(yc[-1].mean()-yc[0].mean()); trend_match[q,k]=float(qtrend==ctrend)
    out=Path(args.output) if args.output else Path(args.candidate_pool).with_name(Path(args.candidate_pool).stem+'__future_compat.npz')
    ensure_dir(out.parent); np.savez_compressed(out,view_id=vids,candidate_episode_id=ids,future_distance=dist,future_trend_match=trend_match)
    log(f'wrote {out}')
if __name__=='__main__': main()
