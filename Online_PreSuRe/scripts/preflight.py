#!/usr/bin/env python3
from pathlib import Path
import argparse,copy,sys,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from online_presure.config import load_config
from online_presure.contract import assert_cache_ready,cache_dir
from online_presure.backbone import load_patchstg
from online_presure.data import OnlineCache
from online_presure.updater import configure_trainable
from online_presure.evidence import node_shift_alignment
from online_presure.utils import log,seed_all

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--split',default=None); ap.add_argument('--project-root',default=None); ap.add_argument('--checkpoint',default=None)
    a=ap.parse_args(); cfg=load_config(ROOT,a.project_root); seed_all(cfg['experiment']['seed'])
    if a.split: cfg['experiment']['stream_split']=a.split
    ident=assert_cache_ready(cfg,a.stage,a.split); log(f"CACHE PASS dir={cache_dir(cfg,a.stage)} fingerprint={ident.get('cache_fingerprint')}")
    ds=OnlineCache(cfg,a.stage,a.split); starts=ds.starts[ds.order]; log(f"STREAM PASS split={ds.split} queries={len(ds)} start=[{starts[0]},{starts[-1]}]")
    r=ds.item(ds.cache_index(0)); log(f"INCOMPLETE PASS x={tuple(r['x_obs'].shape)} history_observed={float(r['history_observed'].float().mean()):.4f} artificial_missing={float(r['artificial_missing'].float().mean()):.4f}")
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); model,path,ck,source=load_patchstg(cfg,a.stage,device,a.checkpoint)
    m=copy.deepcopy(model); ps=configure_trainable(m,cfg['adaptation']['trainable']); log(f"SLOW PATH trainable_parameters={sum(p.numel() for p in ps)} lr={cfg['adaptation']['learning_rate']}")
    log(f"EVIDENCE MEMORY capacity={cfg['evidence_memory']['capacity']} global_top_k={cfg['retrieval']['top_k']} prefilter_k={cfg['retrieval']['prefilter_k']} candidate_pool_k={cfg['compositional']['candidate_pool_k']}")
    log(f"COMPOSITIONAL unit={cfg['compositional']['unit']} patch_size={cfg['utility']['temporal_patch_size']} top_k_per_patch={cfg['utility']['top_k_per_patch']} fixed_completion_gate={cfg['completion']['max_gate']}")
    log(f"UTILITY scorer hidden={cfg['utility']['hidden_dims']} probes_per_query={cfg['utility']['probes_per_query']} min_examples={cfg['utility']['min_examples']} thresholds=[{cfg['utility']['harmful_threshold_pct']},{cfg['utility']['helpful_threshold_pct']}]")
    log(f"ORACLES perfect_completion={cfg['diagnostics']['perfect_completion_oracle']} entry_ks={cfg['diagnostics']['entry_oracle_ks']}")
    log(f"GLOBAL RETRIEVAL weights latent={cfg['retrieval']['latent_weight']} observed={cfg['retrieval']['observed_match_weight']} complementarity={cfg['retrieval']['complementarity_weight']} recency={cfg['retrieval']['recency_weight']}")
    x=np.zeros((3,2),np.float32); o=np.array([[1,1],[1,0],[0,1]],bool); sh=node_shift_alignment(x,o,x,o,1); assert sh.shape==(2,)
    if len(ds)>=2:
        i0=ds.cache_index(0); i1=ds.cache_index(1); log(f"CAUSAL SANITY first_future_end={ds.future_end(i0)} second_origin={ds.origin(i1)} mature_at_second={ds.future_end(i0)<=ds.origin(i1)}")
    log('PREFLIGHT PASS')
if __name__=='__main__': main()
