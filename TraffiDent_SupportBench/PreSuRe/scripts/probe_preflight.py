#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,torch
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe');ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml
from presure.contract import assert_cache_ready
from presure.pipeline import load_artifacts

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development',choices=['development']);a=ap.parse_args();cfg=load_yaml(ROOT/'configs/presure.yaml');ident=assert_cache_ready(cfg,a.stage);art=load_artifacts(cfg,a.stage)
    req=[art/'train_node_embeddings_v05.npy',art/'val_node_embeddings_v05.npy']
    miss=[str(x) for x in req if not x.exists()]
    if miss: raise RuntimeError('Predictive Need Probe requires completed v0.7 M3 embeddings: '+', '.join(miss))
    for v,typ in [('M1','node_sensing_m1_v04'),('M3','node_usefulness_v05')]:
        lp=Path(cfg['project']['exp_root'])/v/cfg['experiment']['subgraph']/a.stage/'LATEST.json'
        if not lp.exists(): raise RuntimeError(f'{v} LATEST.json missing; complete v0.7 Development first')
        j=json.loads(lp.read_text());cp=Path(j['checkpoint'])
        if not cp.exists(): raise RuntimeError(f'{v} checkpoint missing: {cp}')
        ck=torch.load(cp,map_location='cpu',weights_only=False)
        if ck.get('model_type')!=typ: raise RuntimeError(f'{v} checkpoint type {ck.get("model_type")} != {typ}; rerun v0.7 {v}')
    print('PASS PreSuRe v'+str(cfg['version']))
    print('PASS probe role',cfg['predictive_need_probe']['role'])
    print('PASS cache_fingerprint',ident['cache_fingerprint'])
    print('PASS reuse artifacts',art)
    print('PASS M1/M3 checkpoints and cached embeddings')
    print('PREDICTIVE NEED PROBE PREFLIGHT PASS')
if __name__=='__main__':main()
