#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,json,torch,numpy as np
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe'); ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml
from presure.contract import assert_cache_ready,cache_dir
from presure.backbone import resolve_checkpoint

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--checkpoint',default=None); a=ap.parse_args()
 cfg=load_yaml(ROOT/'configs/presure.yaml'); ident=assert_cache_ready(cfg,a.stage); p,ck,src=resolve_checkpoint(cfg,a.stage,a.checkpoint)
 c=cache_dir(cfg,a.stage); tr=np.load(c/'train_x_obs_norm.npy',mmap_mode='r'); va=np.load(c/'val_x_obs_norm.npy',mmap_mode='r')
 for d in [Path(cfg['project']['exp_root']),Path(cfg['project']['log_root']),Path(cfg['project']['log_root'])/'out']: d.mkdir(parents=True,exist_ok=True)
 print('PASS PreSuRe v'+str(cfg['version'])); print('PASS stage',a.stage); print('PASS cache',c); print('PASS train/val',len(tr),len(va)); print('PASS PatchSTG checkpoint',p); print('PASS checkpoint_source',src); print('PASS cache_fingerprint',ident['cache_fingerprint']); print('PREFLIGHT PASS')
if __name__=='__main__': main()
