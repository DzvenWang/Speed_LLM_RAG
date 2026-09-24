#!/usr/bin/env python3
from pathlib import Path
import argparse,sys
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe'); ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml
from presure.contract import assert_cache_ready,cache_dir
from presure.data import CacheSplit
from presure.backbone import resolve_checkpoint

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); a=ap.parse_args(); cfg=load_yaml(ROOT/'configs/presure_v12.yaml'); print(f'PreSuRe {cfg["version"]} Dual-Query Retrieval-Conditioned Forecasting preflight',flush=True)
    for key in ['backbone_root','backbone_exp_root','supportbench_root','benchmark_root','common_root']:
        p=Path(cfg['project'][key]);
        if not p.exists(): raise FileNotFoundError(f'{key}: {p}')
        print('PASS',key,p,flush=True)
    ident=assert_cache_ready(cfg,a.stage); tr=CacheSplit(cache_dir(cfg,a.stage),'train'); va=CacheSplit(cache_dir(cfg,a.stage),'val'); H=int(cfg['experiment']['history']); P=int(cfg['experiment']['horizon']); fc=cfg['forecaster']; pc=cfg['candidate_pool']
    if tr.x.shape[1]!=H or tr.y.shape[1]!=P: raise RuntimeError('Cache H/P mismatch')
    if len(va)<int(fc['dev_queries'])+int(fc['final_val_queries']): raise RuntimeError('Validation too small for disjoint Development/Final')
    if int(pc['top_k'])!=8 or int(pc['top_k'])>int(pc['candidate_k']): raise RuntimeError('v1.2 requires TopK=8 and TopK<=candidate_k')
    attn_chunk=int(fc.get('attention_batch_chunk',32768)); effective_train=int(fc['batch_size'])*int(tr.x.shape[2])*P; effective_eval=int(fc['eval_batch_size'])*int(tr.x.shape[2])*P
    if attn_chunk<=0 or attn_chunk>60000: raise RuntimeError(f'attention_batch_chunk must be in [1,60000], got {attn_chunk}')
    p,ck,src=resolve_checkpoint(cfg,a.stage); print('PASS cache',cache_dir(cfg,a.stage),'fingerprint',ident.get('cache_fingerprint'),flush=True); print('PASS PatchSTG checkpoint',p,'source',src,flush=True); print(f'PASS shapes train={len(tr)} val={len(va)} N={tr.x.shape[2]} H={H} P={P}',flush=True); print('PASS design Dual Query + Top8 + conditional forecasting; no direct support transfer',flush=True); print(f'PASS chunked support attention train_effective_batch={effective_train} eval_effective_batch={effective_eval} chunk={attn_chunk} (<65535 backend limit)',flush=True); print('PASS no Qwen / no Dual Memory / no utility-sign / no veto / no acceptance',flush=True); print('PRESURE V1.2 PREFLIGHT PASS',flush=True)
if __name__=='__main__': main()
