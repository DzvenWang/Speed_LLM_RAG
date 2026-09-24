#!/usr/bin/env python3
from pathlib import Path
import argparse,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from oracle.config import load_config
from oracle.contract import assert_observation_experiment_ready,cache_dir
from oracle.backbone import resolve_checkpoint
from oracle.data import audit_canonical_source


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--checkpoint',default=None); a=ap.parse_args()
    cfg=load_config(ROOT); print(f'PatchSTG Oracle {cfg["version"]} preflight',flush=True)
    print('WORKSPACE_ROOT',cfg['project']['root'],flush=True)
    for key in ['oracle_root','exp_root','log_root','cache_root','backbone_root','backbone_exp_root','common_root','supportbench_root','benchmark_root']:
        p=Path(cfg['project'][key]);
        if key in ['exp_root','log_root']: p.mkdir(parents=True,exist_ok=True)
        elif not p.exists(): raise FileNotFoundError(f'{key}: {p}')
        print('PASS',key,p,flush=True)
    backbone_root=Path(cfg['project']['backbone_root'])
    verify=backbone_root/'scripts'/'source_lock.py'
    sources=backbone_root/'configs'/'backbone_selection.yaml'
    cp=subprocess.run([sys.executable,str(verify),'verify','--root',str(backbone_root),'--sources',str(sources)],cwd=backbone_root)
    if cp.returncode!=0: raise RuntimeError('Backbone official source verification failed')
    ident=assert_observation_experiment_ready(cfg,a.stage); p,ck,source=resolve_checkpoint(cfg,a.stage,a.checkpoint)
    audit=audit_canonical_source(cfg,cache_dir(cfg,a.stage),float(ck['mean']),float(ck['std']))
    cc=cfg['canonical_history']
    if audit['observed_match_mae']>float(cc['observed_match_mae_tol']): raise RuntimeError(f'Canonical observed-match MAE too large: {audit}')
    if audit['observed_match_max_abs']>float(cc['observed_match_max_tol']): raise RuntimeError(f'Canonical observed-match max error too large: {audit}')
    if audit['observed_on_raw_invalid_fraction']>1e-9: raise RuntimeError(f'Observed cache cells disagree with raw-valid mask: {audit}')
    if audit['masked_only_vs_all_raw_valid_max_abs']>float(cc['observed_match_max_tol']): raise RuntimeError(f'Masked-only/all-raw-valid replacement mismatch too large: {audit}')
    if audit['restorable_fraction_of_raw_valid']<=0: raise RuntimeError(f'No recoverable synthetic missing history found: {audit}')
    print('PASS cache fingerprint',ident.get('cache_fingerprint'),flush=True)
    print('PASS PatchSTG checkpoint',p,'source',source,'mean',ck['mean'],'std',ck['std'],flush=True)
    print('COMPLETE_HISTORY_SOURCE',audit,flush=True)
    print('PASS oracle definition: restore only canonical raw-valid history cells hidden by benchmark incomplete sensing; raw-invalid cells are never fabricated',flush=True)
    print('PASS observation experiments: train/val full history, observation masks, and artificial-missing targets are present',flush=True)
    print('PASS ceiling scope: frozen-PatchSTG inference-time recoverable-history information ceiling; not a Bayes or retrained-architecture ceiling',flush=True)
    print('PATCHSTG ORACLE PREFLIGHT PASS',flush=True)
if __name__=='__main__': main()
