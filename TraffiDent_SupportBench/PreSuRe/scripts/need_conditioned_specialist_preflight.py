#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys, torch

ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe')
ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from presure.utils import load_yaml
from presure.contract import assert_cache_ready
from presure.pipeline import load_artifacts
from presure.need_gated import load_latest


def require(paths,label):
    miss=[str(p) for p in paths if not Path(p).exists()]
    if miss: raise RuntimeError(label+' missing: '+', '.join(miss))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development',choices=['development']);a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/presure.yaml')
    ident=assert_cache_ready(cfg,a.stage)
    art=load_artifacts(cfg,a.stage)
    require([art/'train_node_features.npy',art/'val_node_features.npy',art/'train_base_pred.npy',art/'val_base_pred.npy',art/'train_residual.npy'],'v0.7 artifacts')

    _,exact_rd=load_latest(cfg['project']['exp_root'],'PredictiveNeedProbe',cfg['experiment']['subgraph'],a.stage)
    _,state_rd=load_latest(cfg['project']['exp_root'],'PredictiveNeedStateProbe',cfg['experiment']['subgraph'],a.stage)
    _,gate_rd=load_latest(cfg['project']['exp_root'],'NeedGatedUtilityProbe',cfg['experiment']['subgraph'],a.stage)
    require([
        exact_rd/'utility_A_best.pt', exact_rd/'utility_train_ids.npy', exact_rd/'utility_dev_ids.npy',
        exact_rd/'top4_ids_utility_train.npy', exact_rd/'top4_conf_utility_train.npy', exact_rd/'top4_mask_utility_train.npy',
        exact_rd/'top4_ids_utility_dev.npy', exact_rd/'top4_conf_utility_dev.npy', exact_rd/'top4_mask_utility_dev.npy',
    ],'v0.7.1 Utility-A training reference')
    require([
        state_rd/'summary.json',state_rd/'split_manifest.json',state_rd/'need_q90_best.pt',state_rd/'prob_need_q90_final_val.npy',
        state_rd/'top4_ids_final_val.npy',state_rd/'top4_conf_final_val.npy',state_rd/'top4_mask_final_val.npy',
    ],'v0.7.2 Need-q90 reference')
    require([gate_rd/'summary.json',gate_rd/'DECISION.txt'],'v0.7.3 Need-Gated reference')

    ua=torch.load(exact_rd/'utility_A_best.pt',map_location='cpu',weights_only=False)
    if ua.get('model_type')!='support_utility_probe_v071' or ua.get('mode')!='A':
        raise RuntimeError(f'Utility-A checkpoint mismatch: {ua.get("model_type")} / {ua.get("mode")}')
    nq=torch.load(state_rd/'need_q90_best.pt',map_location='cpu',weights_only=False)
    if nq.get('model_type')!='predictive_need_state_probe_v072' or nq.get('task')!='need_q90':
        raise RuntimeError(f'Need-q90 checkpoint mismatch: {nq.get("model_type")} / {nq.get("task")}')
    sm=json.loads((state_rd/'summary.json').read_text())
    q90=sm.get('train_only_thresholds',{}).get('q90')
    if not isinstance(q90,list) or len(q90)!=int(cfg['experiment']['horizon']):
        raise RuntimeError('v0.7.2 Train-only horizon-specific q90 thresholds missing')
    gm=json.loads((gate_rd/'summary.json').read_text())
    if gm.get('v08_need_gated_recommendation')!='NO_GO_V08_NEED_GATED':
        print('WARN v0.7.3 latest decision is',gm.get('v08_need_gated_recommendation'))

    print('PASS PreSuRe v'+str(cfg['version']))
    print('PASS specialist role',cfg['need_conditioned_utility_specialist_probe']['role'])
    print('PASS cache_fingerprint',ident['cache_fingerprint'])
    print('PASS Utility-A initialization',exact_rd)
    print('PASS Need-q90 + Train-only q90 thresholds',state_rd)
    print('PASS v0.7.3 diagnostic reference',gate_rd)
    print('PASS no M0-M5 / no retrieval rebuild / no Need-q90 retraining required')
    print('NEED-CONDITIONED UTILITY SPECIALIST PREFLIGHT PASS')

if __name__=='__main__':main()
