#!/usr/bin/env python3
"""Simple standard evaluation for the currently frozen PatchSTG checkpoint.

Uses FROZEN_BACKBONE.json only to select the checkpoint.  It intentionally avoids
Protocol access gates and formal/evaluation identity coupling.
"""
from pathlib import Path
import argparse, datetime, json, sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent/'TraffiDent_SupportBench'))
from backbone.utils import load_yaml, install_tee, write_json, log, sha256_file
from backbone.shared_cache import cache_path
from backbone.data import CachedSplit
from backbone.patchstg import build_patchstg, compute_patchstg_layout
from backbone.runner import evaluate
from backbone.predictions import save_prediction_npz
from backbone.evaluation_reporting import (
    scenario_rows, severity_row, write_result_json,
    severity_table_lines, scenario_table_lines,
)
from supportbench.dataset import SupportBench

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--severity',type=int,choices=[20,40,60,90],required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--config',default=str(ROOT/'configs/backbone.yaml')); ap.add_argument('--run-stamp'); ap.add_argument('--result-json'); ap.add_argument('--quiet-summary',action='store_true'); a=ap.parse_args(); cfg=load_yaml(a.config)
    stamp=a.run_stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S'); lp=Path(cfg['project']['log_root'])/f'PatchSTG.evaluation.s{a.severity}.seed{a.seed}.{stamp}.log'; install_tee(lp); print('LOG_FILE',lp)
    cache=cache_path(cfg,'evaluation',a.severity)
    if not cache.exists(): raise FileNotFoundError(f'Evaluation cache not found: {cache}')
    exp=Path(cfg['project']['exp_root']); frozen_path=exp/'FROZEN_BACKBONE.json'
    if not frozen_path.exists(): raise FileNotFoundError(frozen_path)
    frozen=json.load(open(frozen_path)); entry=frozen['formal_checkpoints'].get(str(a.seed))
    if not entry: raise RuntimeError(f'Frozen manifest has no checkpoint for seed {a.seed}')
    checkpoint=Path(entry['checkpoint']);
    if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
    got=sha256_file(checkpoint)
    expected_sha=entry.get('checkpoint_sha256')
    if expected_sha and got!=expected_sha: log(f'WARNING checkpoint SHA256 differs from frozen manifest expected={expected_sha} current={got}; continuing')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); ck=torch.load(checkpoint,map_location=device,weights_only=False)
    for key in ('state_dict','model_config','mean','std'):
        if key not in ck: raise RuntimeError(f'Checkpoint missing required key: {key}')
    b=SupportBench(cfg['experiment']['subgraph'],root=cfg['project']['benchmark_root']); test=CachedSplit(cache,'test')
    if int(ck.get('num_nodes',b.N))!=int(b.N): raise RuntimeError(f'Checkpoint num_nodes={ck.get("num_nodes")} != benchmark N={b.N}')
    _,_,_,current_layout=compute_patchstg_layout(ROOT/'vendor'/'PatchSTG',b.nodes,b.adj,b.N,ck['model_config'])
    old_layout=ck.get('adapter_metadata',{}).get('spatial_layout_fingerprint') if isinstance(ck.get('adapter_metadata'),dict) else None
    layout_match=(not old_layout) or str(old_layout)==str(current_layout)
    if not layout_match: log(f'WARNING spatial layout differs checkpoint={old_layout} current={current_layout}; continuing with current layout')
    adapter=build_patchstg(ROOT/'vendor'/'PatchSTG',b.N,device,float(ck['mean']),float(ck['std']),ck['model_config'],b.adj,b.nodes); adapter.model.load_state_dict(ck['state_dict'],strict=True)
    loader=DataLoader(test,batch_size=int(ck.get('batch_size',ck['model_config'].get('batch_size',16))),shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available()); pred,df,summary=evaluate(adapter,loader,test.meta,device)
    families=[str(x) for x in cfg['experiment']['families']]
    query_counts={str(k):int(v) for k,v in test.meta['family'].astype(str).value_counts().to_dict().items()}
    sev_row=severity_row(summary,a.severity,a.seed)
    scen_rows=scenario_rows(summary,families,a.severity,a.seed,query_counts)
    out=exp/'PatchSTG'/cfg['experiment']['subgraph']/'evaluation'/f's{a.severity}'/f'{stamp}_seed{a.seed}'; out.mkdir(parents=True,exist_ok=True)
    save_prediction_npz(out/'test_predictions.npz',test.meta.view_id.astype(str).tolist(),pred.astype(np.float32),manifest_path=out/'test_predictions.manifest.json',metadata={'producer':'Backbone','model':'PatchSTG','seed':a.seed,'stage':'evaluation','severity':a.severity,'checkpoint':str(checkpoint),'checkpoint_sha256':got,'evaluation_policy':'simple_current_cache_current_layout','spatial_layout_match':bool(layout_match)})
    df.to_csv(out/'test_metrics.csv',index=False)
    pd.DataFrame(scen_rows).to_csv(out/'test_scenario_metrics.csv',index=False)
    write_json(out/'test_summary.json',{'project_version':cfg['version'],'model':'PatchSTG','seed':a.seed,'severity':a.severity,'checkpoint':str(checkpoint),'checkpoint_sha256':got,'evaluation_policy':'current evaluation cache + current PatchSTG layout + frozen checkpoint weights','checkpoint_spatial_layout_fingerprint':old_layout,'current_spatial_layout_fingerprint':current_layout,'spatial_layout_match':bool(layout_match),'test':summary,'severity_summary':sev_row,'scenario_summary':scen_rows})
    if a.result_json:
        write_result_json(a.result_json,severity=sev_row,scenarios=scen_rows,metadata={'output_dir':str(out),'checkpoint':str(checkpoint),'checkpoint_sha256':got,'layout_match':bool(layout_match)})
    log(f'EVALUATION COMPLETE s{a.severity} seed={a.seed} MAE={summary["MAE"]:.6f} support_MAE={summary["support_dependent_MAE"]:.6f}')
    if not a.quiet_summary:
        print('\nFINAL SEVERITY SUMMARY',flush=True)
        for line in severity_table_lines([sev_row]): print(line,flush=True)
        print('\nFINAL SCENARIO SUMMARY',flush=True)
        for line in scenario_table_lines(scen_rows): print(line,flush=True)
if __name__=='__main__': main()
