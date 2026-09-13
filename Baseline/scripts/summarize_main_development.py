#!/usr/bin/env python3
from pathlib import Path
import json, sys
import pandas as pd
GADI_ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT=GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline.utils import load_yaml
from baseline.run_identity import load_cache_identity, require_same_identity
from baseline.shared_cache import cache_path
cfg=load_yaml(ROOT/'configs/baseline.yaml'); exp=Path(cfg['project']['exp_root'])
cache=cache_path(cfg,'development')
expected=load_cache_identity(cache,cfg['version'])
rows=[]
for method in ['GinAR','RAFT','SARAF']:
    base=exp/method/cfg['experiment']['subgraph']/'development'
    cands=[]
    for sp in base.glob('*/summary.json') if base.exists() else []:
        try:
            s=json.loads(sp.read_text()); require_same_identity(s,expected,label=str(sp)); cands.append((sp.stat().st_mtime,sp,s))
        except Exception:
            continue
    if not cands:
        rows.append({'method':method,'state':'NOT_RUN_CURRENT_IDENTITY'});continue
    _,sp,s=max(cands,key=lambda x:x[0]); rd=sp.parent; v=s.get('val',{})
    rows.append({'method':method,'state':'COMPLETE','overall_MAE':v.get('overall_MAE'),'overall_RMSE':v.get('overall_RMSE'),'masked_MAE':v.get('masked_MAE'),'support_MAE':v.get('support_MAE'),'H12_MAE':v.get('H12_MAE'),'H12_support_MAE':v.get('H12_support_MAE'),'best_epoch':s.get('best_epoch'),'params':s.get('params'),'run_dir':str(rd)})
df=pd.DataFrame(rows);print(df.to_string(index=False));out=exp/'baseline_main_development_summary.csv';df.to_csv(out,index=False);print('wrote',out)
