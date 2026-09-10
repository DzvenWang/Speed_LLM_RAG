#!/usr/bin/env python3
from pathlib import Path
import json,sys
import pandas as pd
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline') if Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline').exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline.utils import load_yaml
cfg=load_yaml(ROOT/'configs/baseline.yaml'); exp=Path(cfg['project']['exp_root'])
rows=[]
for method in ['GinAR','RAFT','SARAF']:
    base=exp/method/cfg['experiment']['subgraph']/'development'
    latest=base/'LATEST_RUN.txt'
    if not latest.exists():
        rows.append({'method':method,'state':'NOT_RUN'});continue
    rd=Path(latest.read_text().strip()); sp=rd/'summary.json'
    if not sp.exists(): rows.append({'method':method,'state':'INCOMPLETE','run_dir':str(rd)});continue
    s=json.loads(sp.read_text());v=s.get('val',{})
    rows.append({'method':method,'state':'COMPLETE','overall_MAE':v.get('overall_MAE'),'overall_RMSE':v.get('overall_RMSE'),'masked_MAE':v.get('masked_MAE'),'support_MAE':v.get('support_MAE'),'H12_MAE':v.get('H12_MAE'),'H12_support_MAE':v.get('H12_support_MAE'),'best_epoch':s.get('best_epoch'),'params':s.get('params'),'run_dir':str(rd)})
df=pd.DataFrame(rows);print(df.to_string(index=False));out=exp/'baseline_main_development_summary.csv';df.to_csv(out,index=False);print('wrote',out)
