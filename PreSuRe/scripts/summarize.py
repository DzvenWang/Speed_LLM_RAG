#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,pandas as pd
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe');ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));from presure.utils import load_yaml
ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development');a=ap.parse_args();cfg=load_yaml(ROOT/'configs/presure.yaml');rows=[]
for v in ['OracleA','OracleB','OracleC','OracleE','OracleF','OracleD','M0','M1','M2','M3','M4','M5']:
 r=Path(cfg['project']['exp_root'])/v/cfg['experiment']['subgraph']/a.stage;fs=sorted(r.glob('*/summary.json'),key=lambda p:p.stat().st_mtime) if r.exists() else []
 if not fs: continue
 j=json.loads(fs[-1].read_text());m=j.get('validation') or j.get('oracle') or {};ri=j.get('relative_improvement_over_M0') or j.get('relative_improvement') or {}
 rows.append({'variant':v,'overall_MAE':m.get('overall_MAE'),'masked_MAE':m.get('masked_MAE'),'support_MAE':m.get('support_MAE'),'H12_MAE':m.get('H12_overall_MAE'),'H12_masked_MAE':m.get('H12_masked_MAE'),'H12_support_MAE':m.get('H12_support_MAE'),'rel_overall':ri.get('overall'),'rel_masked':ri.get('masked'),'rel_support':ri.get('support'),'queries_improved_fraction':j.get('queries_improved_fraction'),'role_accept_fraction':j.get('query_horizon_acceptance_fraction') or j.get('region_horizon_acceptance_fraction'),'harm_fraction':j.get('harmful_augmentation_fraction_vs_M0'),'run_dir':str(fs[-1].parent)})
df=pd.DataFrame(rows);print(df.to_string(index=False));out=Path(cfg['project']['exp_root'])/f'presure_{a.stage}_summary.csv';df.to_csv(out,index=False);print('wrote',out)
