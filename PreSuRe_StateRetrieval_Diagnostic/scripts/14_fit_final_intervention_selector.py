#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys, time
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from srd.intervention_utility import add_v08_features,add_utility_targets,feature_columns_v08,make_v08_model,fit_v08_model,predict_v08_scores,tune_gate_and_cap
from srd.utils import write_json

def log(msg):print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}",flush=True)
def ints(s):return [int(x.strip()) for x in str(s).split(',') if x.strip()]
def main():
    ap=argparse.ArgumentParser(description='Fit deployable v0.8 intervention-aware selector')
    ap.add_argument('run_dir');ap.add_argument('--primary-scope',type=int,default=12);ap.add_argument('--max-k',type=int,default=20);ap.add_argument('--calibration-fraction',type=float,default=1/6);ap.add_argument('--patch-caps',default='1,2,4,6,8,12,16');ap.add_argument('--seed',type=int,default=10);ap.add_argument('--harmful-margin',type=float,default=0.01)
    a=ap.parse_args();rd=Path(a.run_dir).resolve();out=rd/'intervention_utility_selector';data=pd.read_csv(out/'intervention_action_dataset.csv.gz');data=add_utility_targets(add_v08_features(data),a.harmful_margin)
    comp=pd.read_csv(out/'model_comparison.csv');z=comp[comp.recent_steps==int(a.primary_scope)].sort_values('realized_gain_pct',ascending=False)
    if z.empty:raise RuntimeError(f'No v0.8 model row for recent{a.primary_scope}')
    best=z.iloc[0];mname=str(best.model);fset=str(best.feature_set);feats=feature_columns_v08(data,fset)
    qorder=data[['query_pos','query_id']].drop_duplicates().sort_values('query_pos').query_id.astype(int).to_numpy();ncal=max(8,int(round(len(qorder)*float(a.calibration_fraction))));ncal=min(ncal,max(1,len(qorder)//3));fitq=qorder[:-ncal];calq=qorder[-ncal:]
    fitdf=data[data.query_id.isin(fitq)].copy();caldf=data[(data.query_id.isin(calq))&(data.recent_steps==int(a.primary_scope))].copy();spec=fit_v08_model(make_v08_model(mname,a.seed),fitdf,feats,a.harmful_margin);cs=predict_v08_scores(spec,caldf,feats);gate=tune_gate_and_cap(caldf,cs,a.max_k,ints(a.patch_caps))
    artifact={'version':'0.8.0','model_name':mname,'feature_set':fset,'model':spec,'features':feats,'primary_scope':int(a.primary_scope),'max_k':int(a.max_k),'gate':gate,
              'fit_query_positions':[0,int(len(fitq)-1)],'calib_query_positions':[int(len(fitq)),int(len(qorder)-1)],'oof_realized_gain_pct':float(best.realized_gain_pct),'oracle_reference_pct':float(best.oracle_reference_pct),
              'leakage_contract':'Deployable score uses intervention-aware inference features only; calibration uses the newest historical labeled block.'}
    path=out/'final_intervention_selector.joblib';joblib.dump(artifact,path);meta={k:v for k,v in artifact.items() if k not in ['model','features']};meta['feature_count']=len(feats);meta['artifact']=str(path);write_json(out/'final_intervention_selector.json',meta);log(json.dumps(meta,indent=2));log(f'FINAL_SELECTOR={path}')
if __name__=='__main__':main()
