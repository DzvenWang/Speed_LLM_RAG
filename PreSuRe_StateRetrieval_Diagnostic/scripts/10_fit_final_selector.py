#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from srd.utility_selector import add_inference_features, feature_columns, make_model, fit_model, predict_scores, tune_gate_and_cap, parse_int_list
from srd.utils import write_json

def log(msg):print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}",flush=True)
def main():
    ap=argparse.ArgumentParser(description='Fit deployable v0.7 selector using past fit block + newest calibration block')
    ap.add_argument('run_dir');ap.add_argument('--primary-scope',type=int,default=12);ap.add_argument('--max-k',type=int,default=50);ap.add_argument('--calibration-fraction',type=float,default=1/6);ap.add_argument('--patch-caps',default='1,2,4,6,8,12,16');ap.add_argument('--seed',type=int,default=10)
    a=ap.parse_args();rd=Path(a.run_dir).resolve();out=rd/'predictive_utility_selector';data=pd.read_csv(out/'utility_action_dataset.csv.gz');data=add_inference_features(data);feats=feature_columns(data)
    comp=pd.read_csv(out/'model_comparison.csv');z=comp[comp.recent_steps==int(a.primary_scope)].sort_values('realized_gain_pct',ascending=False)
    if z.empty:raise RuntimeError(f'No model comparison row for recent{a.primary_scope}')
    best=z.iloc[0];mname=str(best.model);qorder=data[['query_pos','query_id']].drop_duplicates().sort_values('query_pos').query_id.astype(int).to_numpy();ncal=max(8,int(round(len(qorder)*float(a.calibration_fraction))));ncal=min(ncal,max(1,len(qorder)//3));fitq=qorder[:-ncal];calq=qorder[-ncal:]
    fitdf=data[data.query_id.isin(fitq)].copy();caldf=data[(data.query_id.isin(calq))&(data.recent_steps==int(a.primary_scope))].copy();spec=fit_model(make_model(mname,a.seed),fitdf,feats);cs=predict_scores(spec,caldf,feats);gate=tune_gate_and_cap(caldf,cs,a.max_k,parse_int_list(a.patch_caps,[1,2,4,6,8,12,16]))
    artifact={'version':'0.7.1','model_name':mname,'model':spec,'features':feats,'primary_scope':int(a.primary_scope),'max_k':int(a.max_k),'gate':gate,'fit_query_min':int(fitq.min()),'fit_query_max':int(fitq.max()),'calib_query_min':int(calq.min()),'calib_query_max':int(calq.max()),'oof_realized_gain_pct':float(best.realized_gain_pct),'oracle_reference_pct':float(best.oracle_reference_pct)}
    path=out/'final_selector.joblib';joblib.dump(artifact,path)
    meta={k:v for k,v in artifact.items() if k not in ['model','features']};meta['feature_count']=len(feats);meta['artifact']=str(path);write_json(out/'final_selector.json',meta);log(json.dumps(meta,indent=2));log(f'FINAL_SELECTOR={path}')
if __name__=='__main__':main()
