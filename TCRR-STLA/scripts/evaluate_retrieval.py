#!/usr/bin/env python3
"""Paired attribution audit for a full Retrieval checkpoint (v2.3).

Distinguishes: within-model base/full, intervention with wrong evidence,
and independently optimized no-retrieval control. Never relabels a
full-checkpoint no-evidence intervention as a separately trained baseline.
"""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tcrr_stla.config import relocate_saved_config,cache_path,file_sha256

MODES=['full','shuffled','cross_history','no_evidence','no_mask_replay']
BASE_MODE='no_evidence'
MECHANISM_MODE='no_mask_replay'

def get_data(ckpath, sev, mode):
    f=Path(ckpath).resolve().parent/'evaluation'/f's{sev}'/mode/'test_predictions.npz'
    if not f.is_file():raise FileNotFoundError(f'Missing audit prediction: {f}')
    with np.load(f,allow_pickle=False) as z:
        return np.array(z['prediction'],dtype=np.float32),np.array(z['base_prediction'],dtype=np.float32)

def safe_metrics(p,y,valid):
    m=np.asarray(valid,bool)&np.isfinite(p)&np.isfinite(y)
    e=np.asarray(p)[m]-np.asarray(y)[m]
    if not e.size:return (float('nan'),float('nan'),0)
    return (float(np.abs(e).mean()),float(np.sqrt(np.mean(e*e))),int(e.size))

def record(model,sev,group,horizon,subset,truth,valid,full,base,wrong,no_ev,indep=None,mechanism=None):
    y=truth[:,horizon:horizon+1];m=valid[:,horizon:horizon+1]
    if subset is not None:
        y=y[subset];m=m[subset]
    ps=[z[:,horizon:horizon+1][subset] if subset is not None else z[:,horizon:horizon+1]
        for z in (full,base,wrong,no_ev)]
    m0,r0,n=safe_metrics(ps[0],y,m)
    if n==0:return None
    bm,br,_=safe_metrics(ps[1],y,m)
    wm,wr,_=safe_metrics(ps[2],y,m)
    em,er,_=safe_metrics(ps[3],y,m)
    if mechanism is not None:
        z=mechanism[:,horizon:horizon+1];z=z[subset] if subset is not None else z
        mechanism_mae,mechanism_rmse,_=safe_metrics(z,y,m)
    else:mechanism_mae=mechanism_rmse=float('nan')
    if indep is not None:
        z=indep[:,horizon:horizon+1];z=z[subset] if subset is not None else z
        im,ir,_=safe_metrics(z,y,m)
    else:im=ir=float('nan')
    ok=m&np.isfinite(ps[0])&np.isfinite(ps[1])&np.isfinite(y)
    base_error=np.abs(ps[1][ok]-y[ok]);full_error=np.abs(ps[0][ok]-y[ok])
    utility=base_error-full_error
    return dict(model=model,severity=sev,family=group,horizon=horizon+1,
       observations=n,MAE=m0,RMSE=r0,base_MAE=bm,base_RMSE=br,
       wrong_evidence_MAE=wm,wrong_evidence_RMSE=wr,mechanism_ablation_MAE=mechanism_mae,mechanism_ablation_RMSE=mechanism_rmse,no_evidence_MAE=em,
       no_evidence_RMSE=er,independent_no_retrieval_MAE=im,
       independent_no_retrieval_RMSE=ir,
       internal_gain_pct=100*(bm-m0)/max(bm,1e-10),
       true_vs_wrong_gain_pct=100*(wm-m0)/max(wm,1e-10),
       independent_gain_pct=100*(im-m0)/max(im,1e-10) if np.isfinite(im) else float('nan'),
       positive_correction_rate=float((utility>0).mean()),
       negative_transfer_rate=float((utility<0).mean()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True,help='v1.0 FULL best.pt')
    ap.add_argument('--no-retrieval-checkpoint',help='separately trained no_retrieval best.pt')
    ap.add_argument('--backbone-checkpoint')
    ap.add_argument('--severities',nargs='+',type=int,default=[20,40,60,90])
    ap.add_argument('--reuse',action='store_true',help='reuse previously exported predictions')
    args=ap.parse_args()
    # subprocesses run with project cwd: resolve CLI paths before dispatch.
    args.checkpoint=str(Path(args.checkpoint).expanduser().resolve())
    if args.no_retrieval_checkpoint:
        args.no_retrieval_checkpoint=str(Path(args.no_retrieval_checkpoint).expanduser().resolve())
    if args.backbone_checkpoint:
        args.backbone_checkpoint=str(Path(args.backbone_checkpoint).expanduser().resolve())
    full_ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if full_ck.get('version')!='2.0.0':raise RuntimeError('v2.3 Retrieval Audit requires v2.0 checkpoint')
    if full_ck['config']['training'].get('ablation','full')!='full':raise ValueError('Audit requires full-mode checkpoint')
    cfg=relocate_saved_config(full_ck['config'])
    os.environ['TRAFFIDENT_TARGET']=cfg['experiment']['target']
    if args.no_retrieval_checkpoint:
        ctrl=torch.load(args.no_retrieval_checkpoint,map_location='cpu',weights_only=False)
        if ctrl.get('version')!='2.0.0':raise RuntimeError('No-retrieval control version mismatch')
        if ctrl['config']['training'].get('ablation')!='no_retrieval':raise ValueError('Independent control must be trained with no_retrieval')
        if ctrl['backbone_sha256']!=full_ck['backbone_sha256'] or ctrl['config']['experiment']!=full_ck['config']['experiment']:
            raise RuntimeError('Independent control must share backbone SHA and experiment protocol')
    def ensure(checkpoint,mode):
        needed=[Path(checkpoint).resolve().parent/'evaluation'/f's{s}'/mode/'test_predictions.npz' for s in args.severities]
        if args.reuse and all(p.is_file() for p in needed):
            h=file_sha256(checkpoint)
            if all((p.parent/'summary.json').is_file() and
                   json.loads((p.parent/'summary.json').read_text()).get('checkpoint_sha256')==h for p in needed):
                return
            print('AUDIT_REUSE_INVALID stale_or_legacy_evaluation',checkpoint,mode,flush=True)
        cmd=[sys.executable,'-u',str(ROOT/'scripts/evaluate.py'),'--checkpoint',str(checkpoint),
             '--evidence-mode',mode,'--severities',*map(str,args.severities)]
        if args.backbone_checkpoint:cmd+=['--backbone-checkpoint',args.backbone_checkpoint]
        subprocess.check_call(cmd,cwd=ROOT)
    for mode in MODES:ensure(args.checkpoint,mode)
    if args.no_retrieval_checkpoint:ensure(args.no_retrieval_checkpoint,'no_retrieval')
    audit=Path(args.checkpoint).resolve().parent/'retrieval_audit';audit.mkdir(exist_ok=True)
    rows=[]
    for sev in args.severities:
        dsroot=cache_path(cfg,'evaluation',sev)
        y=np.load(dsroot/'test_y_raw.npy',mmap_mode='r')
        v=np.load(dsroot/'test_future_valid.npy',mmap_mode='r')
        direct=np.load(dsroot/'test_direct_mask.npy',mmap_mode='r')
        manifest=pd.read_csv(dsroot/'test_manifest.csv')
        full,base=get_data(args.checkpoint,sev,'full')
        wrong,_=get_data(args.checkpoint,sev,'cross_history')
        no_ev,_=get_data(args.checkpoint,sev,BASE_MODE)
        mechanism,_=get_data(args.checkpoint,sev,MECHANISM_MODE)
        no_ret=get_data(args.no_retrieval_checkpoint,sev,'no_retrieval')[0] if args.no_retrieval_checkpoint else None
        if any(a.shape!=y.shape for a in (full,base,wrong,no_ev)) or (no_ret is not None and no_ret.shape!=y.shape):
            raise RuntimeError('Prediction/truth shape mismatch; paired evaluation is not aligned')
        subsets={'ALL':None}
        if 'family' in manifest:
            ff=manifest.family.astype(str).to_numpy()
            for family in sorted(set(ff)):subsets['family:'+family]=ff==family
        # Directly masked nodes / nodes still observed: same sample pairing.
        masked=np.asarray(direct,dtype=bool)
        observed=~masked
        for group,subset in subsets.items():
            for h in range(y.shape[1]):
                vv=np.asarray(v,bool)
                if group=='ALL':
                    for name,condition in [('direct_masked',masked),('not_direct_masked',observed)]:
                        valid_subset=vv & condition[:,None,:]
                        # Full arrays, but validity restricted to the node category.
                        r=record('AUDIT',sev,name,h,None,np.asarray(y),valid_subset,full,base,wrong,no_ev,no_ret,mechanism)
                        if r:rows.append(r)
                r=record('AUDIT',sev,group,h,subset,np.asarray(y),np.asarray(v),full,base,wrong,no_ev,no_ret,mechanism)
                if r:rows.append(r)
        print(f'RETRIEVAL_AUDIT severity={sev} evaluated_samples={len(y)}',flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(audit/'paired_horizon_family.csv',index=False)
    # Microaggregated horizon results; these are NOT simple means across H3/H6/H12.
    agg=[]
    for (sev,grp),part in frame.groupby(['severity','family']):
        counts=part.observations.to_numpy(float)
        def avg(field):
            vals=part[field].to_numpy(float)
            ok=np.isfinite(vals)&(counts>0)
            return float(np.average(vals[ok],weights=counts[ok])) if ok.any() else float('nan')
        # MAE and rates are exact valid-element weighted; RMSE uses squared errors.
        res={'severity':int(sev),'family':grp,'n':int(counts.sum())}
        for field in ('MAE','base_MAE','wrong_evidence_MAE','mechanism_ablation_MAE','no_evidence_MAE','independent_no_retrieval_MAE',
                      'positive_correction_rate','negative_transfer_rate'):
            res[field]=avg(field)
        for field in ('RMSE','base_RMSE','wrong_evidence_RMSE','mechanism_ablation_RMSE','no_evidence_RMSE','independent_no_retrieval_RMSE'):
            z=(part[field].to_numpy(float))**2;ok=np.isfinite(z)&(counts>0)
            res[field]=float(np.sqrt(np.average(z[ok],weights=counts[ok]))) if ok.any() else float('nan')
        def gain(b,a):return 100*(b-a)/max(b,1e-10) if np.isfinite(b) else float('nan')
        res['internal_gain_pct']=gain(res['base_MAE'],res['MAE'])
        res['true_vs_wrong_gain_pct']=gain(res['wrong_evidence_MAE'],res['MAE'])
        res['mechanism_gain_pct']=gain(res['mechanism_ablation_MAE'],res['MAE'])
        res['independent_gain_pct']=gain(res['independent_no_retrieval_MAE'],res['MAE'])
        agg.append(res)
    pd.DataFrame(agg).to_csv(audit/'paired_summary.csv',index=False)
    manifest={'schema':23,'model':'TCRR-STLA','full_checkpoint':str(Path(args.checkpoint).resolve()),
              'full_backbone_sha256':full_ck.get('backbone_sha256'),
              'no_retrieval_checkpoint':args.no_retrieval_checkpoint,
              'warning':'Within-checkpoint interventions are not independent retrained controls.',
              'wrong_evidence':'cross_history: deterministic batch-invariant, nonoverlapping Train future controls; original retrieval rank_shuffle exported separately',
              'mechanism_ablation':MECHANISM_MODE,'severities':args.severities}
    (audit/'audit_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('RETRIEVAL_AUDIT_PASS',audit/'paired_summary.csv',flush=True)
if __name__=='__main__':main()
