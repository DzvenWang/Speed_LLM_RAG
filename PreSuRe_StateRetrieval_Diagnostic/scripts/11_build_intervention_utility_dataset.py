#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._common import setup
from srd.contract import assert_cache_ready
from srd.data import CacheView
from srd.patch_layout import extract_patch_layout
from srd.backbone import load_patchstg_adapter
from srd.evaluate import predict_raw, predict_raw_prehead_batched
from srd.patch_transport import candidate_proposal, patch_overlap_mask
from srd.intervention_utility import add_v08_features, feature_columns_v08, add_utility_targets
from srd.utils import write_json


def log(msg): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def gpu(device):
    if getattr(device, 'type', 'cpu') != 'cuda' or not torch.cuda.is_available(): return 'device=cpu'
    try:
        return f"gpu_alloc={torch.cuda.memory_allocated(device)/2**30:.2f}GB gpu_reserved={torch.cuda.memory_reserved(device)/2**30:.2f}GB peak={torch.cuda.max_memory_allocated(device)/2**30:.2f}GB"
    except Exception: return 'device=cuda'

def _stats(v):
    a=np.asarray(v,np.float64); a=a[np.isfinite(a)]
    return (float(a.mean()),float(a.std())) if a.size else (np.nan,np.nan)

def _delta_abs(x, valid):
    x=np.asarray(x,np.float64); valid=np.asarray(valid,bool)
    if x.shape[0]<2:return np.nan
    m=valid[1:]&valid[:-1]
    return float(np.abs(x[1:]-x[:-1])[m].mean()) if np.any(m) else np.nan

def _cos(a,b):
    a=np.asarray(a,np.float64).reshape(-1); b=np.asarray(b,np.float64).reshape(-1)
    m=np.isfinite(a)&np.isfinite(b)
    if not np.any(m):return np.nan
    a=a[m];b=b[m];den=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/den) if den>1e-12 else np.nan

def _affine_params_inference(query,candidate,nodes,cfg):
    overlap=patch_overlap_mask(query,candidate,nodes,str(cfg['patch_transport'].get('alignment_scope','all')))
    qx=np.asarray(query['x_obs'],np.float32);cx=np.asarray(candidate['x_full'],np.float32)
    qv=qx[overlap].astype(np.float64);cv=cx[overlap].astype(np.float64)
    min_pts=int(cfg['patch_transport'].get('affine_min_points',4));eps=float(cfg['patch_transport'].get('eps',1e-6))
    if qv.size<min_pts:
        a=1.0;b=float(qv.mean()-cv.mean()) if qv.size else 0.0
    else:
        cm,qm=float(cv.mean()),float(qv.mean());var=float(np.mean((cv-cm)**2))
        a=1.0 if var<=eps else float(np.mean((cv-cm)*(qv-qm))/(var+eps))
        lo,hi=cfg['patch_transport'].get('affine_scale_clip',[0.5,2.0]);a=float(np.clip(a,float(lo),float(hi)));b=float(qm-a*cm)
    return a,b,int(qv.size)

def _nearest_anchor_jumps(query, proposal, fill, nodes):
    qx=np.asarray(query['x_obs'],np.float32);obs=np.asarray(query['history_observed'],bool)
    vals=[]
    for n in np.asarray(nodes,np.int64):
        anchors=np.flatnonzero(obs[:,n])
        if anchors.size==0:continue
        for t in np.flatnonzero(fill[:,n]):
            prev=anchors[anchors<t]
            ai=int(prev[-1]) if prev.size else int(anchors[np.argmin(np.abs(anchors-t))])
            vals.append(abs(float(proposal[t,n])-float(qx[ai,n])))
    if not vals:return np.nan,np.nan
    return float(np.mean(vals)),float(np.max(vals))

def _adjacent_nodes(layout,pid):
    ids=[]
    if int(pid)>0:ids.append(int(pid)-1)
    if int(pid)+1<int(layout.patch_count):ids.append(int(pid)+1)
    if not ids:return np.empty((0,),np.int64)
    return np.unique(np.concatenate([np.asarray(layout.groups[j],np.int64) for j in ids]))

def proposal_features(query,candidate,layout,pid,L,method,cfg):
    nodes=np.asarray(layout.groups[int(pid)],np.int64); neigh=_adjacent_nodes(layout,int(pid))
    qexec=dict(query);qexec['restorable']=np.asarray(query['artificial_missing'],bool)
    proposal,fill=candidate_proposal(qexec,candidate,nodes,int(L),method,cfg)
    qx=np.asarray(query['x_obs'],np.float32);obs=np.asarray(query['history_observed'],bool)
    cx=np.asarray(candidate['x_full'],np.float32);cv=np.asarray(candidate['full_valid'],bool)
    H=qx.shape[0];sl=slice(max(0,H-int(L)),H)
    cp=cx[:,nodes];cpv=cv[:,nodes];crm,crs=_stats(cp[sl][cpv[sl]]);cm,cs=_stats(cp[cpv])
    cdelta=_delta_abs(cp,cpv)
    qpatch=qx[:,nodes];qobs=obs[:,nodes];qm,qs=_stats(qpatch[qobs]);qdelta=_delta_abs(qpatch,qobs)
    a,b,ov=_affine_params_inference(qexec,candidate,nodes,cfg) if method=='affine_alignment' else (1.0,0.0,0)
    fillv=np.asarray(proposal,np.float32)[fill]
    basev=qx[fill]
    diff=(fillv-basev).astype(np.float64) if fillv.size else np.asarray([],np.float64)
    fm,fs=_stats(fillv)
    ajm,ajx=_nearest_anchor_jumps(qexec,proposal,fill,nodes)
    after=qx.copy();after[fill]=proposal[fill];valid_after=obs|fill
    ap=after[sl][:,nodes];av=valid_after[sl][:,nodes];am,astd=_stats(ap[av])
    # Adjacent-patch coherence uses only current observed neighbor state.
    ngaps=[]; ngood=0; ntotal=0
    if neigh.size:
        for t in range(max(0,H-int(L)),H):
            pm=valid_after[t,nodes];nm=obs[t,neigh]
            if np.any(pm):
                ntotal+=1
                if np.any(nm):
                    pv=float(after[t,nodes][pm].mean());nv=float(qx[t,neigh][nm].mean())
                    ngaps.append(abs(pv-nv));ngood+=1
    neigh_gap_mean=float(np.mean(ngaps)) if ngaps else np.nan
    neigh_gap_max=float(np.max(ngaps)) if ngaps else np.nan
    neigh_support=float(ngood/max(ntotal,1)) if ntotal else 0.0
    cand_neigh_mae=np.nan
    if neigh.size:
        m=obs[:,neigh]&cv[:,neigh]
        if np.any(m):cand_neigh_mae=float(np.abs(qx[:,neigh][m]-cx[:,neigh][m]).mean())
    lo,hi=cfg['patch_transport'].get('affine_scale_clip',[0.5,2.0])
    diag={
        'candidate_patch_mean':cm,'candidate_patch_std':cs,'candidate_patch_recent_mean':crm,
        'candidate_patch_recent_std':crs,'candidate_patch_delta_abs_mean':cdelta,
        'candidate_query_mean_gap':abs(cm-qm) if np.isfinite(cm) and np.isfinite(qm) else np.nan,
        'candidate_query_std_gap':abs(cs-qs) if np.isfinite(cs) and np.isfinite(qs) else np.nan,
        'candidate_query_delta_gap':abs(cdelta-qdelta) if np.isfinite(cdelta) and np.isfinite(qdelta) else np.nan,
        'candidate_neighbor_observed_mae':cand_neigh_mae,
        'affine_scale':a,'affine_shift':b,'affine_scale_deviation':abs(a-1.0),'affine_overlap_points':ov,
        'affine_at_clip':float(abs(a-float(lo))<1e-7 or abs(a-float(hi))<1e-7),
        'proposed_fill_count':int(fill.sum()),'proposal_fill_mean':fm,'proposal_fill_std':fs,
        'proposal_abs_change_mean':float(np.mean(np.abs(diff))) if diff.size else np.nan,
        'proposal_abs_change_max':float(np.max(np.abs(diff))) if diff.size else np.nan,
        'proposal_change_rms':float(np.sqrt(np.mean(diff**2))) if diff.size else np.nan,
        'proposal_change_l2':float(np.linalg.norm(diff)) if diff.size else np.nan,
        'proposal_anchor_jump_mean':ajm,'proposal_anchor_jump_max':ajx,
        'proposal_patch_after_mean':am,'proposal_patch_after_std':astd,
        'proposal_obs_mean_gap':abs(am-qm) if np.isfinite(am) and np.isfinite(qm) else np.nan,
        'proposal_neighbor_gap_mean':neigh_gap_mean,'proposal_neighbor_gap_max':neigh_gap_max,
        'proposal_neighbor_support_fraction':neigh_support,
    }
    return after,fill,diag

def deep_features(base_pred,base_pre,pred,pre,nodes):
    bp=np.asarray(base_pred,np.float32);pp=np.asarray(pred,np.float32);d=(pp-bp).astype(np.float64)
    nodes=np.asarray(nodes,np.int64);N=d.shape[-1];outside=np.setdiff1d(np.arange(N,dtype=np.int64),nodes,assume_unique=False)
    pmae=float(np.mean(np.abs(d[...,nodes]))) if nodes.size else np.nan
    omae=float(np.mean(np.abs(d[...,outside]))) if outside.size else 0.0
    dp=np.asarray(pre,np.float32)-np.asarray(base_pre,np.float32)
    lpatch=float(np.mean(np.abs(dp[...,nodes,:]))) if nodes.size else np.nan
    loutside=float(np.mean(np.abs(dp[...,outside,:]))) if outside.size else 0.0
    return {
        'prediction_shift_mae':float(np.mean(np.abs(d))),
        'prediction_shift_rmse':float(np.sqrt(np.mean(d**2))),
        'prediction_shift_max':float(np.max(np.abs(d))),
        'prediction_shift_std':float(np.std(d)),
        'prediction_shift_signed_mean':float(np.mean(d)),
        'prediction_patch_shift_mae':pmae,
        'prediction_outside_patch_shift_mae':omae,
        'prediction_spill_ratio':float(omae/max(pmae,1e-6)) if np.isfinite(pmae) else np.nan,
        'latent_global_mae':float(np.mean(np.abs(dp))),
        'latent_global_rmse':float(np.sqrt(np.mean(dp.astype(np.float64)**2))),
        'latent_global_cosine':_cos(base_pre,pre),
        'latent_patch_mae':lpatch,
        'latent_patch_rmse':float(np.sqrt(np.mean(dp[...,nodes,:].astype(np.float64)**2))) if nodes.size else np.nan,
        'latent_patch_cosine':_cos(np.asarray(base_pre)[...,nodes,:],np.asarray(pre)[...,nodes,:]) if nodes.size else np.nan,
        'latent_outside_patch_mae':loutside,
        'latent_spill_ratio':float(loutside/max(lpatch,1e-6)) if np.isfinite(lpatch) else np.nan,
    }

def main():
    ap=argparse.ArgumentParser(description='Build v0.8 proposal/intervention-aware utility dataset')
    ap.add_argument('--run-dir',required=True);ap.add_argument('--stage',default='development',choices=['development','formal'])
    ap.add_argument('--checkpoint',default=None);ap.add_argument('--input',default=None);ap.add_argument('--output',default=None)
    ap.add_argument('--transport',default='affine_alignment');ap.add_argument('--deep-top-k',type=int,default=20)
    ap.add_argument('--eval-batch-size',type=int,default=64);ap.add_argument('--harmful-margin',type=float,default=0.01)
    a=ap.parse_args();cfg=setup(a.stage);assert_cache_ready(cfg,a.stage);rd=Path(a.run_dir).resolve()
    src=Path(a.input).resolve() if a.input else rd/'predictive_utility_selector'/'utility_action_dataset.csv.gz'
    if not src.exists():raise FileNotFoundError(src)
    out=Path(a.output).resolve() if a.output else rd/'intervention_utility_selector';out.mkdir(parents=True,exist_ok=True)
    log(f'LOAD={src}');df=pd.read_csv(src);df=df[df['rank']<=int(a.deep_top_k)].copy().reset_index(drop=True)
    df=add_utility_targets(df,a.harmful_margin)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adapter,cp,ck,source=load_patchstg_adapter(cfg,a.stage,device,a.checkpoint)
    layout=extract_patch_layout(adapter.model,ck,backbone_root=cfg['project']['backbone_root'])
    train,val=CacheView(cfg,a.stage,'train'),CacheView(cfg,a.stage,'val')
    feature_names=[
        'candidate_patch_mean','candidate_patch_std','candidate_patch_recent_mean','candidate_patch_recent_std','candidate_patch_delta_abs_mean',
        'candidate_query_mean_gap','candidate_query_std_gap','candidate_query_delta_gap','candidate_neighbor_observed_mae',
        'affine_scale','affine_shift','affine_scale_deviation','affine_overlap_points','affine_at_clip','proposed_fill_count','proposal_fill_mean','proposal_fill_std',
        'proposal_abs_change_mean','proposal_abs_change_max','proposal_change_rms','proposal_change_l2','proposal_anchor_jump_mean','proposal_anchor_jump_max',
        'proposal_patch_after_mean','proposal_patch_after_std','proposal_obs_mean_gap','proposal_neighbor_gap_mean','proposal_neighbor_gap_max','proposal_neighbor_support_fraction',
        'prediction_shift_mae','prediction_shift_rmse','prediction_shift_max','prediction_shift_std','prediction_shift_signed_mean','prediction_patch_shift_mae',
        'prediction_outside_patch_shift_mae','prediction_spill_ratio','latent_global_mae','latent_global_rmse','latent_global_cosine','latent_patch_mae','latent_patch_rmse',
        'latent_patch_cosine','latent_outside_patch_mae','latent_spill_ratio',
    ]
    arr={c:np.full(len(df),np.nan,np.float32) for c in feature_names}
    arr['proposed_fill_count']=np.zeros(len(df),np.float32);arr['affine_overlap_points']=np.zeros(len(df),np.float32)
    groups=df.groupby('query_id',sort=False).indices;start=time.time();qtotal=len(groups)
    cand_col=df['candidate_id'].to_numpy(np.int64);patch_col=df['patch_id'].to_numpy(np.int64);scope_col=df['recent_steps'].to_numpy(np.int64)
    for qi,(qid,indices) in enumerate(groups.items(),1):
        qs=time.time();idx=np.asarray(indices,np.int64);q=val.row(int(qid))
        base_pred,base_pre=predict_raw(adapter,ck,q['x_obs'],q['tod_hist'],q['dow_hist'],device,return_prehead=True)
        base_pred=base_pred[0];base_pre=base_pre[0]
        candidate_cache={};variants=[];variant_rows=[]
        for ridx in idx:
            ridx=int(ridx);cid=int(cand_col[ridx]);pid=int(patch_col[ridx]);L=int(scope_col[ridx])
            cand=candidate_cache.get(cid)
            if cand is None:cand=train.row(cid);candidate_cache[cid]=cand
            x,fill,diag=proposal_features(q,cand,layout,pid,L,a.transport,cfg)
            for k,v in diag.items():arr[k][ridx]=np.float32(v) if np.isfinite(v) else np.nan
            if np.any(fill):variants.append(x);variant_rows.append((int(ridx),pid))
        # Forward intervention proposals in bounded chunks and extract representation/output shifts.
        for s in range(0,len(variants),max(1,int(a.eval_batch_size))):
            chunk=variants[s:s+a.eval_batch_size];meta=variant_rows[s:s+a.eval_batch_size]
            X=np.stack(chunk).astype(np.float32);tod=np.repeat(q['tod_hist'][None,:],len(X),axis=0);dow=np.repeat(q['dow_hist'][None,:],len(X),axis=0)
            preds,pres=predict_raw_prehead_batched(adapter,ck,X,tod,dow,device,batch_size=a.eval_batch_size)
            for j,(ridx,pid) in enumerate(meta):
                dd=deep_features(base_pred,base_pre,preds[j],pres[j],layout.groups[int(pid)])
                for k,v in dd.items():arr[k][ridx]=np.float32(v) if np.isfinite(v) else np.nan
        if qi%1==0:
            log(f'QUERY {qi}/{qtotal} id={int(qid)} rows={len(idx)} variants={len(variants)} candidates={len(candidate_cache)} elapsed={time.time()-qs:.1f}s total={time.time()-start:.1f}s {gpu(device)}')
    for c,v in arr.items():df[c]=v
    df=add_v08_features(df)
    full_feats=feature_columns_v08(df,'full');proposal_feats=feature_columns_v08(df,'proposal');metadata_feats=feature_columns_v08(df,'metadata')
    # Hard leakage contract: deep/proposal features must not depend on hidden query history or future target.
    forbidden={'forecast_utility','forecast_mae','baseline_forecast_mae','forecast_gain_pct','patch_completion_mae','beneficial','harmful','relative_utility','utility_target','candidate_id','query_id'}
    if forbidden.intersection(full_feats):raise RuntimeError(f'Leakage feature(s): {sorted(forbidden.intersection(full_feats))}')
    out_csv=out/'intervention_action_dataset.csv.gz';df.to_csv(out_csv,index=False,compression='gzip')
    meta={
        'version':'0.8.0','rows':int(len(df)),'queries':int(df.query_id.nunique()),'patches':int(layout.patch_count),'deep_top_k':int(a.deep_top_k),
        'transport':a.transport,'feature_count_full':len(full_feats),'features_full':full_feats,'feature_count_proposal':len(proposal_feats),
        'feature_count_metadata':len(metadata_feats),'beneficial_rate':float(df.beneficial.mean()),'harmful_rate':float(df.harmful.mean()),
        'checkpoint':str(cp),'checkpoint_source':source,'output':str(out_csv),'elapsed_sec':float(time.time()-start),
        'leakage_contract':'Proposal, prediction-shift, and PatchSTG-prehead shift features are computed only from current observed input/mask, retrieved train episodes, time features, and frozen PatchSTG outputs. Hidden current history and future targets are never v0.8 model features.',
    }
    write_json(out/'dataset_summary.json',meta);log(f'DATASET={out_csv}');log(json.dumps({k:v for k,v in meta.items() if not k.startswith('features_')},indent=2));log('V0.8 DATASET BUILD COMPLETE')
if __name__=='__main__':main()
