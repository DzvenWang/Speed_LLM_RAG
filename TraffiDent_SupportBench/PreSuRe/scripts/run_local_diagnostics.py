#!/usr/bin/env python3
"""PreSuRe v0.7 local-support diagnostics.

Validation futures are used only for non-deployable diagnosis:
  E24             : node-local Oracle-E over the first 24 coarse candidates.
  E_M3TOP4        : node-local oracle over M3-ranked Top-4 supports.
  D_M3TOP4        : node x horizon oracle over M3-ranked Top-4 supports.
  SOFTMIX_M3TOP4  : future-labelled soft local-utility mixture over M3 Top-4,
                    matching the SoftMix teacher retained by the v0.7 pairwise M4 student.
For M3-Top4 diagnostics both ranking-only and post-eligibility views are saved.
"""
from pathlib import Path
import argparse,sys,json,numpy as np,torch
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe'); ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from presure.utils import load_yaml,write_json,stamp
from presure.contract import cache_dir
from presure.data import CacheSplit
from presure.metrics import metric_bundle
from presure.pipeline import load_artifacts,run_dir,load_m1,load_m3
from presure.oracles import oracle_node_batch,oracle_node_selected_batch,oracle_node_horizon_selected_batch,relative_improvements
import run_variant as rv


def tt(a,dev,dtype=torch.float32):
    # memmap-backed arrays may be read-only; make an explicit writable copy before torch conversion.
    return torch.as_tensor(np.array(a,copy=True),device=dev,dtype=dtype)

def mb(cfg,pred,ds): return metric_bundle(cfg,pred,np.asarray(ds.y),np.asarray(ds.valid,bool),np.asarray(ds.direct,bool),np.asarray(ds.support,bool))

def _softmix(cfg,b,y,v,rr,mask):
    c=cfg['model']['m4']; alphas=[float(x) for x in c['teacher_alpha_grid']]
    probs,tdelta,tcorr,tmass,hard,rel,ent=rv.m4_soft_teacher_targets(
        b,y,v,rr,mask,alphas,
        temperature=float(c['teacher_temperature']),
        mass_temperature=float(c['teacher_mass_temperature']),
        min_abs_gain=float(c['teacher_min_abs_gain']),
        gain_floor=float(c['teacher_gain_floor']),
        utility_clip=float(c['teacher_utility_clip']))
    return b+tcorr.permute(0,2,1),tmass,ent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development',choices=['development','formal']);ap.add_argument('--seed',type=int,default=10);ap.add_argument('--diagnostic',required=True,choices=['E24','E_M3TOP4','D_M3TOP4','SOFTMIX_M3TOP4']);a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/presure.yaml'); art=load_artifacts(cfg,a.stage); ds=CacheSplit(cache_dir(cfg,a.stage),'val'); dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    base=np.load(art/'val_base_pred.npy',mmap_mode='r'); res=np.load(art/'train_residual.npy',mmap_mode='r'); cand=np.load(art/'val_candidates.npz'); alphas=[float(x) for x in cfg['oracle']['alpha_grid']]; bs=int(cfg['diagnostics']['batch_size']); base_metric=mb(cfg,np.asarray(base,np.float32),ds)
    final=np.empty_like(np.asarray(base,np.float32)); final_masked=None; eligible=[]
    if a.diagnostic=='E24':
        K=min(int(cfg['diagnostics']['oracle_e_candidate_k']),cand['indices'].shape[1]); idsall=np.asarray(cand['indices'][:,:K],np.int64)
        for s in range(0,len(ds),bs):
            e=min(len(ds),s+bs); z=slice(s,e); b=tt(base[z],dev); y=tt(ds.y[z],dev); v=tt(ds.valid[z],dev,torch.bool); rr=tt(res[idsall[z]],dev); out,_,_,_=oracle_node_batch(b,y,v,rr,alphas); final[z]=out.cpu().numpy()
        name='DiagOracleE24'; summary={'variant':name,'diagnostic':'Oracle-E@24','candidate_k':K,'stage':a.stage,'seed':a.seed,'base':base_metric,'oracle':mb(cfg,final,ds),'validation_future_used_for_diagnostic_only':True,'deployable':False,'test_evaluated':False}
        summary['relative_improvement']=relative_improvements(base_metric,summary['oracle'])
    else:
        m1,_,_=load_m1(cfg,a.stage,dev);m3,_,_=load_m3(cfg,a.stage,dev);zt,zv=rv.ensure_embeddings(cfg,a.stage,art,m1,dev,False);nft,nfv=rv._feature_mmaps(art);L=int(cfg['diagnostics']['m3_top_k']); final_masked=np.empty_like(np.asarray(base,np.float32)); masses=[]; masses_f=[]; ents=[]; ents_f=[]
        for s in range(0,len(ds),bs):
            e=min(len(ds),s+bs); ids=np.arange(s,e); b=tt(base[ids],dev);y=tt(ds.y[ids],dev);v=tt(ds.valid[ids],dev,torch.bool)
            with torch.no_grad(): gid,csel,cssel,conf,m3scale,cmask,rr=rv.m3_select_for_router(m3,zt,zv,nft,nfv,cand,res,ids,cfg,dev)
            if rr.shape[2]>L: rr=rr[:,:,:L];cmask=cmask[:,:,:L]
            eligible.append(float(cmask.float().mean().cpu()))
            if a.diagnostic=='E_M3TOP4':
                out,_,_,_=oracle_node_selected_batch(b,y,v,rr,alphas,None); outm,_,_,_=oracle_node_selected_batch(b,y,v,rr,alphas,cmask); name='DiagOracleE_M3Top4'; label='Oracle-E@M3-Top4'
            elif a.diagnostic=='D_M3TOP4':
                out,_,_,_=oracle_node_horizon_selected_batch(b,y,v,rr,alphas,None); outm,_,_,_=oracle_node_horizon_selected_batch(b,y,v,rr,alphas,cmask); name='DiagOracleD_M3Top4'; label='Oracle-D@M3-Top4'
            else:
                allmask=torch.ones_like(cmask,dtype=torch.bool); out,mass,ent=_softmix(cfg,b,y,v,rr,allmask); outm,massf,entf=_softmix(cfg,b,y,v,rr,cmask); masses.append(float(mass.mean().cpu()));masses_f.append(float(massf.mean().cpu()));ents.append(float(ent.mean().cpu()));ents_f.append(float(entf.mean().cpu()));name='DiagOracleSoftMix_M3Top4';label='Soft-Utility-Mix@M3-Top4'
            final[ids]=out.cpu().numpy();final_masked[ids]=outm.cpu().numpy()
        unfiltered=mb(cfg,final,ds);filtered=mb(cfg,final_masked,ds);summary={'variant':name,'diagnostic':label,'candidate_k':L,'stage':a.stage,'seed':a.seed,'base':base_metric,'oracle_unfiltered_top4':unfiltered,'oracle_after_router_threshold':filtered,'relative_improvement_unfiltered':relative_improvements(base_metric,unfiltered),'relative_improvement_after_router_threshold':relative_improvements(base_metric,filtered),'mean_candidate_eligibility_fraction':float(np.mean(eligible)),'validation_future_used_for_diagnostic_only':True,'deployable':False,'test_evaluated':False}
        if a.diagnostic=='SOFTMIX_M3TOP4': summary.update({'teacher_mode':'soft_local_utility_distillation','softmix_support_mass_unfiltered':float(np.mean(masses)),'softmix_support_mass_after_threshold':float(np.mean(masses_f)),'softmix_entropy_unfiltered':float(np.mean(ents)),'softmix_entropy_after_threshold':float(np.mean(ents_f))})
    rd=run_dir(cfg,name,a.stage,stamp(),a.seed);rd.mkdir(parents=True,exist_ok=True);np.save(rd/'val_predictions.npy',final.astype(np.float32))
    if final_masked is not None: np.save(rd/'val_predictions_after_router_threshold.npy',final_masked.astype(np.float32))
    write_json(rd/'summary.json',summary);print(json.dumps(summary,indent=2));print(f'{name} PASS run_dir={rd}')

if __name__=='__main__': main()
