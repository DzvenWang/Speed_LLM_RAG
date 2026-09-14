#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, json, numpy as np, torch
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe'); ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml, write_json, stamp
from presure.contract import cache_dir
from presure.data import CacheSplit
from presure.metrics import metric_bundle
from presure.pipeline import load_artifacts, run_dir
from presure.oracles import (
    oracle_global_batch, oracle_horizon_batch, oracle_node_horizon_batch,
    oracle_node_batch, oracle_region_horizon_batch,
    per_query_mae_np, relative_improvements,
)


def tt(a, device, dtype=None):
    if dtype is None: dtype=torch.float32
    return torch.tensor(np.asarray(a), device=device, dtype=dtype)


def candidate_file(art, typ):
    if typ in ('A','D','E','F'): return art/'val_candidates.npz'
    if typ in ('B','C'): return art/'val_oracle_candidates.npz'
    raise ValueError(typ)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',default='development',choices=['development','formal'])
    ap.add_argument('--seed',type=int,default=10)
    ap.add_argument('--oracle-type',default='A',choices=['A','B','C','D','E','F'])
    a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/presure.yaml'); art=load_artifacts(cfg,a.stage)
    cache=cache_dir(cfg,a.stage); ds=CacheSplit(cache,'val')
    base=np.load(art/'val_base_pred.npy',mmap_mode='r')
    res=np.load(art/'train_residual.npy',mmap_mode='r')
    regions=np.load(art/'node_regions.npy')
    cf=candidate_file(art,a.oracle_type)
    if not cf.exists(): raise RuntimeError(f'Missing {cf}; rebuild retrieval candidates with PreSuRe v0.3.0')
    cand=np.load(cf)['indices']
    oc=cfg['oracle']; alphas=[float(x) for x in oc['alpha_grid']]; w=oc['objective_weights']
    if a.oracle_type=='B': bs=int(oc['expanded_batch_size'])
    else: bs=int(oc['local_batch_size'] if a.oracle_type in ('C','D','E','F') else oc['batch_size'])
    if a.oracle_type=='C': cand=cand[:,:min(int(oc['horizon_candidate_k']),cand.shape[1])]
    elif a.oracle_type=='D': cand=cand[:,:min(int(oc['node_horizon_candidate_k']),cand.shape[1])]
    elif a.oracle_type=='E': cand=cand[:,:min(int(oc['node_candidate_k']),cand.shape[1])]
    elif a.oracle_type=='F': cand=cand[:,:min(int(oc['region_horizon_candidate_k']),cand.shape[1])]
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    final=np.empty_like(np.asarray(base,np.float32))
    q_gains=[]; improved_cells=0; valid_cells=0; masked_improved=0; masked_cells=0; support_improved=0; support_cells=0
    if a.oracle_type in ('A','B'):
        choice_idx=np.full(len(ds),-1,np.int32); choice_alpha=np.zeros(len(ds),np.float32)
    elif a.oracle_type=='C':
        P=int(cfg['experiment']['horizon']); choice_idx=np.full((len(ds),P),-1,np.int32); choice_alpha=np.zeros((len(ds),P),np.float32)
    elif a.oracle_type=='E':
        N=int(np.asarray(base).shape[2]); choice_idx=np.full((len(ds),N),-1,np.int32); choice_alpha=np.zeros((len(ds),N),np.float32)
    elif a.oracle_type=='F':
        P=int(cfg['experiment']['horizon']); R=int(regions.max())+1; choice_idx=np.full((len(ds),P,R),-1,np.int32); choice_alpha=np.zeros((len(ds),P,R),np.float32)
    else:
        choice_idx=choice_alpha=None
    for s in range(0,len(ds),bs):
        e=min(len(ds),s+bs); z=slice(s,e)
        b=tt(base[z],device); y=tt(ds.y[z],device); valid=tt(ds.valid[z],device,torch.bool)
        direct=tt(ds.direct[z],device,torch.bool); support=tt(ds.support[z],device,torch.bool)
        ids=cand[z]; rr=tt(res[ids],device)
        if a.oracle_type in ('A','B'):
            fout,bi,ba,gain=oracle_global_batch(b,y,valid,direct,support,rr,alphas,w)
            bix=bi.cpu().numpy(); choice_idx[z]=np.where(bix>=0,ids[np.arange(len(ids)),np.clip(bix,0,None)],-1); choice_alpha[z]=ba.cpu().numpy(); q_gains.extend(gain.cpu().numpy().tolist())
        elif a.oracle_type=='C':
            fout,bi,ba,gain=oracle_horizon_batch(b,y,valid,direct,support,rr,alphas,w)
            bix=bi.cpu().numpy(); safe=np.clip(bix,0,None); choice_idx[z]=np.where(bix>=0,np.take_along_axis(ids,safe,axis=1),-1); choice_alpha[z]=ba.cpu().numpy(); q_gains.extend(gain.mean(1).cpu().numpy().tolist())
        elif a.oracle_type=='E':
            fout,bi,ba,gain=oracle_node_batch(b,y,valid,rr,alphas)
            bix=bi.cpu().numpy(); safe=np.clip(bix,0,None); choice_idx[z]=np.where(bix>=0,np.take_along_axis(ids,safe,axis=1),-1); choice_alpha[z]=ba.cpu().numpy(); q_gains.extend(gain.mean(1).cpu().numpy().tolist())
        elif a.oracle_type=='F':
            fout,bi,ba,gain=oracle_region_horizon_batch(b,y,valid,direct,support,rr,alphas,regions,w)
            bix=bi.cpu().numpy(); safe=np.clip(bix,0,None)
            # ids [B,K], safe [B,P,R] -> global train-memory indices.
            gid=ids[np.arange(len(ids))[:,None,None],safe]
            choice_idx[z]=np.where(bix>=0,gid,-1); choice_alpha[z]=ba.cpu().numpy(); q_gains.extend(gain.mean((1,2)).cpu().numpy().tolist())
        else:
            fout,bi,ba,cell_gain=oracle_node_horizon_batch(b,y,valid,rr,alphas)
            cg=cell_gain.cpu().numpy(); vv=np.asarray(ds.valid[z],bool); dm=np.asarray(ds.direct[z],bool)[:,None,:]; sm=np.asarray(ds.support[z],bool)[:,None,:]
            improved_cells += int(((cg>0)&vv).sum()); valid_cells += int(vv.sum())
            masked_improved += int(((cg>0)&vv&dm).sum()); masked_cells += int((vv&dm).sum())
            support_improved += int(((cg>0)&vv&sm).sum()); support_cells += int((vv&sm).sum())
        final[z]=fout.cpu().numpy()
        if s % max(bs*25,1) == 0: print(f'Oracle-{a.oracle_type} {s}/{len(ds)} K={cand.shape[1]}',flush=True)
    base_arr=np.asarray(base,np.float32); y_arr=np.asarray(ds.y); valid_arr=np.asarray(ds.valid,bool)
    m0=metric_bundle(cfg,base_arr,y_arr,valid_arr,np.asarray(ds.direct,bool),np.asarray(ds.support,bool))
    mo=metric_bundle(cfg,final,y_arr,valid_arr,np.asarray(ds.direct,bool),np.asarray(ds.support,bool))
    bq=per_query_mae_np(base_arr,y_arr,valid_arr); fq=per_query_mae_np(final,y_arr,valid_arr)
    variant=f'Oracle{a.oracle_type}'; rd=run_dir(cfg,variant,a.stage,stamp(),a.seed); rd.mkdir(parents=True,exist_ok=True)
    np.save(rd/'val_predictions.npy',final.astype(np.float32))
    if choice_idx is not None: np.savez_compressed(rd/'oracle_choices.npz',memory_index=choice_idx,alpha=choice_alpha)
    summary={
        'variant':variant,'oracle_type':a.oracle_type,'stage':a.stage,'seed':a.seed,
        'base':m0,'oracle':mo,'relative_improvement':relative_improvements(m0,mo),
        'queries_improved_fraction':float(np.mean(fq < bq-1e-8)),
        'candidate_k':int(cand.shape[1]),'alpha_grid':alphas,
        'oracle_uses_validation_future':True,'deployable':False,'test_evaluated':False,
    }
    if q_gains: summary['mean_local_objective_gain']=float(np.mean(q_gains))
    descriptions={
        'A':'Query-global Top-64 oracle: one support residual + one global alpha per query.',
        'B':'Expanded query-global oracle: one support residual + one global alpha over a larger leakage-safe pool.',
        'C':'Horizon-local oracle: each forecast horizon chooses support and alpha independently.',
        'D':'Node-horizon oracle: every valid forecast cell chooses support and alpha independently; intentionally very loose.',
        'E':'Node-local oracle: each node chooses one support and alpha shared across all horizons.',
        'F':'Graph-region × horizon oracle: each graph routing region and horizon chooses support and alpha independently.',
    }
    summary['description']=descriptions[a.oracle_type]
    if a.oracle_type=='F':
        summary['n_graph_regions']=int(regions.max())+1
        summary['region_manifest']=json.loads((art/'region_manifest.json').read_text())
    if a.oracle_type=='D':
        summary['valid_cell_improved_fraction']=float(improved_cells/max(valid_cells,1))
        summary['masked_cell_improved_fraction']=float(masked_improved/max(masked_cells,1))
        summary['support_cell_improved_fraction']=float(support_improved/max(support_cells,1))
    write_json(rd/'summary.json',summary); print(json.dumps(summary,indent=2)); print(f'{variant} SUPPORT UPPER BOUND PASS')


if __name__=='__main__': main()
