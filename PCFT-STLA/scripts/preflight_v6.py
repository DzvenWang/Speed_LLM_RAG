#!/usr/bin/env python3
"""Lightweight fail-fast preflight without materializing the 50k-row multivariate cache."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np,torch,pandas as pd
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from pcft_stla.v6.timestamp import install
install()
from pcft_stla.v6.config import load_config,make_paths,resolve_checkpoint,sha256
from pcft_stla.v6.data import open_bench,make_multi_cache,MultiTargetSplit
from pcft_stla.v6.model import load_original_stla,PCFTV6
from pcft_stla.v6.graph import physical_only_graph,graph_audit,DirectedGraph
from pcft_stla.v6.graph_build.functional import build_functional


def preflight(cfg,explicit):
    repo,formal,evs=make_paths(cfg,cfg['experiment']['target'],cfg['experiment']['dataset'])
    ck=resolve_checkpoint(repo,cfg,explicit)
    bench=open_bench(repo,cfg['experiment']['dataset'],target=cfg['experiment']['target'])
    graph=physical_only_graph(bench)
    print('GRAPH_PHYSICAL_PREFLIGHT_PASS',json.dumps(graph_audit(bench,graph)),flush=True)
    if graph.num_nodes!=bench.N:raise ValueError('Graph num_nodes mismatch')
    target=cfg['experiment']['target'];idx=bench.channel_names.index(target)
    from supportbench.sensing import materialize_mask
    for label,r,split in [('train',formal,'train'),('val',formal,'val')]+[(f's{s:02d}',p,'test') for s,p in evs.items()]:
        manifest=pd.read_csv(r/f'{split}_manifest.csv',dtype={'view_id':str})
        x=np.load(r/f'{split}_x_obs_norm.npy',mmap_mode='r')
        m=np.load(r/f'{split}_history_observed.npy',mmap_mode='r')
        direct=np.load(r/f'{split}_direct_mask.npy',mmap_mode='r')
        valid=np.load(r/f'{split}_future_valid.npy',mmap_mode='r')
        if x.shape[1:]!=(bench.H,bench.N) or m.shape!=x.shape or valid.shape[1:]!=(bench.P,bench.N):
            raise ValueError(f'{label}: shared cache tensor shapes invalid')
        if direct.shape!=(len(manifest),bench.N):raise ValueError(f'{label}: direct mask shape mismatch')
        points=sorted(set([0,min(1,len(manifest)-1),len(manifest)//2,len(manifest)-1]))
        for i in points:
            row=manifest.iloc[i];spec=bench.get_view_spec(str(row.view_id));st=int(spec['window_start'])
            if st!=int(row.window_start) or str(spec['split'])!=split:raise ValueError(f'{label}: view identity mismatch row={i}')
            raw=np.asarray(bench.traffic[st:st+bench.H],dtype=np.float32)
            rv=np.asarray(bench.valid[st:st+bench.H],dtype=bool)
            art=materialize_mask(spec,bench.H,bench.N,bench.sg_dir,rv[...,idx])
            obs=rv[...,idx]&(~art)
            stat=bench.stats['channels'][target]
            target_x=(raw[...,idx]-float(stat['mean']))/max(float(stat['std']),1e-6)
            target_x=np.where(obs,target_x,0.)
            if not np.allclose(target_x,x[i],rtol=1e-4,atol=1e-4) or not np.array_equal(obs,m[i].astype(bool)):
                raise ValueError(f'{label}: target cache channel/mask mismatch row={i}')
        print(f'PREFLIGHT_CACHE_PASS {label} rows={len(manifest)} N={bench.N}',flush=True)
    # Exercise the REAL functional graph constructor with a small TRAIN-only
    # observed-only input cache, before users queue a long GPU allocation.
    # Separate identity/output namespace; no old PCFT result is reused.
    sketch_root=repo/'PCFT-STLA_cache'/'v6'/'preflight'/(target+'_'+cfg['experiment']['dataset']+'_'+sha256(formal/'cache_manifest.json')[:12])
    pre_multi=make_multi_cache(bench,formal,'train',sketch_root,max_rows=64)
    pre_ds=MultiTargetSplit(formal,'train',pre_multi,limit=64)
    fi,fa,faudit=build_functional(pre_ds,graph.edge_index.numpy(),
        topk=int(cfg['graph']['functional_topk']),max_lag=int(cfg['model']['max_lag']),
        max_timestamps=512,shortlist=int(cfg['graph']['functional_shortlist']),
        min_pairs=4,min_abs_corr=0.01)
    graph=DirectedGraph(graph.edge_index,graph.edge_attr,graph.num_nodes,
                        torch.from_numpy(fi).long(),torch.from_numpy(fa).float())
    print('GRAPH_FUNCTIONAL_PREFLIGHT_PASS',json.dumps({
        'observed_train_rows':64,'functional_sample_edges':int(fi.shape[1]),
        'train_timestamps':faudit['train_timestamps']}),flush=True)
    base,raw=load_original_stla(repo,ck)
    if int(raw['num_nodes'])!=bench.N:raise RuntimeError('Backbone checkpoint graph shape mismatch')
    model=PCFTV6(base,graph,idx,max_lag=int(cfg['model']['max_lag']))
    samples=1;H=12;N=bench.N;C=3
    # Derive a REAL sample for an input-only forward with no future ground truth.
    row=pd.read_csv(formal/'train_manifest.csv',dtype={'view_id':str}).iloc[0]
    spec=bench.get_view_spec(str(row.view_id));s=int(row.window_start)
    hist=np.asarray(bench.traffic[s:s+H],np.float32);rv=np.asarray(bench.valid[s:s+H],bool)
    art=materialize_mask(spec,H,N,bench.sg_dir,rv[...,idx]);obs=rv&(~art[...,None])
    xx=np.where(obs,bench.normalize(hist),0.).astype('float32')
    # Missing age here only needs to be well-formed, full training cache has exact ages.
    ag=np.zeros_like(xx);prev=np.zeros((N,C),np.float32)
    for t in range(H):prev=np.where(obs[t],0,np.minimum(prev+1,H));ag[t]=prev/H
    tod=np.asarray(np.load(formal/'train_tod_hist.npy',mmap_mode='r')[0],dtype=np.int64)
    dow=np.asarray(np.load(formal/'train_dow_hist.npy',mmap_mode='r')[0],dtype=np.int64)
    batch={'x_multi':torch.from_numpy(xx[None]),'observed_mask':torch.from_numpy(obs.astype(np.float32)[None]),
           'missing_age':torch.from_numpy(ag[None]),'x_target':torch.from_numpy(xx[None,...,idx].copy()),
           'tod':torch.from_numpy(tod[None]),'dow':torch.from_numpy(dow[None])}
    model.eval()
    with torch.no_grad():
        for stage in model.STAGES:
            model.set_stage(stage)
            y=model(batch)['prediction']
            if y.shape!=(1,12,N) or not torch.isfinite(y).all():raise RuntimeError(f'{stage} invalid output')
    print('V6_PREFLIGHT_PASS',json.dumps({'target':target,'dataset':cfg['experiment']['dataset'],
        'checkpoint':str(ck),'N':N,'E':int(graph.edge_index.shape[1]),'stages':model.STAGES}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target',required=True,choices=['speed','flow'])
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--backbone-checkpoint')
    p.add_argument('--config',default=str(PROJECT/'configs/pcft_v6.yaml'))
    a=p.parse_args()
    os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(a.config,target=a.target,dataset=a.dataset,seed=a.seed)
    preflight(cfg,a.backbone_checkpoint)
if __name__=='__main__':main()
