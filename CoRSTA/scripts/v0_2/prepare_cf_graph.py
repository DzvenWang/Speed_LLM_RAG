#!/usr/bin/env python3
"""Optional TRAIN-only counterfactual forecast-utility graph builder.

Each edge source j -> predicted node i is measured by withholding source j's
observations from the SAME frozen teacher. It is a forecasting dependency,
NOT a physical causal effect or a replacement for the directional road graph.

This is deliberately offline/optional; formal CoRSTA v0.2 defaults to Road only.
"""
from __future__ import annotations
import argparse,json,os,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from corsta.observability import install_timestamps
install_timestamps()
import numpy as np,torch
from torch.utils.data import DataLoader,Subset
from corsta.data import SupportSplit
from corsta_v2.stla.backbone import make_stla,base_features
from corsta_v2.spatial.graph import build_road
from corsta_v2.util import atomic_json,sha256,source_hash,to_device,seed_all


def cli():
    p=argparse.ArgumentParser()
    p.add_argument('--target',required=True,choices=['speed','flow'])
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--mask-proj-checkpoint',required=True)
    p.add_argument('--formal-cache',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--windows',type=int,default=48,help='Fixed sampled TRAIN windows per intervention')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--candidate-per-node',type=int,default=16)
    p.add_argument('--graph-neighbors',type=int,default=8)
    p.add_argument('--road-per-direction',type=int,default=4)
    p.add_argument('--cpu',action='store_true')
    return p.parse_args()


def candidate_sources(raw,road,indices,k):
    """Road candidates + TRAIN full-history correlations; never val/test."""
    n=road['ids'].shape[0]
    snapshots=[]
    for ix in indices:
        v=np.asarray(raw.x_full[int(ix)],np.float32).copy()
        original=np.asarray(raw.observed[int(ix)],bool).copy()
        if raw.original_valid is not None:
            original|=np.asarray(raw.original_valid[int(ix)],bool)
        v[~original]=np.nan
        snapshots.append(v)
    a=np.concatenate(snapshots,axis=0)
    means=np.nanmean(a,axis=0)
    means=np.nan_to_num(means,0)
    a=np.where(np.isfinite(a),a,means)
    a=a-a.mean(0,keepdims=True)
    a=a/np.maximum(a.std(0,keepdims=True),1e-5)
    corr=np.nan_to_num((a.T@a)/max(len(a),1),0).astype('float32')
    np.fill_diagonal(corr,-np.inf)
    output=[]
    for i in range(n):
        road_ids=road['ids'][i,road['valid'][i]].tolist()
        candidate=np.argsort(-np.abs(corr[i]))[:k].tolist()
        sources=[]
        for j in road_ids+candidate:
            if j!=i and j not in sources:sources.append(int(j))
        output.append(sources)
    return output


def main():
    a=cli()
    if a.windows<4 or a.batch_size<1 or a.graph_neighbors<1:raise ValueError('Invalid build budget')
    root=Path(os.environ.get('SPEED_LLM_RAG_ROOT',ROOT.parent)).resolve()
    seed_all(a.seed)
    device=torch.device('cpu' if a.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type!='cuda' and not a.cpu:raise RuntimeError('Need CUDA, or explicitly --cpu')
    ck=Path(a.backbone_checkpoint).resolve();mask=Path(a.mask_proj_checkpoint).resolve()
    raw=SupportSplit(a.formal_cache,'train',cf_training=True)
    n=int(raw.x_obs.shape[2]);count=min(len(raw),a.windows)
    rng=np.random.default_rng(a.seed+3401)
    indices=np.sort(rng.choice(len(raw),size=count,replace=False)).tolist()
    stla,bc=make_stla(root,ck,mask,device)
    if stla.base.num_nodes!=n:raise RuntimeError('STLA/graph nodes mismatch')
    road=build_road(root,a.target,a.dataset,int(a.road_per_direction))
    candidates=candidate_sources(raw,road,indices,a.candidate_per_node)
    by_source=defaultdict(list)
    for i,sources in enumerate(candidates):
        for j in sources:by_source[j].append(i)
    # Cache a small set of TRAIN-only forward results in host RAM.
    samples=[]
    for ix in indices:
        b=raw[int(ix)]
        samples.append({'x_full':b['x_full_training_only'],
                        'valid':b['original_valid'].float(),
                        'tod':b['tod'],'dow':b['dow'],
                        'y':b['y_norm'],'future_valid':b['future_valid']})
    batches=[]
    for start in range(0,len(samples),a.batch_size):
        subset=samples[start:start+a.batch_size]
        b={key:torch.stack([item[key] for item in subset],0).to(device) for key in subset[0]}
        b['y']=torch.where(b['future_valid'].bool(),b['y'],torch.zeros_like(b['y']))
        with torch.no_grad():
            _,_,full=base_features(stla,b['x_full'],b['valid'],b['tod'],b['dow'])
        batches.append((b,full))
    sensitivity=np.zeros((n,n),np.float64);benefit=np.zeros((n,n),np.float64)
    positive=np.zeros((n,n),np.float64);counts=np.zeros((n,n),np.float64)
    stla.eval()
    with torch.no_grad():
        for counter,j in enumerate(sorted(by_source),1):
            targets=by_source[j]
            for b,pf in batches:
                altered=b['valid'].clone();altered[:,:,j]=0
                x_altered=torch.where(altered.bool(),b['x_full'],torch.zeros_like(b['x_full']))
                _,_,pm=base_features(stla,x_altered,altered,b['tod'],b['dow'])
                ids=torch.tensor(targets,device=device,dtype=torch.long)
                v=b['future_valid'][:,:,ids].float()
                denom=v.sum((0,1)).clamp_min(1)
                sens=(((pf[:,:,ids]-pm[:,:,ids]).abs())*v).sum((0,1))/denom
                gain=(((pm[:,:,ids]-b['y'][:,:,ids]).abs()-(pf[:,:,ids]-b['y'][:,:,ids]).abs())*v).sum((0,1))/denom
                good=((((pm[:,:,ids]-b['y'][:,:,ids]).abs()-(pf[:,:,ids]-b['y'][:,:,ids]).abs())>0).float()*v).sum((0,1))/denom
                sensitivity[j,targets]+=sens.cpu().numpy()
                benefit[j,targets]+=gain.cpu().numpy()
                positive[j,targets]+=good.cpu().numpy()
                counts[j,targets]+=1
            if counter%25==0:print(f'CORSTA_V2_CF_GRAPH_PROGRESS sources={counter}/{len(by_source)}',flush=True)
    sens=np.divide(sensitivity,counts,out=np.zeros_like(sensitivity),where=counts>0)
    gain=np.divide(benefit,counts,out=np.zeros_like(benefit),where=counts>0)
    stability=np.divide(positive,counts,out=np.zeros_like(positive),where=counts>0)
    k=int(a.graph_neighbors)
    ids=np.zeros((n,k),np.int64);valid=np.zeros((n,k),bool)
    score_s=np.zeros((n,k),np.float32);score_g=np.zeros((n,k),np.float32)
    for i,sources in enumerate(candidates):
        options=[j for j in sources if counts[j,i]>0 and gain[j,i]>0 and stability[j,i]>.5]
        options.sort(key=lambda j:(gain[j,i]*stability[j,i],sens[j,i],-j),reverse=True)
        for z,j in enumerate(options[:k]):
            ids[i,z]=j;valid[i,z]=True;score_s[i,z]=sens[j,i];score_g[i,z]=gain[j,i]
    out=Path(a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise RuntimeError('Graph already exists; refusing to overwrite: '+str(out))
    np.savez_compressed(out,source_ids=ids,valid=valid,sensitivity=score_s,gain=score_g)
    atomic_json({'schema':1,'target':a.target,'dataset':a.dataset,
                 'num_windows':count,'seed':a.seed,'source':'TRAIN only',
                 'checkpoint_sha256':sha256(ck),'mask_proj_sha256':sha256(mask),
                 'graph_sha256':sha256(out),'edge_count':int(valid.sum()),
                 'note':'Forecast sensitivity / gain under observed-node removal; not physical causation'},
                out.with_suffix('.json'))
    print(f'CORSTA_V2_CF_GRAPH_PASS nodes={n} edges={int(valid.sum())} output={out}',flush=True)

if __name__=='__main__':main()
