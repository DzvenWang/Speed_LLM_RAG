#!/usr/bin/env python3
"""Frozen v7.1 C1 diagnostic and C1-only held-out TEST evaluation.

This is an ADD-ON script: no changes to code-digested v7.1 training files.
Selection diagnostics are descriptive and never alter model weights, cached sources,
or final TEST mode selection. TEST uses a fixed checkpoint and predeclared metrics.
"""
from __future__ import annotations
import argparse, csv, gc, hashlib, json, math, os, sys, time
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path: sys.path.insert(0, str(PROJ))
from run_v7 import Params, code_digest
from tcrr_stla.config import cache_path, file_sha256
from tcrr_stla.data import SupportSplit
from tcrr_stla.v7.version import VERSION
from tcrr_stla.v7.runner import (
    as_device, selection_indices, load_network, new_network, make_teacher,
    train_cache, graph_file, seed_everything, atomic_json,
)
from tcrr_stla.v7.replay import ReplayDataset, prepare_replay, KEYS
from tcrr_stla.v7.graph import graph_identity
from tcrr_stla.v7.losses import full_activation_targets, ranking_targets

SCALES = (20, 40, 60, 90)
METHODS = ('backbone', 'normal', 'learned', 'uniform_keep_none', 'permute_keep_none',
           'top1_keep_none', 'none_only', 'normal_component_only', 'cf_component_only',
           'shuffled_cf', 'support_occlusion')
TEST_METHODS = ('original_stlaformer', 'backbone', 'normal', 'c1')
VERSION_ANALYSIS = 'c1_frozen_1.0.0'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('verify', 'diagnostic', 'test', 'all', 'aggregate'), default='all')
    p.add_argument('--target', choices=('speed', 'flow'), required=True)
    p.add_argument('--backbone-checkpoint', required=True)
    p.add_argument('--dataset', default='TraffiDent_Net_06')
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--config', default=str(PROJ/'configs/tcrr_stla_v7.yaml'))
    p.add_argument('--run-dir', help='Existing v7.1 C1 formal run; NEVER a new run')
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--eval-batch-size', type=int, default=8)
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--severities', default='20,40,60,90')
    p.add_argument('--block-steps', type=int, default=288,
                   help='Fixed chronological bootstrap block width in original 5-min steps')
    p.add_argument('--bootstrap', type=int, default=400)
    p.add_argument('--rank-samples-per-batch', type=int, default=32)
    p.add_argument('--cache-only', action='store_true',
                   help='Only use already-complete TEST replay/teacher caches; do not build')
    p.add_argument('--no-original-stla', action='store_true',
                   help='Skip extra original frozen STLAformer forward (S0/Normal/C1 still reported)')
    p.add_argument('--force-output', action='store_true',
                   help='Only regenerate analysis files after checking frozen checkpoint identity')
    return p.parse_args()


def sha(path): return file_sha256(path)


def require(cond, message):
    if not cond: raise RuntimeError(message)


def manifest_params(a):
    os.environ['TRAFFIDENT_TARGET'] = a.target
    os.environ.setdefault('BACKBONE_MODE', 'incomplete')
    run = (Path(a.run_dir).resolve() if a.run_dir else
           Path(os.environ.get('SPEED_LLM_RAG_ROOT', PROJ.parent))/'TCRR-STLA_exps'/a.target/
           os.environ['BACKBONE_MODE']/a.dataset/'v7_1_0_c1'/'formal'/f'seed{a.seed}')
    identity = run/'run_identity.json'
    require(identity.is_file(), f'Existing v7.1 run_identity.json required: {identity}')
    saved = json.loads(identity.read_text())
    er = saved.get('effective_runtime', {})
    limits = saved.get('limits', {})
    ns = SimpleNamespace(target=a.target, dataset=a.dataset, seed=a.seed, config=a.config,
                         backbone_checkpoint=a.backbone_checkpoint, run_dir=str(run),
                         end_stage=saved.get('end_stage','c1'), smoke=bool(saved.get('smoke',False)),
                         batch_size=int(er['batch_size']), workers=int(er['workers']),
                         limit_train=limits.get('train'),limit_val=limits.get('val'),
                         limit_test=limits.get('test'))
    params=Params(ns)
    require(saved.get('version')==VERSION, 'Not a v7.1.0 run')
    require(saved.get('code_sha256')==code_digest(),
            'SOURCE SHA MISMATCH. Restore the exact v7.1 training source; do not modify its files.')
    for k,v in [('target',a.target),('dataset',a.dataset),('seed',a.seed),('smoke',ns.smoke)]:
        require(saved.get(k)==v, f'Run identity mismatch: {k}')
    require(saved.get('backbone_sha256')==sha(params.backbone_checkpoint),
            'Original Backbone best.pt SHA mismatch')
    require(int(er.get('k',-1))==params.k, 'Candidate K differs from frozen run')
    require(str(saved.get('configuration',{}).get('training',{}).get('candidate_target_temperature')) ==
            str(params.target_temperature), 'Candidate target temperature mismatch')
    require((run/'best_c1.pt').is_file() and (run/'s0.pt').is_file(),
            'Frozen best_c1.pt and s0.pt must exist')
    ck=torch.load(run/'best_c1.pt',map_location='cpu',weights_only=False)
    require(ck.get('version')==VERSION and ck.get('backbone_sha256')==sha(params.backbone_checkpoint),
            'C1 checkpoint version/backbone mismatch')
    require(ck.get('run_identity_sha256')==sha(identity),
            'C1 checkpoint run identity SHA mismatch')
    s0=torch.load(run/'s0.pt',map_location='cpu',weights_only=False)
    require(s0.get('backbone_sha256')==sha(params.backbone_checkpoint), 'S0 backbone mismatch')
    require(s0.get('run_identity_sha256')==sha(identity), 'S0 run identity mismatch')
    require((run/'train_only_neighbours.pt').exists(), 'Train functional graph missing')
    require((run/'graph/graph_context.npz').exists() and (run/'graph/graph_identity.json').exists(),
            'Frozen road graph artifacts missing')
    stamp={'analysis_version':VERSION_ANALYSIS,'target':a.target,'dataset':a.dataset,'seed':a.seed,
           'checkpoint_sha256':sha(run/'best_c1.pt'), 's0_sha256':sha(run/'s0.pt'),
           'backbone_sha256':sha(params.backbone_checkpoint),'run_identity_sha256':sha(identity),
           'training_code_sha256':code_digest(), 'analysis_script_sha256':sha(Path(__file__))}
    return params,stamp


def load_frozen_graph(params):
    """Reads EXISTING graph only. No graph generation in diagnostic evaluator."""
    graph_obj=torch.load(params.run/'train_only_neighbours.pt',map_location='cpu',weights_only=False)
    original=graph_obj['identity']
    expected={'file_sha256':sha(cache_path(params.config,'formal')/'train_x_full_norm.npy'),
              'queries':int(original['queries']), 'neighbors':params.graph_neighbours,
              'nodes':params.nodes,
              'graph_algorithm_sha256':sha(PROJ/'tcrr_stla/v7/model.py')}
    require(original==expected, 'Frozen functional graph identity does not match source')
    train=SupportSplit(cache_path(params.config,'formal'),'train',limit=params.limit('train'))
    require(len(train)==original['queries'],'Frozen graph/train length mismatch')
    expected_road=graph_identity(params.root,params.target,params.dataset,
                                  train.root/'adjacency.npy',params.road_k)
    actual_road=json.loads((params.run/'graph/graph_identity.json').read_text())
    require(actual_road==expected_road,'Frozen road graph source/identity mismatch')
    with np.load(params.run/'graph/graph_context.npz',allow_pickle=False) as z:
        road={key:z[key].copy() for key in z.files}
    return {'functional':graph_obj['neighbours'],
            'functional_valid':graph_obj['neighbours_valid'],'road':road}


def load_frozen_model(params,device):
    graph=load_frozen_graph(params)
    ck=torch.load(params.backbone_checkpoint,map_location='cpu',weights_only=False)
    net=load_network(params,device,graph,ck,'c1')
    net.eval()
    return net,ck


def verify_replay(params,split,verify_hashes=True,severity=None):
    name=('test_s'+str(severity) if severity is not None else split)
    folder=params.run/'paired_replay'/name
    config_path=folder/'identity.json'
    require(config_path.exists(),f'Missing replay identity: {config_path}')
    meta=json.loads(config_path.read_text())
    require(meta.get('target')==params.target and meta.get('dataset')==params.dataset,
            'Replay target/dataset mismatch')
    require(meta.get('k')==params.k and meta.get('chunk_queries')==params.replay_chunk,
            'Replay K/chunk mismatch')
    require(meta.get('checkpoint_sha256')==sha(params.backbone_checkpoint) and
            meta.get('s0_sha256')==sha(params.run/'s0.pt'), 'Replay Backbone/S0 mismatch')
    require(meta.get('graph_sha256')==sha(params.run/'train_only_neighbours.pt'),
            'Replay graph source SHA mismatch')
    require(meta.get('implementation_sha256')==sha(PROJ/'tcrr_stla/v7/replay.py'),
            'Replay implementation SHA mismatch')
    require(tuple(meta.get('periods',()))==params.retrieval_periods and
            meta.get('shortlist')==params.retrieval_shortlist and
            float(meta.get('history_weight'))==params.retrieval_history_weight and
            float(meta.get('forecast_weight'))==params.retrieval_forecast_weight,
            'Replay retrieval settings mismatch')
    from tcrr_stla.v7.storage import file_manifest
    formal=cache_path(params.config,'evaluation',severity) if severity is not None else cache_path(params.config,'formal')
    src=SupportSplit(formal,split,limit=params.limit('test' if severity is not None else split))
    require(meta.get('source')==file_manifest(src,split,sha(params.run/'s0.pt'),params.groups),
            'Replay source identity mismatch')
    train_src=SupportSplit(cache_path(params.config,'formal'),'train',
                           limit=params.limit('train'))
    require(meta.get('train_source')==file_manifest(train_src,'train',
            sha(params.run/'s0.pt'),params.groups), 'Replay TRAIN source mismatch')
    rows=len(src);require(meta.get('rows')==rows, 'Replay row mismatch')
    count=0
    for i in range(0,rows,params.replay_chunk):
        j=min(rows,i+params.replay_chunk)
        folder_chunk=folder/f'chunk_{i:06d}_{j:06d}'
        info=folder_chunk/'DONE.json'
        require(info.exists(),f'Missing replay chunk: {info}')
        obj=json.loads(info.read_text())
        require(obj.get('start')==i and obj.get('end')==j,'Replay chunk boundaries mismatch')
        for key in KEYS:
            f=folder_chunk/f'{key}.npy';require(f.exists(),f'Missing replay array {f}')
            if verify_hashes:
                require(sha(f)==obj.get('hashes',{}).get(key),f'Replay corrupted {f}')
        count+=1
    return src,folder,{'rows':rows,'chunks':count,'cache_identity_sha256':sha(config_path)}


def teacher_records(params,split,severity=None):
    folder=params.run/'teacher_cache'/('test_s'+str(severity) if severity is not None else split)
    from tcrr_stla.v7.storage import open_existing_teacher,file_manifest
    cache=cache_path(params.config,'evaluation',severity) if severity is not None else cache_path(params.config,'formal')
    src=SupportSplit(cache,split,limit=params.limit('test' if severity is not None else split))
    records=open_existing_teacher(split,folder,file_manifest(src,split,sha(params.run/'s0.pt'),params.groups),
                                  len(src),params.nodes,params.horizon,params.model_dim)
    require(records is not None, f'Incomplete teacher cache: {folder}')
    return records


def data_loader(params,src,records,folder,indices,batch):
    ds=ReplayDataset(src,records,folder,params.replay_chunk)
    return DataLoader(Subset(ds,indices) if indices is not None else ds,
                      batch_size=batch,shuffle=False,num_workers=0,pin_memory=False)


def candidate_prediction(out,net,weights):
    safe=out['candidate_residuals'].float()
    bound=float(out['residual_bound'])
    mix=(weights[:,:safe.shape[1]].float()*safe).sum(dim=1)
    proposal=bound*torch.tanh(mix/bound)
    return out['normal'].float()+torch.tanh(net.c1_activation).float()*\
        (out['base_prediction'].float()+proposal-out['normal'].float())


def candidate_weights_interventions(out):
    w=out['candidate_weights'].float()
    valid=out['candidate_available'].bool()
    k=valid.shape[1]
    none=w[:,-1:]
    mass=(1-none)
    n=valid.sum(1,keepdim=True).clamp_min(1)
    uniform=torch.cat((torch.where(valid,mass/n,torch.zeros_like(w[:,:k])),none),dim=1)
    maxindex=torch.where(valid,w[:,:k],-1.).argmax(dim=1,keepdim=True)
    top=torch.zeros_like(w[:,:k]).scatter_(1,maxindex,mass)
    top=torch.where(valid,top,torch.zeros_like(top))
    top1=torch.cat((top,none),dim=1)
    # Permute the learned weights AMONG VALID candidates, leaving None and
    # invalid positions exactly unchanged. Deterministic cyclic 1-step shift.
    order=torch.argsort((~valid).long(),dim=1,stable=True)
    sorted_weights=torch.gather(w[:,:k],1,order)
    counts=valid.sum(1,keepdim=True)
    rank=torch.arange(k,device=w.device).reshape(1,k,1,1)
    source_rank=(rank+1)%counts.clamp_min(1)
    rotated=torch.gather(sorted_weights,1,source_rank.expand_as(sorted_weights))
    rotated=torch.where(rank<counts,rotated,torch.zeros_like(rotated))
    perm=torch.zeros_like(w[:,:k]).scatter_(1,order,rotated)
    perm=torch.cat((perm,none),dim=1)
    only_none=torch.cat((torch.zeros_like(w[:,:k]),torch.ones_like(none)),dim=1)
    return {'uniform_keep_none':uniform,'permute_keep_none':perm,
            'top1_keep_none':top1,'none_only':only_none}


def flat_stats_tensor(pred,y,valid,direct,scale=1.,offset=0.):
    p=pred.float()*scale+offset
    truth=torch.where(valid,y.float(),torch.zeros_like(y.float()))
    err=torch.where(valid,(p-truth),torch.zeros_like(p))
    dm=direct.bool()
    if dm.ndim==2:dm=dm[:,None].expand_as(valid)
    elif dm.shape!=valid.shape:raise ValueError('Direct mask shape mismatch')
    dm=dm & valid
    return err,dm


class Summary:
    def __init__(self,methods,p):
        self.methods=list(methods);self.p=p
        self.total={m:np.zeros(7,dtype=np.float64) for m in methods}
        self.horizon={m:np.zeros((p,2),dtype=np.float64) for m in methods}
        self.groups=defaultdict(lambda:{m:np.zeros(3,dtype=np.float64) for m in methods})
        self.coverage_groups=defaultdict(lambda:{m:np.zeros(3,dtype=np.float64) for m in methods})
        self.rows=[]
    def add(self,method,err,valid,direct,indices,starts,families,coverages):
        ae=err.abs();v=valid
        total=self.total[method]
        total[0]+=float(ae.sum());total[1]+=float(err.square().sum());total[2]+=int(v.sum())
        total[3]+=float(torch.where(direct,ae,0.).sum());total[4]+=int(direct.sum())
        total[5]+=float(ae[:,-1].sum());total[6]+=int(v[:,-1].sum())
        hmae=torch.where(v,ae,torch.zeros_like(ae)).sum((0,2)).cpu().numpy()
        counts=v.sum((0,2)).cpu().numpy()
        self.horizon[method][:,0]+=hmae;self.horizon[method][:,1]+=counts
        per_sum=ae.sum((1,2)).cpu().numpy();per_ct=v.sum((1,2)).cpu().numpy()
        for local,idx in enumerate(indices):
            family=families[int(idx)];coverage=float(coverages[local]);grp=family or 'unknown'
            row_sse=float((err[local]**2).sum().cpu())
            self.groups[grp][method]+=np.array([per_sum[local],row_sse,per_ct[local]])
            bucket=min(int(coverage*5),4)
            cov_name=f'coverage_{bucket*20:02d}_{(bucket+1)*20:02d}pct'
            self.coverage_groups[cov_name][method]+=np.array([per_sum[local],row_sse,per_ct[local]])
            self.rows.append({'index':int(idx),'window_start':int(starts[local]),
                  'family':grp,'coverage':coverage,'method':method,
                  'absolute_error_sum':float(per_sum[local]),'valid_count':int(per_ct[local]),
                  'query_mae':float(per_sum[local]/per_ct[local]) if per_ct[local] else None})
    def finalize(self):
        out={}
        for key,v in self.total.items():
            out[key]={'mae':float(v[0]/max(1,v[2])),'rmse':float(math.sqrt(v[1]/max(1,v[2]))),
                      'masked_mae':float(v[3]/max(1,v[4])),
                      'h12_mae':float(v[5]/max(1,v[6])),
                      'count':int(v[2]),'direct_count':int(v[4]),
                      'horizon_mae':[(float(s/c) if c else None) for s,c in self.horizon[key]]}
        group={k:{m:{'mae':float(x[0]/max(1,x[2])), 'rmse':float(math.sqrt(x[1]/max(1,x[2]))),
                      'count':int(x[2])} for m,x in g.items() if x[2]>0} for k,g in self.groups.items()}
        coverage={k:{m:{'mae':float(x[0]/max(1,x[2])),
                     'count':int(x[2])} for m,x in v.items() if x[2]>0}
                  for k,v in self.coverage_groups.items()}
        return out,group,coverage


def get_families(ds):
    for key in ('fault_family','failure_family','mask_family','fault_type','mask_type',
                'family','failure_type','outage_type'):
        if key in ds.meta.columns:
            return ds.meta[key].fillna('unknown').astype(str).tolist(),key
    return ['unknown']*len(ds),'unavailable'


def auc_from_labels(values,labels):
    values=np.asarray(values,dtype=float);labels=np.asarray(labels,dtype=bool)
    pos=int(labels.sum());neg=int(len(labels)-pos)
    if pos==0 or neg==0:return None
    order=np.argsort(values,kind='stable');z=values[order]
    ranks=np.arange(1,len(z)+1,dtype=float)
    k=0
    while k<len(z):
        j=k+1
        while j<len(z) and z[j]==z[k]:j+=1
        ranks[k:j]=(k+1+j)/2.
        k=j
    full=np.empty_like(ranks);full[order]=ranks
    return float((full[labels].sum()-pos*(pos+1)/2)/(pos*neg))


def sampled_ranking_score(truth,pred,mask):
    t=np.asarray(truth);p=np.asarray(pred);v=np.asarray(mask,dtype=bool)
    rank_corr=[];pairs_hit=0;pairs_total=0;all_pred=[];all_harm=[]
    for x,y,ok in zip(t,p,v):
        x=x[ok];y=y[ok]
        if len(x)<2:continue
        all_pred.extend(y.tolist());all_harm.extend((x<0).tolist())
        # Spearman within valid candidate set; skip effectively constant rows
        if np.std(x)>1e-9 and np.std(y)>1e-9:
            rx=np.argsort(np.argsort(x,kind='stable'),kind='stable').astype(float)
            ry=np.argsort(np.argsort(y,kind='stable'),kind='stable').astype(float)
            rank_corr.append(float(np.corrcoef(rx,ry)[0,1]))
        d1=x[:,None]-x[None,:];d2=y[:,None]-y[None,:]
        informative=np.triu(np.abs(d1)>1e-4,1)
        pairs_total+=int(informative.sum())
        pairs_hit+=int(((d1*d2)>0)[informative].sum())
    return {'sampled_point_count':len(t),
            'mean_within_point_spearman':float(np.mean(rank_corr)) if rank_corr else None,
            'pairwise_accuracy':pairs_hit/pairs_total if pairs_total else None,
            'pairwise_pairs':pairs_total,
            'harm_auc_using_negative_predicted_utility':auc_from_labels(-np.array(all_pred),all_harm)}


def bootstrap_pair(rows,first,second,block_steps,iterations,seed):
    """Paired block bootstrap: positive means first method has HIGHER MAE (second better)."""
    table=defaultdict(dict)
    for row in rows:
        if row['method'] in (first,second):
            table[row['index']][row['method']]=row
    paired=[v for v in table.values() if first in v and second in v]
    if not paired:return None
    starts=np.array([r[first]['window_start'] for r in paired],dtype=np.int64)
    g=((starts-starts.min())//int(block_steps)).astype(np.int64)
    keys=np.unique(g)
    if len(keys)<2:return {'note':'fewer than 2 chronological blocks; confidence interval unavailable',
                           'blocks':len(keys),'queries':len(paired)}
    a=np.array([[r[first]['absolute_error_sum'],r[first]['valid_count'],
                 r[second]['absolute_error_sum'],r[second]['valid_count']] for r in paired],dtype=float)
    blocks=np.stack([a[g==key].sum(0) for key in keys]);rng=np.random.default_rng(seed)
    samples=rng.integers(0,len(keys),size=(iterations,len(keys)))
    totals=blocks[samples].sum(axis=1)
    diff=totals[:,0]/np.maximum(totals[:,1],1)-totals[:,2]/np.maximum(totals[:,3],1)
    exact=a.sum(0);mean=exact[0]/max(1,exact[1])-exact[2]/max(1,exact[3])
    return {'first':first,'second':second,'difference_mae_first_minus_second':float(mean),
            'bootstrap_95pct':[float(x) for x in np.quantile(diff,[.025,.975])],
            'blocks':int(len(keys)),'queries':len(paired),'block_steps':int(block_steps)}


def save_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    tmp=path.with_suffix('.partial.csv')
    with tmp.open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.replace(tmp,path)


def analyze_loader(params,net,dl,ds,device,bc,a,diagnostic,original_stla=None):
    mean=float(bc['mean']);std=float(bc['std'])
    if not std>0:raise ValueError('Backbone std must be positive')
    methods=METHODS if diagnostic else (TEST_METHODS if original_stla is not None else TEST_METHODS[1:])
    summ=Summary(methods,params.horizon)
    families,family_key=get_families(ds)
    rng=np.random.default_rng(20261104)
    weight_tot=np.zeros(9,dtype=np.float64)
    entropy_tot=np.zeros(3,dtype=np.float64)
    rank_truth=[];rank_pred=[];rank_valid=[]
    for bid,raw in enumerate(dl):
        b=as_device(raw,device)
        with torch.inference_mode():
            o=net.forward_progressive(b['cached_base'],b['cached_latent'],b['x'],b['mask'],
                    b['candidate_normal'],b['candidate_cf'],candidate_scores=b['candidate_scores'],
                    candidate_coverage=b['candidate_coverage'],candidate_valid=b['candidate_valid'],
                    candidate_support_raw=b['candidate_support_raw'],max_stage=1)
            y=b['y_raw'].float();v=b['future_valid'].bool()
            require(v.any().item(),f'No valid targets in batch {bid}')
            indices=b['index'].detach().cpu().tolist()
            starts=b['window_start'].detach().cpu().tolist()
            coverages=b['mask'].float().mean((1,2)).cpu().tolist()
            dm=b['direct_mask']
            predictions={'backbone':o['backbone'],'normal':o['normal']}
            if diagnostic:
                predictions['learned']=o['c1']
                for key,w in candidate_weights_interventions(o).items():
                    predictions[key]=candidate_prediction(o,net,w)
                act=torch.tanh(net.c1_activation).float()
                p0=o['base_prediction'].float();pn=o['normal'].float()
                predictions['normal_component_only']=pn+act*(p0+o['normal_response'].float()-pn)
                predictions['cf_component_only']=pn+act*(p0+o['cf_response'].float()-pn)
                swapped=net.forward_progressive(b['cached_base'],b['cached_latent'],b['x'],b['mask'],
                    b['candidate_normal'],b['candidate_cf'].roll(1,1),candidate_scores=b['candidate_scores'],
                    candidate_coverage=b['candidate_coverage'],candidate_valid=b['candidate_valid'],
                    candidate_support_raw=b['candidate_support_raw'],max_stage=1)
                predictions['shuffled_cf']=swapped['c1']
                occluded=net.forward_progressive(b['cached_base'],b['cached_latent'],b['x'],b['mask'],
                    b['candidate_normal'],b['candidate_cf'],candidate_scores=b['candidate_scores'],
                    candidate_coverage=b['candidate_coverage'],candidate_valid=b['candidate_valid'],
                    candidate_support_raw=torch.zeros_like(b['candidate_support_raw']),max_stage=1)
                predictions['support_occlusion']=occluded['c1']
                weights=o['candidate_weights'].float()
                mask_n=v
                entropy=-(weights.clamp_min(1e-8)*weights.clamp_min(1e-8).log()).sum(1)
                eff=weights[:,:-1].sum(1).square()/weights[:,:-1].square().sum(1).clamp_min(1e-8)
                weight_tot[0]+=float(torch.where(mask_n,entropy,0.).sum())
                weight_tot[1]+=float(torch.where(mask_n,weights[:,-1],0.).sum())
                weight_tot[2]+=float(torch.where(mask_n,eff,0.).sum())
                weight_tot[3]+=int(v.sum())
                u,nu,valid=full_activation_targets(o,b)
                target=ranking_targets(u,nu,valid,params.target_temperature)
                target_entropy=-(target.clamp_min(1e-8)*target.clamp_min(1e-8).log()).sum(1)
                entropy_tot[0]+=float(torch.where(v,target_entropy,0.).sum())
                entropy_tot[1]+=float(torch.where(v,(target-weights).abs().sum(1),0.).sum())
                entropy_tot[2]+=int(v.sum())
                predicted=(o['candidate_transfer_gain']-o['candidate_transfer_harm']).float()
                diff=torch.where(valid,(predicted-u).abs(),torch.zeros_like(predicted))
                weight_tot[4]+=float(diff.sum());weight_tot[5]+=int(valid.sum())
                safe=o['candidate_residuals'].float()
                cw=weights[:,:-1];mass=cw.sum(1).clamp_min(1e-8)
                cm=(cw*safe).sum(1)/mass
                set_variance=(cw*(safe-cm[:,None]).square()).sum(1)/mass
                set_agreement=((cw*torch.sign(safe)).sum(1).abs()/mass).clamp(0,1)
                weight_tot[6]+=float(torch.where(v,set_variance.sqrt(),0.).sum())
                weight_tot[7]+=float(torch.where(v,set_agreement,0.).sum())
                # Sample bounded number of [B,P,N] points for CPU ranking metrics
                good_idx=torch.nonzero(v.flatten(),as_tuple=False).flatten()
                if good_idx.numel():
                    sample=min(a.rank_samples_per_batch,good_idx.numel())
                    choice=good_idx[torch.from_numpy(rng.choice(len(good_idx),size=sample,replace=False)).to(device)]
                    truth=u.permute(0,2,3,1).reshape(-1,u.shape[1])[choice].cpu().numpy()
                    pred=predicted.permute(0,2,3,1).reshape(-1,u.shape[1])[choice].cpu().numpy()
                    valid_flat=valid.permute(0,2,3,1).reshape(-1,u.shape[1])[choice].cpu().numpy()
                    rank_truth.extend(truth.tolist());rank_pred.extend(pred.tolist());rank_valid.extend(valid_flat.tolist())
            else:
                predictions['c1']=o['c1']
                if original_stla is not None:
                    pred,_=original_stla(b['x'],b['mask'],b['tod'],b['dow'])
                    predictions['original_stlaformer']=pred
            for method in methods:
                pred=predictions[method]
                if not torch.isfinite(torch.where(v,pred,torch.zeros_like(pred))).all():
                    raise RuntimeError(f'Nonfinite {method} on valid target cells')
                err,masked=flat_stats_tensor(pred,y,v,dm,scale=std,offset=mean)
                if not torch.isfinite(err).all():raise RuntimeError('Nonfinite valid raw errors')
                summ.add(method,err,v,masked,indices,starts,families,coverages)
        if bid==0 or (bid+1)%50==0:
            print(f'C1_FROZEN_PROGRESS target={params.target} mode={"diagnostic" if diagnostic else "test"} '
                  f'batches={bid+1} queries={len(summ.rows)//len(methods)}',flush=True)
    metrics,groups,coverage_groups=summ.finalize()
    result={'metrics':metrics,'failure_family_key':family_key,'by_failure_family':groups,'by_coverage_group':coverage_groups,
            'count_queries':len(summ.rows)//len(methods),'source':'selection' if diagnostic else 'test',
            'reference':'backbone means frozen STLAformer + S0 (not original STLAformer)'}
    if diagnostic:
        result['candidate_weight_diagnostic']={
            'predicted_entropy':float(weight_tot[0]/max(1,weight_tot[3])),
            'mean_none_mass':float(weight_tot[1]/max(1,weight_tot[3])),
            'effective_candidates':float(weight_tot[2]/max(1,weight_tot[3])),
            'target_entropy':float(entropy_tot[0]/max(1,entropy_tot[2])),
            'target_predicted_l1':float(entropy_tot[1]/max(1,entropy_tot[2])),
            'utility_target_mae':float(weight_tot[4]/max(1,weight_tot[5])),
            'candidate_set_mean_disagreement':float(weight_tot[6]/max(1,weight_tot[3])),
            'candidate_set_direction_agreement':float(weight_tot[7]/max(1,weight_tot[3])),
            'note':'Ranking targets computed on Selection for diagnosis only; no training or model selection'}
        result['utility_ranking']=sampled_ranking_score(rank_truth,rank_pred,rank_valid)
        result['paired_bootstrap']={name:bootstrap_pair(summ.rows,left,right,a.block_steps,a.bootstrap,101)
                    for name,left,right in [
                        ('uniform_vs_learned','uniform_keep_none','learned'),
                        ('normal_vs_learned','normal','learned'),
                        ('shuffled_cf_vs_learned','shuffled_cf','learned'),
                        ('support_occlusion_vs_learned','support_occlusion','learned')]}
        q=defaultdict(dict)
        for row in summ.rows:q[row['index']][row['method']]=row['query_mae']
        delta=[]
        for per in q.values():
            if per.get('normal') is not None and per.get('learned') is not None:
                delta.append(per['learned']-per['normal'])
        regret=[]
        for per in q.values():
            if all(per.get(x) is not None for x in ('normal','learned','uniform_keep_none')):
                regret.append(per['learned']-min(per['normal'],per['learned'],per['uniform_keep_none']))
        result['harmful_correction']={'harmful_query_fraction_vs_normal':float(np.mean(np.array(delta)>0)) if delta else None,
              'beneficial_query_fraction_vs_normal':float(np.mean(np.array(delta)<0)) if delta else None,
              'mean_query_delta_mae':float(np.mean(delta)) if delta else None,
              'selection_regret_vs_normal_and_uniform_oracle_mean':float(np.mean(regret)) if regret else None,
              'oracle_is_diagnostic_not_deployable':True,
              'queries':len(delta)}
    else:
        result['paired_bootstrap']={'normal_vs_c1':bootstrap_pair(summ.rows,'normal','c1',a.block_steps,a.bootstrap,102),
                                  's0_vs_c1':bootstrap_pair(summ.rows,'backbone','c1',a.block_steps,a.bootstrap,102)}
        if original_stla is not None:
            result['paired_bootstrap']['original_stla_vs_c1']=bootstrap_pair(
                   summ.rows,'original_stlaformer','c1',a.block_steps,a.bootstrap,102)
    return result,summ.rows


def provenance(params,stamp,split,severity=None):
    folder=params.run/'teacher_cache'/('test_s'+str(severity) if severity is not None else split)
    obj=dict(stamp)
    obj['teacher_identity_sha256']=sha(folder/f'{split}_identity.json')
    replay=params.run/'paired_replay'/('test_s'+str(severity) if severity is not None else split)
    obj['replay_identity_sha256']=sha(replay/'identity.json')
    obj['source_split']=split
    if severity is not None:obj['severity']=severity
    return obj


def load_original(params,device):
    from tcrr_stla.stla import load_pretrained_stla, MaskAwareSTLAformer
    base,_=load_pretrained_stla(params.root,params.backbone_checkpoint,device)
    teacher=MaskAwareSTLAformer(base).to(device).eval()
    # Two exact zero-init tensors reproduce the original frozen backbone mapping.
    require(float(teacher.mask_proj[-1].weight.detach().abs().sum())==0 and
            float(teacher.mask_proj[-1].bias.detach().abs().sum())==0,
            'Original STLAformer zero-adapter identity contract broken')
    return teacher


def verify_selection(params):
    src,folder,meta=verify_replay(params,'val')
    rec=teacher_records(params,'val')
    indices,_=selection_indices(src)
    require(len(indices)>0,'No Selection queries')
    return src,rec,folder,indices,meta


def diagnostic(params,stamp,device,a):
    out=params.run/'diagnostic_c1';path=out/'summary.json'
    src,records,folder,indices,replay_info=verify_selection(params)
    expected=provenance(params,stamp,'val')
    expected['analysis_protocol']={'block_steps':a.block_steps,'bootstrap':a.bootstrap,
            'rank_samples_per_batch':a.rank_samples_per_batch,
            'eval_batch_size':a.eval_batch_size, 'methods':list(METHODS)}
    if path.exists() and not a.force_output:
        saved=json.loads(path.read_text())
        require(saved.get('provenance')==expected,'Old diagnosis identity mismatch; pass --force-output if intentional')
        print(f'C1_DIAGNOSTIC_REUSE {path}',flush=True);return saved
    net,bc=load_frozen_model(params,device)
    dl=data_loader(params,src,records,folder,indices,a.eval_batch_size)
    result,rows=analyze_loader(params,net,dl,src,device,bc,a,True)
    out.mkdir(parents=True,exist_ok=True)
    result['provenance']=expected
    result['selection_queries']=len(indices)
    result['replay_info']=replay_info
    result['interventions_are_frozen_not_retrained']=True
    save_csv(out/'query_methods.csv',rows)
    atomic_json(result,path)
    print(f'C1_DIAGNOSTIC_PASS {path} '+json.dumps({k:v['mae'] for k,v in result['metrics'].items()}),flush=True)
    del net;gc.collect()
    return result


def prepare_test_cache(params,device,severity,cache_only):
    name='test_s'+str(severity)
    teacher_dir=params.run/'teacher_cache'/name
    replay_dir=params.run/'paired_replay'/name
    if cache_only:
        # Verify identities and content in read-only mode, without invoking any writer.
        teacher_records(params,'test',severity)
        verify_replay(params,'test',severity=severity)
        return
    # These existing v7.1 writers use TRAIN-only candidate sources, SHA-checked
    # chunk identities and atomic per-chunk DONE markers. Interrupted prep resumes.
    train_cache(params,device,'test',severity=severity)
    prepare_replay(params,device,'test',severity=severity)
    require(teacher_dir.exists() and replay_dir.exists(), 'Test cache preparation failed')


def test_once(params,stamp,device,a,severity):
    out=params.run/'test_c1_only';target=out/f's{severity}.json'
    # Never overwrite or interpret old joint TEST files. Freeze C1 checkpoint for all severities.
    prepare_test_cache(params,device,severity,a.cache_only)
    src,folder,cache_info=verify_replay(params,'test',severity=severity)
    records=teacher_records(params,'test',severity)
    expected=provenance(params,stamp,'test',severity)
    expected['evaluation_protocol']={'original_stla':not a.no_original_stla,
         'block_steps':a.block_steps,'bootstrap':a.bootstrap,'methods':
         list(TEST_METHODS if not a.no_original_stla else TEST_METHODS[1:])}
    if target.exists() and not a.force_output:
        saved=json.loads(target.read_text())
        require(saved.get('provenance')==expected,'Old Test identity mismatch; do not mix results')
        print(f'C1_TEST_REUSE severity={severity} {target}',flush=True)
        return saved
    net,bc=load_frozen_model(params,device)
    original=load_original(params,device) if not a.no_original_stla else None
    dl=data_loader(params,src,records,folder,None,a.eval_batch_size)
    result,rows=analyze_loader(params,net,dl,src,device,bc,a,False,original_stla=original)
    result.update(severity=severity,provenance=expected,replay_info=cache_info,
                  no_test_based_model_or_hyperparameter_selection=True)
    out.mkdir(parents=True,exist_ok=True)
    save_csv(out/f's{severity}_query_methods.csv',rows)
    atomic_json(result,target)
    print(f'C1_TEST_PASS severity=s{severity} C1_MAE={result["metrics"]["c1"]["mae"]:.6f} '
          f'S0_MAE={result["metrics"]["backbone"]["mae"]:.6f}',flush=True)
    del net,original;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return result


def aggregate(params,stamp,severities):
    out=params.run/'test_c1_only'
    reports=[]
    for severity in severities:
        f=out/f's{severity}.json';require(f.is_file(),f'Missing Test report for s{severity}: {f}')
        item=json.loads(f.read_text())
        require(item.get('provenance',{}).get('checkpoint_sha256')==stamp['checkpoint_sha256'],
                'Cross-severity C1 checkpoint mismatch')
        require(item.get('provenance',{}).get('severity')==severity,'Wrong Test severity report')
        reports.append(item)
    modes=set.intersection(*(set(it['metrics']) for it in reports))
    keys=('mae','rmse','masked_mae','h12_mae')
    macro={m:{k:float(np.mean([it['metrics'][m][k] for it in reports])) for k in keys} for m in sorted(modes)}
    summary={'version':VERSION_ANALYSIS,'target':params.target,'dataset':params.dataset,
              'seed':params.seed,'checkpoint_sha256':stamp['checkpoint_sha256'],
              'severities':{f's{it["severity"]}':it['metrics'] for it in reports},
              'macro_equal_severity':macro,
              'notes':['Equal-severity macro average; not pooled-cell average.',
                       'Test never changes weights, hyperparameters or checkpoint.',
                       'backbone means S0. original_stlaformer is unadapted model.']}
    atomic_json(summary,out/'summary.json')
    lines=[f'# Frozen v7.1 C1 Test — {params.target}', '',
           f'Checkpoint SHA256: `{stamp["checkpoint_sha256"]}`','',
           '| Severity | Original STLA MAE | S0 MAE | Normal MAE | C1 MAE | C1 vs S0 | C1 vs Normal |',
           '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return f'{x:.6f}' if x is not None else '—'
    for r in reports:
        m=r['metrics'];c=m['c1']['mae'];s=m['backbone']['mae'];n=m['normal']['mae']
        o=m.get('original_stlaformer',{}).get('mae')
        lines.append(f'| s{r["severity"]} | {fmt(o)} | {fmt(s)} | {fmt(n)} | {fmt(c)} | '
                     f'{fmt(100*(s-c)/s)}% | {fmt(100*(n-c)/n)}% |')
    m=macro;c=m['c1']['mae'];s=m['backbone']['mae'];n=m['normal']['mae'];o=m.get('original_stlaformer',{}).get('mae')
    lines.append(f'| Macro | {fmt(o)} | {fmt(s)} | {fmt(n)} | {fmt(c)} | '
                 f'{fmt(100*(s-c)/s)}% | {fmt(100*(n-c)/n)}% |')
    lines.extend(['','## Other metrics','',
                 '| Severity | C1 RMSE | C1 Masked MAE | C1 H12 MAE |',
                 '|---|---:|---:|---:|'])
    for r in reports:
        v=r['metrics']['c1']
        lines.append(f'| s{r["severity"]} | {v["rmse"]:.6f} | {v["masked_mae"]:.6f} | {v["h12_mae"]:.6f} |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'C1_TEST_SUMMARY_PASS {out/"summary.json"}',flush=True)
    return summary


def main():
    a=parser()
    os.environ['TRAFFIDENT_TARGET']=a.target
    torch.set_num_threads(max(1,a.threads))
    if a.eval_batch_size<1 or a.block_steps<24 or a.bootstrap<20:
        raise ValueError('eval-batch-size>=1, block-steps>=24, bootstrap>=20 required')
    severities=tuple(int(x.strip()) for x in a.severities.replace(':',',').split(',') if x.strip())
    require(bool(severities) and len(set(severities))==len(severities) and
            all(x in SCALES for x in severities),'Severities must be a subset of 20,40,60,90')
    params,stamp=manifest_params(a)
    device=torch.device(a.device)
    if device.type=='cuda':require(torch.cuda.is_available(),'CUDA requested but unavailable')
    seed_everything(10)
    print('C1_FROZEN_IDENTITY_PASS '+json.dumps(stamp),flush=True)
    if a.mode=='verify':
        src,records,folder,ids,metadata=verify_selection(params)
        net,bc=load_frozen_model(params,device)
        # Validate all requested held-out source schemas before expensive PBS queueing.
        for s in severities:
            source=cache_path(params.config,'evaluation',s)
            test_src=SupportSplit(source,'test',limit=params.limit('test'))
            require(len(test_src)>0 and test_src.x_obs.shape[1:]==(params.history,params.nodes) and
                    test_src.y_norm.shape[1:]==(params.horizon,params.nodes),
                    f'Invalid TEST source shapes at severity s{s}')
            print(f'C1_TEST_SOURCE_PASS severity=s{s} rows={len(test_src)}',flush=True)
        print(f'C1_VERIFY_PASS target={params.target} selection_queries={len(ids)} '
              f'val_chunks={metadata["chunks"]} checkpoint={stamp["checkpoint_sha256"]}',flush=True)
        return
    if a.mode in ('diagnostic','all'):diagnostic(params,stamp,device,a)
    if a.mode in ('test','all'):
        for s in severities:
            print(f'C1_TEST_SEVERITY_START target={params.target} severity=s{s} device={device}',flush=True)
            test_once(params,stamp,device,a,s)
        aggregate(params,stamp,severities)
    elif a.mode=='aggregate':aggregate(params,stamp,severities)
    print(f'C1_FROZEN_COMPLETE target={params.target} mode={a.mode}',flush=True)

if __name__=='__main__':
    main()
