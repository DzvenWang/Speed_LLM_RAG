"""FARE P0.1 diagnostics; NEVER used by training or deployed inference.

No future ground truth enters any model input, memory candidate, or retrieval
operation. Privileged oracle labels are computed *only after* the forward pass.
Selected/random interventions are fixed-checkpoint tests, not causal retraining.
"""
from __future__ import annotations
import csv
import re
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from .data import to_device
from .config import project_root, json_write

FAMILIES=('entry_random','sensor_outage','temporal_outage','regional_outage','directional_outage')
INP=('x_obs','obs_mask','tod','dow','window_start')
ALPHAS=(0.,.05,.1,.12,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6)


def csv_save(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    cols=list(dict.fromkeys(c for row in rows for c in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)


def small_summary(a):
    a=np.asarray(a,dtype=np.float64)
    if not a.size:return {'count':0}
    return {'count':int(a.size),'mean':float(a.mean()),'std':float(a.std()),
            'p10':float(np.percentile(a,10)),'p50':float(np.percentile(a,50)),
            'p90':float(np.percentile(a,90))}


def metric_np(pred, target, valid):
    a=np.asarray(pred); y=np.asarray(target);m=np.asarray(valid,dtype=bool)
    if a.shape!=y.shape or m.shape!=y.shape:raise ValueError('Inconsistent prediction/target/mask')
    e=(a-y)[m]
    if not len(e):return {'MAE':None,'RMSE':None,'count':0}
    if not np.isfinite(e).all():raise FloatingPointError('Nonfinite prediction')
    return {'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.square(e).mean())), 'count':int(len(e))}


def relative_difference(a,b):
    if a is None or b is None:return None
    if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor):return None
    if a.shape != b.shape:raise ValueError('Different evidence tensor shapes')
    q=(a.detach().float()-b.detach().float()).abs()
    return {'delta_abs':float(q.mean()), 'delta_rel':float(q.mean()/a.detach().float().abs().mean().clamp_min(1e-8)),
            'reference_abs':float(a.detach().float().abs().mean())}


def input_only(batch):
    return {k:batch[k] for k in INP}


def _valid_selected(ids,ctx,regions,memory,window_start,overlap):
    """Genuine legal and jointly observable candidate test using only historical masks."""
    b,g,k=ids.shape;h=ctx['anchor_mask'].shape[2];a=regions.a
    st=memory.starts[ids]
    legal=(window_start[:,None,None]-st).abs()>=overlap
    anchor_ids=regions.anchor_ids.reshape(1,g,1,1,a).expand(b,g,k,h,a)
    mv=memory.valid[ids]  # 8 candidates only, not full coarse pool
    anchor_valid=mv.gather(-1,anchor_ids)
    joint=(anchor_valid & ctx['anchor_mask'][:,:,None]).any((-1,-2))
    return legal&joint


def _sample_from_pool(pool,legal,seed,k):
    """Vectorized deterministic sampling without replacement, no target labels."""
    b,g,m=pool.shape
    generator=torch.Generator(device='cpu');generator.manual_seed(int(seed))
    scores=torch.rand((b,g,m),generator=generator).to(pool.device)
    scores=scores.masked_fill(~legal,-1.)
    ranks=scores.topk(min(k,m),-1).indices
    ids=pool.gather(-1,ranks)
    if m<k:
        ids=torch.cat([ids,ids[:,:,:1].expand(b,g,k-m)],-1)
    return ids


def make_intervention(selected,kind,ctx,regions,memory,starts,seed):
    """Keep x/valid/latent/tod/dow aligned by changing the single candidate ID.

    random_top64: sample from actual coarse retrieval pool.
    random_global: sample from full train-only memory with overlap exclusion.
    time_matched: sample from legal memory in same TOD/DOW slot, falling back to
      legal global memory only when no matched windows exist (fallback tracked).
    """
    if kind=='selected':return selected,{'fallback_groups':0}
    b,g,k=selected['ids'].shape
    if kind=='uniform_selected':
        ids=selected['ids']
        fallback=0
    elif kind=='random_top64':
        pool=selected['coarse_ids']
        legal=(starts[:,None,None]-memory.starts[pool]).abs()>=24
        legal=legal & (selected['coarse_scores']>-1e3)
        ids=_sample_from_pool(pool,legal,seed,k)
        fallback=0
    else:
        m=len(memory.x)
        pool=torch.arange(m,device=starts.device).view(1,1,m).expand(b,g,m)
        legal=(starts[:,None,None]-memory.starts[None,None,:]).abs()>=24
        legal=legal.expand(b,g,m)
        fallback=0
        if kind=='time_matched':
            if memory.tod is None or memory.dow is None:
                raise RuntimeError('Time-matched random requires Memory TOD/DOW')
            # Last observed historical time slot is a time conditioning control.
            query_tod=ctx['_query_tod'][:,-1].long()
            query_dow=ctx['_query_dow'][:,-1].long()
            same=(memory.tod[:,-1][None,:]==query_tod[:,None]) & (memory.dow[:,-1][None,:]==query_dow[:,None])
            restricted=legal&same[:,None,:]
            fallback=int((restricted.sum(-1)<k).sum().item())
            legal=torch.where((restricted.sum(-1)>=k)[...,None],restricted,legal)
        elif kind!='random_global':raise ValueError(kind)
        ids=_sample_from_pool(pool,legal,seed,k)
    valid=(selected['valid'] if kind=='uniform_selected' else _valid_selected(ids,ctx,regions,memory,starts,24))
    # Candidate indices are independent from its stored state and latent: the
    # real Memory lookup for *every branch* always uses the same sampled IDs.
    weights=valid.float()/valid.sum(-1,keepdim=True).clamp_min(1)
    support=valid.any(-1)
    entropy=-(weights*weights.clamp_min(1e-9).log()).sum(-1)
    confidence=torch.stack([ctx['coverage'],ctx['missing'],support.float(),entropy],-1)
    data={k:v for k,v in selected.items() if k not in ('coarse_ids','coarse_scores','match_error','trend_error')}
    data.update({'ids':ids,'valid':valid,'weights':weights,'group_valid':support,
                 'confidence':confidence,'scores':torch.zeros_like(weights),
                 'coverage':torch.zeros_like(weights)})
    if (((starts[:,None,None]-memory.starts[ids]).abs()<24) & valid).any():
        raise ValueError('Candidate temporal overlap')
    return data,{'fallback_groups':fallback}


class CanonicalValRemask(Dataset):
    """Rebuild validation input masks using Base_SupportBench's official sensing.

    This is a *diagnostic re-mask view*, NOT the benchmark's official held-out
    paired test protocol. Exact reference s40 mask equivalence must be audited
    before use; if it fails the tool stops instead of silently claiming parity.
    """
    def __init__(self,base,severity,dataset,target,verify_rows=32):
        self.base=base;self.severity=int(severity);self.dataset=dataset
        self.num_nodes=base.num_nodes;self.history=base.history
        self.family_column=base.family_column
        from .config import insert_paths
        insert_paths()
        from supportbench.sensing import make_view_spec,materialize_mask
        import yaml
        root=project_root()
        cfg_path=root/'Base_SupportBench'/'configs'/'base_supportbench.yaml'
        if not cfg_path.exists():raise FileNotFoundError(f'Need canonical sensing config: {cfg_path}')
        self.sensing_cfg=yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
        self.sg_dir=root/'data'/'processed'/'traffident'/'base_supportbench'/target/dataset
        if not self.sg_dir.is_dir():raise FileNotFoundError(f'Canonical sensing layouts not found: {self.sg_dir}')
        self.make=make_view_spec;self.materialize=materialize_mask
        self.meta=base.meta.iloc[:len(base)].copy().reset_index(drop=True)
        if 'family' not in self.meta:raise RuntimeError('Validation manifest missing family')
        self.audit={'checked':0,'different_mask_entries':0,'total_mask_entries':0}
        if int(verify_rows)>0:
            n=min(int(verify_rows),len(self))
            for i in np.linspace(0,len(self)-1,n,dtype=int):
                original=self.base[int(i)]; rebuilt=self._mask(int(i),original,40)
                diff=(rebuilt!=original['obs_mask'].numpy())
                rebuilt_x=torch.where(torch.as_tensor(rebuilt),original['x_full'],torch.zeros_like(original['x_full']))
                if not torch.allclose(rebuilt_x,original['x_obs'],atol=1e-5,rtol=1e-5):
                    raise RuntimeError(f'FARE_P01_S40_INPUT_PARITY_ERROR index={i}')
                art=original['raw_valid']&~torch.as_tensor(rebuilt)
                if not torch.equal(art.any(0),original['direct_mask'].bool()):
                    raise RuntimeError(f'FARE_P01_S40_DIRECT_MASK_MISMATCH index={i}')
                self.audit['checked']+=1
                self.audit['different_mask_entries']+=int(diff.sum())
                self.audit['total_mask_entries']+=int(diff.size)
            if self.audit['different_mask_entries']:
                raise RuntimeError('FARE_P01_S40_MASK_PROTOCOL_MISMATCH '+json.dumps(self.audit)+
                                   ' -- regeneration and original validation masks disagree; do not report four-severity diagnostics')

    def __len__(self):return len(self.base)
    def _mask(self,i,item,severity):
        row=self.meta.iloc[i]; family=str(row['family'])
        if family not in FAMILIES:raise ValueError(f'Unsupported family {family}')
        view_id=str(row.get('view_id',''))
        mm=re.search(r'__i(\d+)__',view_id)
        instance=int(row['instance']) if 'instance' in row and str(row['instance'])!='nan' else (int(mm.group(1)) if mm else 0)
        distribution=str(row.get('distribution','id'))
        spec=self.make(self.dataset,
                       'val',int(row['window_start']),family,float(severity)/100.,
                       instance,self.sensing_cfg,distribution=distribution)
        # Explicit externally recorded s40 seed takes priority if provided.
        if severity==40 and 'seed' in row and str(row['seed'])!='nan':spec['seed']=int(row['seed'])
        raw=item['raw_valid'].numpy().astype(bool)
        missing=self.materialize(spec,self.history,self.num_nodes,self.sg_dir,raw_valid_speed=raw)
        return torch.as_tensor(raw&~missing,dtype=torch.bool).numpy()

    def __getitem__(self,i):
        item=self.base[i]
        mask=self._mask(i,item,self.severity)
        out=dict(item)
        out['obs_mask']=torch.from_numpy(mask)
        out['artificial_mask']=item['raw_valid']&~out['obs_mask']
        out['direct_mask']=out['artificial_mask'].any(0)
        out['x_obs']=torch.where(out['obs_mask'],item['x_full'],torch.zeros_like(item['x_full']))
        return out


def validate_remask_identity(ds,base):
    count=min(4,len(ds));rows=[]
    for i in range(count):
        a=ds[i];b=base[i]
        rows.append({'index':i,'mask_equal':bool(torch.equal(a['obs_mask'],b['obs_mask'])),
                     'values_equal':bool(torch.allclose(a['x_obs'],b['x_obs'],atol=1e-5))})
    return rows


def oracle_alpha_metrics(base,candidate,y,valid,max_alpha=.6,support=None):
    """PRIVILEGED point-wise L1 oracle, used ONLY to assess a ceiling."""
    d=candidate-base;e=base-y
    alpha=np.zeros_like(base,dtype=np.float32)
    good=(np.abs(d)>1e-9)
    alpha[good]=np.clip(-e[good]/d[good],0.,max_alpha)
    if support is not None:
        alpha=alpha*np.asarray(support,dtype=np.float32)
    oracle=base+alpha*d
    return metric_np(oracle,y,valid),float(alpha[valid].mean()) if valid.any() else None


def corrected_with_alpha(base,candidate,has_support,alpha):
    return base+float(alpha)*has_support*(candidate-base)


def spearman_pairs(xs,ys):
    x=np.asarray(xs,dtype=np.float64);y=np.asarray(ys,dtype=np.float64)
    good=np.isfinite(x)&np.isfinite(y)
    x=x[good];y=y[good]
    if len(x)<3 or np.std(x)<1e-10 or np.std(y)<1e-10:return None
    def rank(v):
        order=np.argsort(v,kind='mergesort');r=np.empty(len(v),dtype=float);r[order]=np.arange(len(v))
        _,inv,cnt=np.unique(v,return_inverse=True,return_counts=True)
        for group in np.where(cnt>1)[0]:r[inv==group]=r[inv==group].mean()
        return r
    return float(np.corrcoef(rank(x),rank(y))[0,1])


def _input_relevance_metrics(out,y,valid):
    pred=out['prediction'].float().cpu().numpy();base=out['base_prediction'].float().cpu().numpy();cand=out['candidate_prediction'].float().cpu().numpy()
    return pred,base,cand
