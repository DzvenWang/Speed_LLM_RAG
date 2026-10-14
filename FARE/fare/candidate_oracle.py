"""FARE v2.1.3: deterministic, causally-legal, actual-forward audit.

Targets the REAL v2.1.1 topology: Region Top64 -> Node Top16 -> Utility Top8.
Does not modify model weights or use current future labels in retrieval.

Each candidate intervention forces a single historical window for ALL nodes owned
by one region. Other regions keep their original Real retrieval. The target-region
NULL intervention is the matched control. This is a fixed-model diagnostic, NOT
an inference algorithm or a jointly optimal global oracle.
"""
from __future__ import annotations
from typing import Any
import math
import numpy as np
import torch

SAFE_INFERENCE_KEYS=('x_obs','obs_mask','tod','dow','window_start')


def inference_only(batch:dict)->dict:
    missing=[k for k in SAFE_INFERENCE_KEYS if k not in batch]
    if missing:raise KeyError(f'Missing inference inputs: {missing}')
    return {k:batch[k] for k in SAFE_INFERENCE_KEYS}


def validate_candidate_pool(model,memory,inputs,pool):
    required=('coarse_ids','ids','weights','selected_evidence','node_valid',
              'null_weight','utility','risk','tokens','forecast_delta',
              'valid','coarse_scores','node_query')
    absent=[k for k in required if k not in pool]
    if absent:raise RuntimeError('FARE212_INTERFACE_INCOMPATIBLE missing pool fields: '+str(absent))
    b=inputs['x_obs'].shape[0]
    if b!=1:raise ValueError('Oracle candidate audit requires batch_size=1')
    if pool['coarse_ids'].shape[:2]!=(1,model.regions.g):
        raise ValueError('Region Top64 pool dimensions do not match v2.1.1')
    if pool['ids'].shape[:2]!=(1,model.n):
        raise ValueError('Node Top8 pool dimensions do not match v2.1.1')
    if memory.future is None:raise RuntimeError('Historical TRAIN future missing')
    legal=model.retrieval._legal(memory,inputs['window_start'])
    ids=pool['coarse_ids'][0]
    if (ids<0).any() or (ids>=len(memory.x)).any():
        raise RuntimeError('Coarse candidate ID out of memory range')
    valid=legal[0,ids]
    # Illegal candidates can occur as padded ranks only when insufficient past
    # memory exists; forbid using them, rather than throwing for padded slots.
    return valid & (pool['coarse_scores'][0]>-1e3)


def _clone_pool_for_intervention(pool:dict):
    """Only clone tensors written by oracle; never mutate cached retrieval state."""
    out=pool.copy()
    mutable=('ids','weights','valid','node_valid','null_weight','tokens',
             'forecast_delta','utility','risk')
    for key in mutable:out[key]=pool[key].clone()
    out['selected_evidence']={k:v.clone() for k,v in pool['selected_evidence'].items()}
    return out


def _reaggregate(pool):
    weights=pool['weights']
    chosen=pool['selected_evidence']
    pool['tokens']=chosen['token']*weights[...,None]
    pool['forecast_delta']=(chosen['forecast_delta']*weights[...,None]).sum(2)
    pool['node_valid']=pool['valid'].any(-1)
    return pool


def _target_nodes(model,g:int):
    if not 0<=g<model.regions.g:raise ValueError(f'Invalid region {g}')
    return torch.where(model.regions.owner==g)[0]


def null_intervention(model,pool,g:int)->dict:
    out=_clone_pool_for_intervention(pool)
    nodes=_target_nodes(model,g)
    out['weights'][:,nodes,:]=0
    out['valid'][:,nodes,:]=False
    out['null_weight'][:,nodes]=1
    return _reaggregate(out)


def selected_top1_intervention(model,pool,g:int)->dict:
    """Force original per-node rank-1 evidence, not a whole-region shared ID."""
    out=_clone_pool_for_intervention(pool)
    nodes=_target_nodes(model,g)
    good=pool['valid'][:,nodes,0].bool()
    out['weights'][:,nodes,:]=0
    out['weights'][:,nodes,0]=good.float()
    out['valid'][:,nodes,:]=False
    out['valid'][:,nodes,0]=good
    out['null_weight'][:,nodes]=(~good).float()
    return _reaggregate(out)


def singleton_intervention(model,pool,g:int,candidate:dict)->dict:
    """Reweight singleton candidate via *real* v2.1.1 Adapter + Utility head."""
    out=_clone_pool_for_intervention(pool)
    nodes=_target_nodes(model,g)
    enc=candidate['evidence']
    valid=candidate['effective'].squeeze(-1).bool()  # [1,S]
    # Use candidate's sparse gathered evidence for the actual region nodes.
    # Original Top8 slots are left intact elsewhere and ignored in this region.
    out['ids'][:,nodes,:]=candidate['coarse_id']
    out['weights'][:,nodes,:]=0
    out['weights'][:,nodes,0]=valid.float()
    out['valid'][:,nodes,:]=False
    out['valid'][:,nodes,0]=valid
    out['null_weight'][:,nodes]=(~valid).float()
    out['utility'][:,nodes,:]=0
    out['risk'][:,nodes,:]=0
    out['utility'][:,nodes,0]=candidate['utility'].squeeze(-1)
    out['risk'][:,nodes,0]=candidate['risk'].squeeze(-1)
    for name,source in enc.items():
        if name not in out['selected_evidence']:
            raise RuntimeError(f'Selected evidence missing candidate field {name}')
        destination=out['selected_evidence'][name]
        if destination[:,nodes,0].shape != source.squeeze(2).shape:
            raise RuntimeError(f'Candidate evidence shape mismatch: {name}')
        destination[:,nodes,0]=source.squeeze(2)
    return _reaggregate(out)


def _extract_prediction(out)->torch.Tensor:
    if not isinstance(out,dict) or 'prediction' not in out:
        raise RuntimeError('Audit forward must return FARE prediction dictionary')
    pred=out['prediction']
    if pred.ndim!=3 or not bool(torch.isfinite(pred).all().item()):
        raise RuntimeError('Invalid audit prediction tensor')
    return pred


def target_mae(pred,truth,valid,node_mask,scale,missing=None)->float:
    mask=valid.bool() & node_mask[None,None,:]
    if missing is not None:
        if missing.ndim==2:
            missing=missing[:,None,:]
        if missing.ndim!=3:raise ValueError('direct mask must have [B,N] or [B,P,N]')
        mask=mask & missing.bool()
    if not bool(mask.any().item()):return float('nan')
    return float((pred-truth).abs()[mask].mean().item()*scale)


def _nanavg(values):
    values=np.asarray(values,dtype=np.float64)
    good=np.isfinite(values)
    return float(values[good].mean()) if good.any() else float('nan')


def _spearman(x,y)->float:
    """Small candidate-set rank correlation without an external SciPy dependency."""
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    good=np.isfinite(x)&np.isfinite(y)
    x=x[good];y=y[good]
    if len(x)<3 or np.ptp(x)<1e-10 or np.ptp(y)<1e-10:return float('nan')
    def ranks(v):
        order=np.argsort(v,kind='mergesort');out=np.empty(len(v),dtype=np.float64)
        sorted_values=v[order];i=0
        while i<len(v):
            j=i+1
            while j<len(v) and sorted_values[j]==sorted_values[i]:j+=1
            out[order[i:j]]=(i+j-1)/2.
            i=j
        return out
    return float(np.corrcoef(ranks(x),ranks(y))[0,1])


@torch.no_grad()
def evaluate_query_regions(model,memory,batch:dict,*,std:float,
                           max_candidates:int=64,region_ids:list[int]|None=None,
                           strict_pool_match:bool=True,
                           audit_modes:tuple[str,...]=("singleton","loo","replacement","scaling","node_loo"),
                           max_nodes:int=4,replacement_slot:int=0,
                           scaling_factors:tuple[float,...]=(0.,.25,.5,.75,1.,1.25))->list[dict]:
    """Measure full FARE forecast MAE under single-region evidence interventions.

    Future labels are *only* used by target_mae after all model forwards.
    Every candidate is first checked against model's causal retrieval eligibility.
    """
    model.eval()
    allowed={"singleton","loo","replacement","scaling","node_loo"}
    if set(audit_modes)-allowed:raise ValueError(f"Unknown audit modes: {set(audit_modes)-allowed}")
    if max_nodes<0:raise ValueError("max_nodes must be >=0")
    inputs=inference_only(batch)
    base,pool=model.retrieve_for_diagnostics(inputs,memory)
    legal=validate_candidate_pool(model,memory,inputs,pool)
    real=_extract_prediction(model.forward_from_retrieved(inputs,base,pool))
    if strict_pool_match:
        # Verifies that replayed pool produces exactly the original FARE output.
        fresh=_extract_prediction(model(inputs,memory,mode='full'))
        max_abs=float((fresh-real).abs().max().item())
        if max_abs>1e-4:
            raise RuntimeError(f'FARE212_POOL_REPLAY_MISMATCH max_abs={max_abs:.7f}')
    truth=batch['y_norm'].float()
    future_valid=batch['future_valid'].bool()
    direct=batch.get('direct_mask')
    if real.shape != truth.shape:
        raise ValueError('Target shape does not match model prediction')
    if region_ids is None:region_ids=list(range(model.regions.g))
    rows=[]
    for g in region_ids:
        nodes=_target_nodes(model,g)
        mask=model.regions.target_mask[g].bool()
        # Real is the baseline for all other regions. Null switches off ONLY
        # retrieval for region g, controlling for external-region benefits.
        null=_extract_prediction(model.forward_from_retrieved(
            inputs,base,null_intervention(model,pool,g)))
        first=_extract_prediction(model.forward_from_retrieved(
            inputs,base,selected_top1_intervention(model,pool,g)))
        null_mae=target_mae(null,truth,future_valid,mask,std)
        real_mae=target_mae(real,truth,future_valid,mask,std)
        first_mae=target_mae(first,truth,future_valid,mask,std)
        base_mae=target_mae(base['prediction'],truth,future_valid,mask,std)
        ranks=torch.where(legal[g])[0].tolist()
        # Dense top-m coarse candidates, no synthetic candidates or label-based pool.
        if max_candidates>0:ranks=ranks[:max_candidates]
        cand_rows=[]
        candidate_encodings={}
        for rank in (ranks if "singleton" in audit_modes else []):
            encoded=model.retrieval.encode_audit_candidate(
                inputs,base,memory,model.regions,pool,g,rank)
            if not bool(encoded['effective'].any()):continue
            candidate_encodings[rank]=encoded
            switched=_extract_prediction(model.forward_from_retrieved(
                inputs,base,singleton_intervention(model,pool,g,encoded)))
            cand_mae=target_mae(switched,truth,future_valid,mask,std)
            cand_missing=target_mae(switched,truth,future_valid,mask,std,direct)
            cand_rows.append(dict(coarse_rank=int(rank),memory_id=encoded['coarse_id'],
                legal=True,eligible_node_fraction=float(encoded['effective'].float().mean().item()),
                predicted_score=float(encoded['ranking'].mean().item()),
                predicted_utility=float(encoded['utility'].mean().item()),
                prediction_mae=cand_mae,gain_vs_region_null=null_mae-cand_mae,
                directly_masked_mae=cand_missing))
        gains=[x['gain_vs_region_null'] for x in cand_rows]
        best=max(cand_rows,key=lambda x:x['gain_vs_region_null']) if cand_rows else None
        predicted_best=max(cand_rows,key=lambda x:x['predicted_score']) if cand_rows else None
        rank_correlation=_spearman([x['predicted_score'] for x in cand_rows],
                                   [x['gain_vs_region_null'] for x in cand_rows])
        selected_ids=set(int(x) for x in pool['ids'][0,nodes].flatten().tolist())
        top8=[x for x in cand_rows if x['memory_id'] in selected_ids]
        best_selected=max(top8,key=lambda x:x['gain_vs_region_null']) if top8 else None
        direct_null=target_mae(null,truth,future_valid,mask,std,direct)
        direct_real=target_mae(real,truth,future_valid,mask,std,direct)
        result=dict(
            region_id=int(g),num_region_nodes=int(len(nodes)),
            num_legal_coarse_candidates=int(legal[g].sum().item()),
            num_evaluated_candidates=int(len(cand_rows)),
            null_mae=null_mae,base_mae=base_mae,real_mae=real_mae,
            selected_top1_mae=first_mae,
            real_gain_vs_null=null_mae-real_mae,
            selected_top1_gain_vs_null=null_mae-first_mae,
            oracle_best_singleton_gain=(best['gain_vs_region_null'] if best is not None else float('nan')),
            oracle_with_abstention_gain=(max(0.,best['gain_vs_region_null']) if best is not None else 0.),
            best_singleton_coarse_rank=(best['coarse_rank'] if best else None),
            best_singleton_memory_id=(best['memory_id'] if best else None),
            selected_union_best_singleton_gain=(best_selected['gain_vs_region_null'] if best_selected else float('nan')),
            ranker_predicted_top1_gain=(predicted_best['gain_vs_region_null'] if predicted_best else float('nan')),
            ranker_candidate_spearman=rank_correlation,
            singleton_oracle_minus_ranker=((best['gain_vs_region_null']-predicted_best['gain_vs_region_null']) if best and predicted_best else float('nan')),
            random_singleton_expected_gain=_nanavg(gains),
            positive_candidate_fraction=_nanavg([x>0 for x in gains]),
            harmful_candidate_fraction=_nanavg([x<0 for x in gains]),
            direct_null_mae=direct_null,direct_real_mae=direct_real,
            direct_real_gain_vs_null=direct_null-direct_real,
            real_selected_unique_memories=int(len(selected_ids)),
            candidates=cand_rows)
        if any(mode in audit_modes for mode in ("loo","replacement","scaling","node_loo")):
            from .weight_preserving_audit import audit_region_interventions
            cf=audit_region_interventions(
                model,memory,inputs,base,pool,region_id=g,real=real,null=null,
                truth=truth,future_valid=future_valid,std=std,direct=direct,
                legal_ranks=ranks,max_nodes=max_nodes,replacement_slot=replacement_slot,
                scaling_factors=scaling_factors,modes=audit_modes,
                cached_candidates=candidate_encodings)
            result.update(cf)
        rows.append(result)
    return rows
