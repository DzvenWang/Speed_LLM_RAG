"""FARE v2.1.3: faithful TopK-slot counterfactual probes.

No parameters, checkpoint changes, training labels or optimizers.
Selected evidence remains node-specific. Changes are confined to one target
region (or one target node) while the same retrieved pool is replayed.

LOO removes the selected evidence mass at slot k and sends it to NULL; it
never renormalizes other retrieved candidates.  Replacement retains the
original slot weight and transfers the mass to NULL if a replacement is
invalid for that particular node.  All outcomes use the trained integration
and controller, *not* a forecast-delta-only shortcut.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from .candidate_oracle import (_clone_pool_for_intervention, _reaggregate,
    _target_nodes, target_mae, _nanavg, _spearman, _extract_prediction)


def _selected_nodes(model, region_id: int, node_id: int | None = None):
    nodes = _target_nodes(model, region_id)
    if node_id is None:
        return nodes
    node = int(node_id)
    if not bool((nodes == node).any()):
        raise ValueError(f'Node {node} is not owned by region {region_id}')
    return nodes[nodes == node]


def leave_one_slot_out(model, pool: dict, region_id: int, slot: int,
                       node_id: int | None = None) -> dict:
    """Transfer slot's *original* per-node probability mass to NULL.

    Slot k denotes per-node rank k, not a shared historical-memory ID.
    """
    out = _clone_pool_for_intervention(pool)
    nodes = _selected_nodes(model, region_id, node_id)
    if not 0 <= slot < pool['weights'].shape[-1]:
        raise ValueError(f'Invalid selected evidence slot {slot}')
    removed = out['weights'][:, nodes, slot].clone()
    out['weights'][:, nodes, slot] = 0
    out['valid'][:, nodes, slot] = False
    out['null_weight'][:, nodes] += removed
    _check_mass(pool, out, nodes)
    return _reaggregate(out)


def replace_slot_keep_weight(model, pool: dict, region_id: int, slot: int,
                             candidate: dict, node_id: int | None = None) -> dict:
    """Replace history episode without changing existing slot weight.

    A per-node invalid candidate sends its slot mass to NULL. Only target
    nodes' selected_evidence, utility, risk and IDs are changed.
    """
    out = _clone_pool_for_intervention(pool)
    nodes = _selected_nodes(model, region_id, node_id)
    if not 0 <= slot < pool['weights'].shape[-1]:
        raise ValueError(f'Invalid selected evidence slot {slot}')
    all_nodes = _target_nodes(model, region_id)
    # Candidate encoding is dense only over this region, in owner order.
    local_idx = torch.searchsorted(all_nodes, nodes)
    if not bool(torch.equal(all_nodes[local_idx], nodes)):
        raise RuntimeError('Candidate node indexing mismatch')
    effective = candidate['effective'][:, local_idx, 0].bool()
    before = out['weights'][:, nodes, slot].clone()
    out['weights'][:, nodes, slot] = torch.where(effective, before, torch.zeros_like(before))
    out['null_weight'][:, nodes] += torch.where(effective, torch.zeros_like(before), before)
    out['valid'][:, nodes, slot] = effective & (before > 0)
    out['ids'][:, nodes, slot] = int(candidate['coarse_id'])
    out['utility'][:, nodes, slot] = candidate['utility'][:, local_idx, 0]
    out['risk'][:, nodes, slot] = candidate['risk'][:, local_idx, 0]
    for name, source in candidate['evidence'].items():
        if name not in out['selected_evidence']:
            raise RuntimeError(f'Selected evidence missing candidate field {name}')
        destination = out['selected_evidence'][name]
        part = source[:, local_idx, 0]
        if destination[:, nodes, slot].shape != part.shape:
            raise RuntimeError(f'Candidate field {name} shape mismatch')
        destination[:, nodes, slot] = part
    _check_mass(pool, out, nodes)
    return _reaggregate(out)


def _check_mass(original, modified, nodes):
    old = original['weights'][:, nodes].sum(-1) + original['null_weight'][:, nodes]
    new = modified['weights'][:, nodes].sum(-1) + modified['null_weight'][:, nodes]
    if not torch.allclose(old, new, atol=1e-5, rtol=0):
        raise RuntimeError('Evidence/Null probability mass was not conserved')
    if bool((modified['weights'][:, nodes] < -1e-7).any()) or bool((modified['null_weight'][:, nodes] < -1e-7).any()):
        raise RuntimeError('Negative evidence/Null probability detected')


def predict_from_pool(model, inputs, base, pool):
    return _extract_prediction(model.forward_from_retrieved(inputs, base, pool))


def horizon_gains(reference: torch.Tensor, variant: torch.Tensor,
                  truth: torch.Tensor, valid: torch.Tensor, mask: torch.Tensor,
                  std: float, direct: torch.Tensor | None = None):
    """Positive = variant has less error; fixed thirds of the prediction horizon."""
    h=truth.shape[1]
    results={}
    for i,(lo,hi) in enumerate(((0,h//3),(h//3,2*h//3),(2*h//3,h)),start=1):
        if hi <= lo:
            results[f'horizon_group_{i}_gain']=float('nan')
            continue
        v=valid[:,lo:hi]
        results[f'horizon_group_{i}_gain']=(
            target_mae(reference[:,lo:hi],truth[:,lo:hi],v,mask,std)
            -target_mae(variant[:,lo:hi],truth[:,lo:hi],v,mask,std))
    if direct is not None:
        results['direct_gain']=(target_mae(reference,truth,valid,mask,std,direct)
                               -target_mae(variant,truth,valid,mask,std,direct))
    return results


def _sample_region_nodes(nodes: torch.Tensor, max_nodes: int):
    if max_nodes < 0:
        raise ValueError('max_nodes must be nonnegative')
    if max_nodes == 0 or len(nodes) <= max_nodes:
        return [int(i) for i in nodes.tolist()]
    positions=np.linspace(0,len(nodes)-1,max_nodes,dtype=np.int64)
    return [int(nodes[i]) for i in positions]


@torch.no_grad()
def audit_region_interventions(model, memory, inputs, base, pool, *,
                               region_id: int, real: torch.Tensor, null: torch.Tensor,
                               truth: torch.Tensor, future_valid: torch.Tensor,
                               std: float, direct: torch.Tensor | None,
                               legal_ranks: list[int],
                               max_nodes: int=4, replacement_slot: int=0,
                               scaling_factors=(0., .25, .5, .75, 1., 1.25),
                               modes=('loo','replacement','scaling','node_loo'),
                               cached_candidates:dict[int,dict]|None=None) -> dict:
    modes=set(modes)
    mask=model.regions.target_mask[region_id].bool()
    nodes=_target_nodes(model,region_id)
    real_mae=target_mae(real,truth,future_valid,mask,std)
    null_mae=target_mae(null,truth,future_valid,mask,std)
    width=pool['weights'].shape[-1]
    result={'counterfactual_scope':'fixed_pool_region_node_specific_topk',
            'real_gain_vs_region_null':null_mae-real_mae}

    # Faithful rank-slot deletion: positive `keep_gain` means retaining this
    # slot is better than moving its exact mass to the Null candidate.
    loo=[]
    if 'loo' in modes:
        for slot in range(width):
            deleted=leave_one_slot_out(model,pool,region_id,slot)
            y=predict_from_pool(model,inputs,base,deleted)
            deleted_mae=target_mae(y,truth,future_valid,mask,std)
            row={'slot':slot,
                 'mean_slot_mass':float(pool['weights'][:,nodes,slot].mean().item()),
                 'removed_slot_mae':deleted_mae,
                 'keep_gain_vs_removing':deleted_mae-real_mae,
                 'removal_improvement_over_real':real_mae-deleted_mae}
            row.update(horizon_gains(y,real,truth,future_valid,mask,std,direct))
            loo.append(row)
    result['leave_one_out']=loo
    result['loo_mean_keep_gain']=_nanavg([x['keep_gain_vs_removing'] for x in loo])
    result['loo_harmful_slot_fraction']=_nanavg([x['keep_gain_vs_removing']<0 for x in loo])
    result['loo_best_removal_improvement']=max([0.]+[x['removal_improvement_over_real'] for x in loo])
    result['loo_all_slots_beneficial']=bool(loo and all(x['keep_gain_vs_removing']>=0 for x in loo))

    # Keep original Top-8 weights, Null mass, other 7 slots and other regions.
    replacements=[]
    if 'replacement' in modes:
        for rank in legal_ranks:
            candidate=(cached_candidates or {}).get(rank)
            if candidate is None:
                candidate=model.retrieval.encode_audit_candidate(
                    inputs,base,memory,model.regions,pool,region_id,rank)
            effective=candidate['effective'][:, :, 0]
            if not bool(effective.any()):
                continue
            changed=replace_slot_keep_weight(model,pool,region_id,replacement_slot,candidate)
            y=predict_from_pool(model,inputs,base,changed)
            mae=target_mae(y,truth,future_valid,mask,std)
            replacements.append({
                'coarse_rank':int(rank),'memory_id':int(candidate['coarse_id']),
                'replacement_slot':int(replacement_slot),
                'effective_node_fraction':float(effective.float().mean().item()),
                'mean_original_slot_mass':float(pool['weights'][:,nodes,replacement_slot].mean().item()),
                'replacement_mae':mae,
                'gain_vs_real':real_mae-mae,
                'gain_vs_region_null':null_mae-mae,
                'predicted_score':float(candidate['ranking'].mean().item()),
                **horizon_gains(real,y,truth,future_valid,mask,std,direct),
            })
    result['same_weight_replacements']=replacements
    result['replacement_best_improvement_over_real']=max([0.]+[x['gain_vs_real'] for x in replacements])
    result['replacement_random_mean_gain_vs_real']=_nanavg([x['gain_vs_real'] for x in replacements])
    result['replacement_beneficial_fraction']=_nanavg([x['gain_vs_real']>0 for x in replacements])
    result['replacement_ranker_spearman']=_spearman([x['predicted_score'] for x in replacements],
                                                    [x['gain_vs_real'] for x in replacements])
    result['replacement_best_memory_id']=(max(replacements,key=lambda x:x['gain_vs_real'])['memory_id']
                                          if replacements else None)

    # Exact prediction-space interpolation. alpha=0 NULL / alpha=1 Real.
    # This is a *retrospective region-level* scaling diagnostic, not an
    # existing trained gate or a deployable policy.
    scaled=[]
    if 'scaling' in modes:
        for alpha in scaling_factors:
            if not math.isfinite(alpha) or alpha<0:
                raise ValueError(f'Invalid finite nonnegative scaling factor: {alpha}')
            prediction=null+float(alpha)*(real-null)
            mae=target_mae(prediction,truth,future_valid,mask,std)
            scaled.append({'alpha':float(alpha),'prediction_mae':mae,
                'gain_vs_null':null_mae-mae,
                'improvement_over_real':real_mae-mae,
                **horizon_gains(null,prediction,truth,future_valid,mask,std,direct)})
    result['correction_scaling']=scaled
    result['best_scaling_alpha']=(min(scaled,key=lambda x:x['prediction_mae'])['alpha'] if scaled else None)
    result['best_scaling_improvement_over_real']=max([0.]+[x['improvement_over_real'] for x in scaled])
    result['scaling_alpha_0_null_parity']=None
    result['scaling_alpha_1_real_parity']=None
    for row in scaled:
        if row['alpha']==0.: result['scaling_alpha_0_null_parity']=abs(row['prediction_mae']-null_mae)
        if row['alpha']==1.: result['scaling_alpha_1_real_parity']=abs(row['prediction_mae']-real_mae)
    if result['scaling_alpha_0_null_parity'] is not None and result['scaling_alpha_0_null_parity']>1e-5:
        raise RuntimeError('Correction scaling alpha=0 must reproduce null MAE')
    if result['scaling_alpha_1_real_parity'] is not None and result['scaling_alpha_1_real_parity']>1e-5:
        raise RuntimeError('Correction scaling alpha=1 must reproduce Real MAE')

    # Node-conditioned slots, evaluated on the *same node*. The other region
    # nodes retain the original Top8 pool and may propagate information.
    node_records=[]
    if 'node_loo' in modes and max_nodes!=0:
        for node in _sample_region_nodes(nodes,max_nodes):
            node_mask=torch.zeros_like(mask)
            node_mask[node]=True
            for slot in range(width):
                if float(pool['weights'][0,node,slot].item())<=1e-9:
                    continue
                removed=leave_one_slot_out(model,pool,region_id,slot,node)
                y=predict_from_pool(model,inputs,base,removed)
                gain=target_mae(y,truth,future_valid,node_mask,std)-target_mae(
                    real,truth,future_valid,node_mask,std)
                node_records.append({'node_id':int(node),'slot':int(slot),
                    'memory_id':int(pool['ids'][0,node,slot].item()),
                    'original_weight':float(pool['weights'][0,node,slot].item()),
                    'keep_gain_vs_removing':gain,
                    **horizon_gains(y,real,truth,future_valid,node_mask,std,direct)})
    result['node_leave_one_out']=node_records
    result['node_loo_mean_keep_gain']=_nanavg([x['keep_gain_vs_removing'] for x in node_records])
    result['node_loo_harmful_fraction']=_nanavg([x['keep_gain_vs_removing']<0 for x in node_records])
    for i in (1,2,3):
        key=f'horizon_group_{i}_gain'
        result[f'loo_horizon_group_{i}_mean_keep_gain']=_nanavg([x.get(key,float('nan')) for x in loo])
        result[f'replacement_horizon_group_{i}_mean_gain']=_nanavg([x.get(key,float('nan')) for x in replacements])
        result[f'node_loo_horizon_group_{i}_mean_keep_gain']=_nanavg([x.get(key,float('nan')) for x in node_records])
    result['node_loo_direct_mean_keep_gain']=_nanavg([x.get('direct_gain',float('nan')) for x in node_records])
    result['loo_direct_mean_keep_gain']=_nanavg([x.get('direct_gain',float('nan')) for x in loo])
    result['replacement_direct_mean_gain']=_nanavg([x.get('direct_gain',float('nan')) for x in replacements])
    return result
