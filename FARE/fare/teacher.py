"""Exact train-only counterfactual teacher with snapshot-context replay.

LOO: removed Evidence weight moved to Null. Swap: same weight budget, only
legal previously unselected candidate accepted. Risk actions scale unscaled
candidate-minus-null, not an already risk-scaled final prediction.
"""
from __future__ import annotations
from pathlib import Path
import torch,numpy as np
from torch.utils.data import Subset
from .protocol import make_loader,device_batch


def _detach_cpu(x):
    if torch.is_tensor(x):
        y=x.detach().cpu()
        return y.half() if y.is_floating_point() else y
    if isinstance(x,dict):return {k:_detach_cpu(v) for k,v in x.items() if isinstance(v,(dict,torch.Tensor))}
    return x

def sample_legal_replacement(pool):
    """Sample candidate positions from a [B,N,K] legal-unselected mask.

    Empty candidate rows use index zero only as a placeholder; their returned
    validity is false. No invalid replacement may contribute to Swap loss.
    """
    if pool.ndim != 3 or pool.shape[-1] == 0:
        raise ValueError("Expected non-empty [B,N,K] legal replacement mask")
    b, n, k = pool.shape
    choices = pool.to(dtype=torch.float32)
    empty = ~pool.any(dim=-1)
    choices[..., 0] = torch.where(
        empty, torch.ones_like(choices[..., 0]), choices[..., 0]
    )
    positions = torch.multinomial(choices.reshape(-1, k), 1).reshape(b, n)
    valid = pool.gather(-1, positions.unsqueeze(-1)).squeeze(-1)
    return positions, valid


@torch.no_grad()
def generate_teacher(model,memory,train_ds,device,limit=128,batch_size=2,purge=0,
                     seed=10,nodes_per_query=0):
    model.eval();records={};count=0
    rng=np.random.default_rng(seed)
    indices=np.linspace(0,len(train_ds)-1,min(int(limit),len(train_ds)),dtype=int).tolist()
    ds=Subset(train_ds,indices)
    loader=make_loader(ds,batch_size,shuffle=False,workers=0)
    slot_usage={}
    for batch_index,raw in enumerate(loader):
        batch=device_batch(raw,device)
        full=model(batch,memory,purge=purge)
        cache=full['prepared'];selected=full['selection'];short=full['candidate_pool']
        k=selected['selected_ids'].shape[-1];K=short['ids'].shape[-1]
        slot=batch_index%k;slot_usage[slot]=slot_usage.get(slot,0)+len(batch['row_id'])
        loo=model.edit_selection(selected,cache['proposals'],short,drop_slot=slot)
        minus=model.predict_with_selection(batch,cache,loo)
        # Sample a distinct unselected replacement for each query/node.
        available=short['valid']
        unchosen=~(short['ids'][...,None]==selected['selected_ids'][:,:,None,:]).any(-1)
        pool=available&unchosen
        # Sampling handles [B,N] rows without any valid unselected candidate.
        # Those rows obtain a harmless placeholder and changed=False.
        position,changed=sample_legal_replacement(pool)
        swap=model.edit_selection(selected,cache['proposals'],short,replace_slot=slot,replace_position=position)
        replaced=model.predict_with_selection(batch,cache,swap)
        target=batch['target'];truth=batch['future_valid'].bool()
        full_err=(full['prediction']-target).abs()
        keep_gain=(minus['prediction']-target).abs()-full_err
        swap_gain=full_err-(replaced['prediction']-target).abs()
        risk_gain=(full['null_prediction']-target).abs()-full_err
        action_grid=torch.tensor([0.,.25,.5,.75,1.,1.25],device=device)
        # Risk Teacher optimizes exactly the action multiplier used by FARE inference.
        delta=full['candidate_prediction']-full['null_prediction']
        action=torch.ones_like(delta)
        for lo in range(0,target.shape[1],4):
            hi=min(lo+4,target.shape[1]);errors=[]
            for alpha in action_grid:
                candidate=full['null_prediction'][:,lo:hi]+alpha*delta[:,lo:hi]
                valid=truth[:,lo:hi].float()
                errors.append(((candidate-target[:,lo:hi]).abs()*valid).sum(1)/valid.sum(1).clamp_min(1))
            chosen=action_grid[torch.stack(errors).argmin(0)]
            action[:,lo:hi]=chosen[:,None,:]
        for i,row in enumerate(batch['row_id'].tolist()):
            # Memory efficient: replay a spread of nodes including missing and observed.
            n=short['ids'].shape[1]
            if nodes_per_query and nodes_per_query<n:
                miss=(~batch['mask'][i].any(0)).nonzero().flatten().cpu().numpy()
                sampled=rng.choice(n,size=min(nodes_per_query,n),replace=False)
                sample=np.unique(np.concatenate([miss[:nodes_per_query//2],sampled]))[:nodes_per_query]
            else:sample=np.arange(n)
            ni=torch.as_tensor(sample,device=device,dtype=torch.long)
            def node_dict(d):
                return {name:_detach_cpu(value[i:i+1,ni]) for name,value in d.items()
                        if torch.is_tensor(value) and value.ndim>=2 and value.shape[1]==n}
            support=node_dict(full['support'])
            cand=node_dict(short)
            prop=node_dict(cache['proposals'])
            records[int(row)]={
                'support':support,'short':cand,'proposals':prop,
                'teacher_selection':node_dict(selected),
                'selected_positions':_detach_cpu(selected['selected_positions'][i:i+1,ni]),
                'selected_ids':_detach_cpu(selected['selected_ids'][i:i+1,ni]),
                'slot':slot,'replacement_pos':_detach_cpu(position[i:i+1,ni]),
                'replacement_valid':_detach_cpu(changed[i:i+1,ni]),
                'keep_gain':_detach_cpu(keep_gain[i:i+1,:,ni].permute(0,2,1)),
                'replace_gain':_detach_cpu(swap_gain[i:i+1,:,ni].permute(0,2,1)),
                'risk_gain':_detach_cpu(risk_gain[i:i+1,:,ni].permute(0,2,1)),
                'action_alpha':_detach_cpu(action[i:i+1,:,ni].permute(0,2,1)),
                'valid':_detach_cpu(truth[i:i+1,:,ni].permute(0,2,1)),
                'node_ids':_detach_cpu(ni),
                'null_prediction':_detach_cpu(full['null_prediction'][i:i+1,:,ni].permute(0,2,1)),
                'candidate_prediction':_detach_cpu(full['candidate_prediction'][i:i+1,:,ni].permute(0,2,1)),
            }
        count+=len(batch['row_id'])
    print('FARE4_TEACHER_STATS '+str({'queries':len(records),'slot_usage':slot_usage}),flush=True)
    return records


def save_teacher(records,path,metadata):
    torch.save({'records':records,'metadata':metadata},path)


def load_teacher(path,expected):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    for k,v in expected.items():
        if obj['metadata'].get(k)!=v:raise RuntimeError(f'Teacher provenance mismatch: {k}')
    return obj['records']


def replay_loss(model,teacher_records,device,max_replays=8):
    """Student Utility/Risk forward over FIXED teacher candidate sets; no equality match discard."""
    if not teacher_records:
        return next(model.parameters()).sum()*0,{'teacher_examples':0}
    import torch.nn.functional as F
    losses=[];keeps=[];swaps=[];risks=[];valid_count=0
    for record in list(teacher_records.values())[:max_replays]:
        d=lambda x:x.to(device,dtype=torch.float32) if torch.is_tensor(x) and x.is_floating_point() else x.to(device)
        short={k:d(v) for k,v in record['short'].items()}
        prop={k:d(v) for k,v in record['proposals'].items()}
        sup={k:d(v) for k,v in record['support'].items()}
        # The frozen Teacher selected set is used for exact label correspondence.
        teacher_pos=d(record['selected_positions'])
        pred=model.utility(short,prop,sup,forced_initial=teacher_pos)
        k=pred['all_utility'].shape[2]
        slot=int(record['slot'])
        chosen=teacher_pos[:,:,slot]
        i=pred['all_utility'].gather(2,chosen[...,None,None,None].expand(-1,-1,1,2,pred['all_utility'].shape[-1]))
        keep=i.squeeze(2).mean(2)
        actual_keep=d(record['keep_gain']);valid=d(record['valid']).bool()
        if valid.any():
            keeps.append(F.smooth_l1_loss(keep[valid],actual_keep[valid]))
            # Direct pairwise swap score uses teacher selected slot and legal candidate per node.
            repl=d(record['replacement_pos']).long()
            estimated=pred['swap_scores'][:,:,slot,:].gather(-1,repl[...,None]).squeeze(-1)
            target_swap=d(record['replace_gain'])
            good=valid & d(record['replacement_valid']).bool()[...,None]
            if good.any():
                estimate=estimated[...,None].expand_as(target_swap)
                swaps.append(F.smooth_l1_loss(estimate[good],target_swap[good]))
            actual_alpha=d(record['action_alpha'])
            # Snapshot null and candidate correspond to action labels. Risk consumes the
            # replayed current utility weights, but cannot see any Train target as feature.
            from .model import RiskController
            integrated={'null_prediction':d(record['null_prediction']).permute(0,2,1),
                        'candidate_prediction':d(record['candidate_prediction']).permute(0,2,1)}
            # Replay Risk uses the EXACT frozen Teacher selection, preventing
            # the Alpha label from accidentally supervising a new evidence set.
            risk_selection={k:d(v) for k,v in record['teacher_selection'].items()}
            risk=model.risk(sup,integrated,risk_selection)
            target_gain=d(record['risk_gain'])
            harm=(target_gain<0).float()
            risks.append(F.smooth_l1_loss(risk['alpha'][valid],actual_alpha[valid]) +
                         .5*F.smooth_l1_loss(risk['gain_estimate'][valid],target_gain[valid]) +
                         .2*F.binary_cross_entropy_with_logits(risk['harm_logit'][valid],harm[valid]))
            valid_count+=int(valid.sum())
    if not keeps:return next(model.parameters()).sum()*0,{'teacher_examples':0}
    main=torch.stack(keeps).mean()
    if swaps:main=main+torch.stack(swaps).mean()
    if risks:main=main+.5*torch.stack(risks).mean()
    return main,{'teacher_examples':len(keeps),'teacher_valid_elements':valid_count,
                 'teacher_keep_loss':float(torch.stack(keeps).mean().detach()),
                 'teacher_swap_labels':len(swaps),'teacher_risk_labels':len(risks)}
