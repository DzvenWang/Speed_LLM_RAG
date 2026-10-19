"""Train-only multi-candidate counterfactual labels and ranker/risk replay.

Labels use realized target improvement, never a similarity pseudo-label.
Each record is tagged by the frozen teacher model step / provenance.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from .protocol import make_loader,device_batch
from .forecast_utility import utility_loss


def _cpu(x):
    if torch.is_tensor(x):
        a=x.detach().cpu()
        return a.half() if a.is_floating_point() else a
    if isinstance(x,dict):return {k:_cpu(v) for k,v in x.items() if isinstance(v,dict) or torch.is_tensor(v)}
    return x

@torch.no_grad()
def generate_utility_teacher(model,memory,train_ds,device,limit=256,batch_size=2,
                             purge=0,seed=10,nodes_per_query=16,candidates=6):
    model.eval();records={};rng=np.random.default_rng(seed)
    idx=np.linspace(0,len(train_ds)-1,min(len(train_ds),int(limit)),dtype=np.int64)
    ds=Subset(train_ds,idx.tolist())
    for raw in make_loader(ds,batch_size,workers=0):
        batch=device_batch(raw,device)
        full=model(batch,memory,purge=purge)
        prep=full['prepared'];short=prep['candidate_pool']
        b,n,k=short['ids'].shape;p=batch['target'].shape[1]
        labels=torch.zeros(b,n,k,p,device=device);checked=torch.zeros_like(labels,dtype=torch.bool)
        # Stratify over both high-ranked and poorly-ranked legal candidates.
        # Score-head hard-negative mining: include the highest similarity items,
        # low-ranked candidates, and uniformly sampled contrastive candidates.
        # Actual harmful/beneficial labels are NEVER inferred from similarity.
        fixed=[0,min(1,k-1),max(0,k-1)]
        remaining=max(0,int(candidates)-len(set(fixed)))
        others=rng.choice(k,size=min(k,remaining),replace=False) if remaining else []
        indices=np.unique(np.asarray([*fixed,*others],dtype=int))
        ref=full['base_prediction']
        base_error=(ref-batch['target']).abs().permute(0,2,1)
        target_valid=batch['future_valid'].bool().permute(0,2,1)
        for slot in indices:
            positions=torch.full((b,n),int(slot),device=device,dtype=torch.long)
            trial=model.candidate_single(batch,prep,positions)
            err=(trial['prediction']-batch['target']).abs().permute(0,2,1)
            labels[:,:,slot]=base_error-err
            checked[:,:,slot]=target_valid & short['valid'][:,:,slot,None] & prep['proposals']['available'][:,:,slot].any(2)
        # Risk target is the optimal TRAIN-only interpolation along frozen evidence delta.
        grid=ref.new_tensor([0.,.25,.5,.75,1.])
        movement=full['candidate_prediction']-ref
        err=torch.stack([(ref+a*movement-batch['target']).abs() for a in grid],0)
        action=grid[err.argmin(0)].permute(0,2,1)
        actual_gain=(base_error-(full['prediction']-batch['target']).abs().permute(0,2,1))
        for j,row in enumerate(batch['row_id'].tolist()):
            if nodes_per_query and nodes_per_query<n:
                missing=(~batch['mask'][j].any(0)).nonzero().flatten().cpu().numpy()
                chosen=rng.choice(n,size=min(n,int(nodes_per_query)),replace=False)
                chosen=np.unique(np.concatenate([chosen,missing[:max(1,nodes_per_query//4)]]))[:nodes_per_query]
            else:chosen=np.arange(n)
            node=torch.as_tensor(chosen,dtype=torch.long,device=device)
            def take(d):
                return {name:_cpu(value[j:j+1,node]) for name,value in d.items()
                        if torch.is_tensor(value) and value.ndim>=2 and value.shape[1]==n}
            records[int(row)]={'support':take(prep['support']),
                'short':take(prep['candidate_pool']),'proposal':take(prep['proposals']),
                'labels':_cpu(labels[j:j+1,node]),'checked':_cpu(checked[j:j+1,node]),
                'risk_action':_cpu(action[j:j+1,node]),'risk_gain':_cpu(actual_gain[j:j+1,node]),
                'valid':_cpu(target_valid[j:j+1,node]),
                'integrated':{'null_prediction':_cpu(ref[j:j+1].index_select(2,node)),
                              'candidate_prediction':_cpu(full['candidate_prediction'][j:j+1].index_select(2,node))},
                'selection':take(full['selection'])}
    print(f'FARE5_COUNTERFACTUAL_TEACHER_PASS rows={len(records)} candidates={len(indices)}',flush=True)
    return records


def replay_utility_loss(model,records,device,max_replays=3):
    if not records:return next(model.parameters()).sum()*0,{'utility_records':0}
    def move(v):
        if torch.is_tensor(v):return v.to(device=device,dtype=torch.float32 if v.is_floating_point() else v.dtype)
        return {k:move(x) for k,x in v.items()}
    losses=[];h_losses=[];a_losses=[];counts=[];deficits=[]
    for record in list(records.values())[:max_replays]:
        sup,short,prop=(move(record[x]) for x in ('support','short','proposal'))
        sup['forecast_deficit_logits']=model.deficit(sup['embedding']).squeeze(-1)
        sup['forecast_deficit']=sup['forecast_deficit_logits'].sigmoid()
        rank=model.ranker(short,prop,sup)
        lbl=move(record['labels']);valid=move(record['checked']).bool()
        if valid.any():
            losses.append(utility_loss(rank,lbl,valid))
            counts.append(int(valid.sum()))
            supervised=valid.any(2)
            if supervised.any():
                helps=((lbl>.02)&valid).any(2).float()
                deficits.append(F.binary_cross_entropy_with_logits(sup['forecast_deficit_logits'][supervised],helps[supervised]))
        # Risk is trained with exactly the teacher's frozen selected evidence.
        integ=move(record['integrated'])
        selection=move(record['selection'])
        # Stored prediction layout already [B,H,N].
        predicted=model.risk(sup,integ,selection)
        good=move(record['valid']).bool()
        if good.any():
            action=move(record['risk_action'])
            gain=move(record['risk_gain'])
            a_losses.append(F.smooth_l1_loss(predicted['alpha'][good],action[good]) +
                            .2*F.smooth_l1_loss(predicted['gain_estimate'][good],gain[good]))
            h_losses.append(F.binary_cross_entropy_with_logits(predicted['harm_logit'][good],
                                  (gain[good]<-.02).float()))
    reference=next(model.parameters()).sum()*0
    loss=(torch.stack(losses).mean() if losses else reference)
    if deficits:loss=loss+.2*torch.stack(deficits).mean()
    if a_losses:loss=loss+.2*torch.stack(a_losses).mean()+.1*torch.stack(h_losses).mean()
    return loss,{'utility_records':len(losses),'utility_labels':sum(counts)}
