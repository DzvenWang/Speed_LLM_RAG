"""Paired source-mask perturbations on the SAME frozen checkpoint/samples."""
from __future__ import annotations
import torch

SIMPLE=('uniform','fixed_shrink','hard_action')
SOURCE=('no_functional','no_physical','forward_only','reverse_only','fresh_only','skip_coverage_multiply')
POSTHOC=('no_c1','no_c2','no_local','no_direction','no_null')
ALL_INTERVENTIONS=SOURCE+POSTHOC


def fixed_predictions(out):
    can=out['candidates']
    weights=out['policy']['weights']
    return {'base':can[...,0], 'local':can[...,1], 'propagation':can[...,2],
            'controlled':out['prediction'], 'uniform':can.mean(-1),
            'fixed_shrink':can[...,0]+(2/3)*out['local_delta']+(1/3)*out['prop_delta'],
            'hard_action':can.gather(-1,weights.argmax(-1,keepdim=True)).squeeze(-1),
            # Uses future targets only in analysis, never served as real inference.
           }

def source_keep_mask(model,batch,name,max_age=3):
    prop=model.propagation
    ep=int(prop.edge_index.shape[1]);ef=int(prop.functional_index.shape[1]) if prop.functional else 0;e=ep+ef
    if name=='no_functional':return torch.cat((torch.ones(ep),torch.zeros(ef))).bool()
    if name=='no_physical':return torch.cat((torch.zeros(ep),torch.ones(ef))).bool()
    if name=='forward_only':return torch.cat((prop.edge_attr[:,6]>0.5,torch.zeros(ef,dtype=torch.bool)))
    if name=='reverse_only':return torch.cat((prop.edge_attr[:,7]>0.5,torch.zeros(ef,dtype=torch.bool)))
    if name=='fresh_only':
        target=batch['observed_mask'][...,model.target_idx]>0.5
        B,H,N=target.shape
        both=target[:,1:]&target[:,:-1]
        t=torch.arange(1,H,device=target.device)[None,:,None]
        last=torch.where(both,t,-1).amax(dim=1)
        age=torch.where(last>=0,(H-1-last),H)
        src=torch.cat((prop.edge_index[0],prop.functional_index[0] if prop.functional else prop.functional_index[0,:0]))
        return age[:,src]<=int(max_age)
    raise ValueError('Unsupported source intervention '+name)


def run_intervention(model,batch,baseline_out,name,max_age=3):
    """Returns re-evaluated strategy and fixed-policy response.

    Interventions do not retrain and are NOT independent component ablations.
    """
    if name in POSTHOC:
        fields={'no_c1':'use_support','no_c2':'use_propagation','no_local':'use_local',
                'no_direction':'use_direction','no_null':'use_null_source'}
        saved=dict(model.ablation)
        try:
            model.set_ablation(**{fields[name]:False})
            outcome=model(batch)
        finally:
            model.set_ablation(**saved)
    elif name=='skip_coverage_multiply':
        outcome=model(batch,coverage_intervention=True)
    else:
        keep=source_keep_mask(model,batch,name,max_age=max_age)
        outcome=model(batch,edge_keep_mask=keep)
    unchanged_policy=(baseline_out['policy']['weights']*outcome['candidates']).sum(-1)
    return outcome['prediction'],unchanged_policy,outcome['candidates'][...,2]
