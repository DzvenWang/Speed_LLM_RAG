"""Train-only counterfactual observation sensitivity supervision.

Measures the FROZEN backbone forecast error change from removing additional
legally observed values. This is a model-sensitivity proxy, NOT an empirical
physical observability label or shockwave truth. Never accesses Val/Test labels.
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Subset
from .protocol import make_loader,device_batch


@torch.no_grad()
def build_support_teacher(model,train_ds,device,limit=256,seed=10,batch_size=8):
    rng=np.random.default_rng(seed)
    ids=np.linspace(0,len(train_ds)-1,min(len(train_ds),limit),dtype=int).tolist()
    loader=make_loader(Subset(train_ds,ids),batch_size,shuffle=False,workers=0)
    model.eval();labels={};examples=0
    graph=model.graph
    for raw in loader:
        b=device_batch(raw,device)
        base=model.backbone(b['x'],b['mask'],b['tod'],b['dow'])['prediction']
        new_mask=b['mask'].clone()
        new_input=b['x'].clone()
        B,H,N=new_mask.shape
        # Deterministic, non-future-informed spatially localized removals.
        for i in range(B):
            node=int(rng.integers(0,N))
            neighbor=graph.path_nodes[node][graph.path_valid[node]].detach().cpu().numpy()
            chosen=np.array([node]+[int(z) for z in neighbor[:3]],dtype=int)
            # Drop only ACTUALLY observable entries, never treat natural missing as truth.
            period=int(rng.integers(0,max(1,H-2)))
            new_mask[i,period:period+3,chosen]=False
            new_input[i,period:period+3,chosen]=0
        pred=model.backbone(new_input,new_mask,b['tod'],b['dow'])['prediction']
        v=b['future_valid']
        changed=(pred-b['target']).abs()-(base-b['target']).abs()       # [B,P,N]
        sensitivity=torch.relu(changed).permute(0,2,1)
        normalized=sensitivity/(1+sensitivity)
        for i,row in enumerate(b['row_id'].tolist()):
            labels[int(row)]={'sensitivity':normalized[i].detach().cpu().half(),
                              'valid':v[i].permute(1,0).detach().cpu()}
        examples+=B
    print(f'FARE5_SUPPORT_TEACHER_PASS queries={examples}',flush=True)
    return labels


def support_auxiliary_loss(out,batch,teacher,device):
    if not teacher:return out['prediction'].sum()*0,{'support_teacher_examples':0}
    import torch.nn.functional as F
    losses=[]
    for i,row in enumerate(batch['row_id'].tolist()):
        record=teacher.get(int(row))
        if record is None:continue
        valid=record['valid'].to(device).bool()
        if not valid.any():continue
        target=record['sensitivity'].to(device).float()
        prediction=out['support']['demand'][i]
        # Forecast-support demand is constrained using signed downstream
        # prediction sensitivity, no future features are fed into the model.
        losses.append(F.smooth_l1_loss(prediction[valid],target[valid]))
    if not losses:return out['prediction'].sum()*0,{'support_teacher_examples':0}
    return torch.stack(losses).mean(),{'support_teacher_examples':len(losses)}
