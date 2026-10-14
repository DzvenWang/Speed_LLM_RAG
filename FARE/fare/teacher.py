"""Train-only counterfactual teacher with frozen-model snapshots.

Never use validation/test targets to produce training labels.
"""
import torch
from .protocol import make_loader,device_batch

@torch.no_grad()
def generate_teacher(model,memory,train_ds,device,limit=128,batch_size=4,purge=0):
    model.eval();records={};count=0
    loader=make_loader(train_ds,batch_size,shuffle=False,workers=0)
    for raw in loader:
        if count>=limit:break
        batch=device_batch(raw,device)
        full=model(batch,memory,purge=purge)
        k=full['selection']['selected_ids'].shape[-1]
        slot=count%k
        alt=model(batch,memory,drop_slot=slot,purge=purge)
        # Positive keep gain: removing candidate increases error.
        keep=(alt['prediction']-batch['target']).abs()-(full['prediction']-batch['target']).abs()
        replacement_pos=count%max(1,full['candidate_pool']['ids'].shape[-1])
        replace=model(batch,memory,replace_slot=slot,replace_position=replacement_pos,purge=purge)
        # Positive replacement gain means replacement is better.
        rep=(full['prediction']-batch['target']).abs()-(replace['prediction']-batch['target']).abs()
        null=(full['null_prediction']-batch['target']).abs()
        real=(full['prediction']-batch['target']).abs()
        gain=null-real
        # Discrete TRAIN-only action teacher; use horizon blocks to reduce noise.
        action_grid=(0.,0.25,0.5,0.75,1.,1.25)
        action=torch.ones_like(gain)
        for h0 in range(0,gain.shape[1],4):
            h1=min(h0+4,gain.shape[1])
            errors=[]
            for alpha in action_grid:
                trial=full['null_prediction']+alpha*(full['prediction']-full['null_prediction'])
                err=(trial[:,h0:h1]-batch['target'][:,h0:h1]).abs()
                vv=batch['future_valid'][:,h0:h1].float()
                errors.append((err*vv).sum(1)/vv.sum(1).clamp_min(1))
            preference=torch.tensor(action_grid,device=device)[torch.stack(errors).argmin(0)]
            action[:,h0:h1]=preference[:,None,:]
        v=batch['future_valid'].bool()
        for i,idx in enumerate(batch['row_id'].tolist()):
            records[int(idx)]={'keep_gain':keep[i].permute(1,0).cpu().half(),
                               'replace_gain':rep[i].permute(1,0).cpu().half(),
                               'risk_gain':gain[i].permute(1,0).cpu().half(),
                               'action_alpha':action[i].permute(1,0).cpu().half(),
                               'valid':v[i].permute(1,0).cpu(),
                               'slot':slot,'replacement_pos':replacement_pos,
                               'selected_ids':full['selection']['selected_ids'][i].detach().cpu().long(),
                               'replacement_candidate':full['candidate_pool']['ids'][i,:,replacement_pos].detach().cpu().long()}
        count+=len(batch['row_id'])
    return records

def save_teacher(records,path,metadata):
    torch.save({'records':records,'metadata':metadata},path)

def load_teacher(path,expected):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    for key,value in expected.items():
        if obj['metadata'].get(key)!=value:raise RuntimeError(f'Teacher provenance mismatch {key}')
    return obj['records']
