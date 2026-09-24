from __future__ import annotations
import torch


def make_inputs(x,tod_hist,dow_hist,device):
    x=x.to(device,non_blocking=True).float().unsqueeze(-1)
    B,T,N,_=x.shape
    tod=tod_hist.to(device,non_blocking=True).long().unsqueeze(-1).expand(B,T,N)
    dow=dow_hist.to(device,non_blocking=True).long().unsqueeze(-1).expand(B,T,N)
    te=torch.stack([tod,dow],dim=-1)
    return x,te


def restore_original_nodes(model,rex):
    B,Tp,_,D=rex.shape
    out=rex.new_zeros((B,Tp,int(model.node_num),D))
    out[:,:,model.ori_parts_idx,:]=rex[:,:,model.reo_parts_idx,:]
    return out


def decode_prehead(model,prehead_4d):
    B,Tp,N,D=prehead_4d.shape
    if int(model.regression_conv.in_channels)!=Tp*D:
        raise RuntimeError(f'Head channels mismatch {model.regression_conv.in_channels} != {Tp*D}')
    pred=model.regression_conv(prehead_4d.transpose(2,3).reshape(B,Tp*D,N,1))
    if pred.ndim==4 and pred.shape[-1]==1: pred=pred.squeeze(-1)
    return pred


def forward_with_deep_states(model,x,te,capture_layers=(2,4)):
    embedded=model.embedding(x,te); rex=embedded[:,:,model.reo_all_idx,:]
    states={'embedding_4d':embedded}; cap=set(int(i) for i in capture_layers)
    for i,block in enumerate(model.spa_encoder):
        rex=block(rex)
        if i in cap:
            states[f'layer_{i+1}_patched']=rex
            states[f'layer_{i+1}_node_4d']=restore_original_nodes(model,rex)
    prehead=restore_original_nodes(model,rex); pred=decode_prehead(model,prehead)
    states['prehead_4d']=prehead
    return pred,states
