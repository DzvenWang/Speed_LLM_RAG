from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn

from .deep import decode_prehead,restore_original_nodes


def missingness_features(observed: torch.Tensor) -> torch.Tensor:
    """Return [B,H,N,4] observation semantics.

    Channels are observation indicator, causal missing-run length at each step,
    age of the latest observation at the forecast origin, and history coverage.
    All non-binary channels are normalized to [0,1].
    """
    if observed.ndim!=3: raise ValueError(f'observed must be [B,H,N], got {tuple(observed.shape)}')
    obs=observed.float(); miss=1.0-obs; B,H,N=obs.shape
    run=torch.zeros_like(obs)
    for t in range(H):
        run[:,t]=miss[:,t]*(1.0+(run[:,t-1] if t else 0.0))
    run=run/max(float(H),1.0)
    age=run[:,-1:].expand(B,H,N)
    coverage=obs.mean(dim=1,keepdim=True).expand(B,H,N)
    return torch.stack([obs,run,age,coverage],dim=-1)


def nested_restore_mask(restorable,indices,fraction,seed):
    """Select an exact, deterministic, nested fraction of real hidden cells."""
    r=np.asarray(restorable,dtype=bool); ids=np.asarray(indices,dtype=np.int64)
    if r.ndim!=3 or len(r)!=len(ids): raise ValueError('restorable/indices shape mismatch')
    f=float(fraction)
    if f<0 or f>1: raise ValueError(f'fraction outside [0,1]: {f}')
    out=np.zeros_like(r,bool)
    if f==0: return out
    if f==1: return r.copy()
    for i,qid in enumerate(ids):
        pos=np.flatnonzero(r[i].reshape(-1)); k=int(np.floor(f*len(pos)+1e-12))
        if k<=0: continue
        # Recreating the same permutation for every fraction makes the curve nested.
        ss=np.random.SeedSequence([int(seed),int(qid)&0xffffffff])
        chosen=np.random.default_rng(ss).permutation(pos)[:k]
        out[i].reshape(-1)[chosen]=True
    return out


def apply_true_coverage(x_obs,x_complete,observed,restorable,indices,fraction,seed):
    """Restore actual cells and update the observation mask consistently."""
    selected=nested_restore_mask(
        restorable.detach().cpu().numpy() if torch.is_tensor(restorable) else restorable,
        indices.detach().cpu().numpy() if torch.is_tensor(indices) else indices,
        fraction,seed,
    )
    sel=torch.as_tensor(selected,dtype=torch.bool,device=x_obs.device)
    xc=x_complete.to(x_obs.device)
    xr=torch.where(sel,xc,x_obs)
    mr=observed.to(x_obs.device).bool()|sel
    return xr,mr,sel


class ObservationAwarePatchSTG(nn.Module):
    """Non-invasive missingness adapter around the pinned PatchSTG.

    The adapter is added after the official value/time/node embedding and is
    zero-initialized, so its initial forecast is exactly the frozen checkpoint.
    The optional reconstruction head reads the shared post-spatial state but
    its output is never fed back into PatchSTG.
    """
    def __init__(self,base_model,history,mask_hidden=64):
        super().__init__(); self.base_model=base_model; self.history=int(history)
        patch=int(base_model.input_st_fc.kernel_size[1]); stride=int(base_model.input_st_fc.stride[1])
        dims=(int(base_model.input_st_fc.out_channels)+int(base_model.node_emb.shape[1])+
              int(base_model.time_in_day_emb.shape[1])+int(base_model.day_in_week_emb.shape[1]))
        tokens=(self.history-patch)//stride+1
        if tokens<=0: raise ValueError(f'history={history} incompatible with temporal patch={patch}')
        if int(base_model.regression_conv.in_channels)!=tokens*dims:
            raise RuntimeError('Inferred PatchSTG temporal layout disagrees with regression head')
        hidden=int(mask_hidden)
        self.mask_encoder=nn.Sequential(
            nn.Conv2d(4,hidden,kernel_size=(1,patch),stride=(1,stride),bias=True),
            nn.GELU(),
            nn.Conv2d(hidden,dims,kernel_size=(1,1),bias=True),
        )
        nn.init.zeros_(self.mask_encoder[-1].weight); nn.init.zeros_(self.mask_encoder[-1].bias)
        self.state_head=nn.Conv2d(tokens*dims,self.history,kernel_size=(1,1),bias=True)
        nn.init.zeros_(self.state_head.weight); nn.init.zeros_(self.state_head.bias)
    def set_trainable(self,freeze_backbone=True,train_regression_head=False,train_state_head=True):
        for p in self.base_model.parameters(): p.requires_grad=not bool(freeze_backbone)
        if freeze_backbone and train_regression_head:
            for p in self.base_model.regression_conv.parameters(): p.requires_grad=True
        for p in self.mask_encoder.parameters(): p.requires_grad=True
        for p in self.state_head.parameters(): p.requires_grad=bool(train_state_head)
        self.freeze_backbone=bool(freeze_backbone)
    def train(self,mode=True):
        super().train(mode)
        # A frozen PatchSTG remains in inference mode so attention dropout does
        # not turn the fixed reference forecast into a moving training target.
        if getattr(self,'freeze_backbone',False): self.base_model.eval()
        return self
    def forward(self,x,te,observed,return_state=False):
        embedded=self.base_model.embedding(x,te)
        mf=missingness_features(observed).permute(0,3,2,1).contiguous()
        delta=self.mask_encoder(mf).permute(0,3,2,1).contiguous()
        if delta.shape!=embedded.shape:
            raise RuntimeError(f'mask embedding {tuple(delta.shape)} != base embedding {tuple(embedded.shape)}')
        rex=(embedded+delta)[:,:,self.base_model.reo_all_idx,:]
        for block in self.base_model.spa_encoder: rex=block(rex)
        prehead=restore_original_nodes(self.base_model,rex)
        forecast=decode_prehead(self.base_model,prehead)
        if not return_state: return forecast
        B,Tp,N,D=prehead.shape
        head_in=prehead.transpose(2,3).reshape(B,Tp*D,N,1)
        state=self.state_head(head_in).squeeze(-1)
        return forecast,state,{'mask_delta':delta,'prehead':prehead}


def build_observation_aware(base_model,history,mask_hidden=64):
    return ObservationAwarePatchSTG(copy.deepcopy(base_model),history,mask_hidden)


def masked_mae_torch(pred,target,mask,weights=None):
    m=mask.bool()&torch.isfinite(pred)&torch.isfinite(target)
    if not torch.any(m): return (pred*0).sum()
    err=torch.abs(pred-target)
    if weights is None: return err[m].mean()
    w=torch.broadcast_to(weights,err.shape)
    return (err[m]*w[m]).sum()/torch.clamp(w[m].sum(),min=1e-8)
