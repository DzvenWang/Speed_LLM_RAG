"""Exact original STLA forward; parallel representations never perturb its input."""
from __future__ import annotations
import torch
from torch import nn


def official_input(batch, model):
    x = batch['x_target']
    b,h,n = x.shape
    if h != int(model.in_steps) or n != int(model.num_nodes):
        raise ValueError(f'STLA input dimensions {x.shape} vs checkpoint {model.in_steps}/{model.num_nodes}')
    if int(model.input_dim) != 2:raise ValueError('This STLA checkpoint requires input_dim=2')
    slots=batch['dow'].long()*288+batch['tod'].long()
    week=torch.nextafter(slots.float()/float(model.steps_per_week),torch.full_like(slots.float(),float('inf')))
    week=week[:,:,None].expand(b,h,n)
    return torch.stack((x,week),dim=-1)


class STLAAnchor(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.base=base
        self.frozen=True
        self._last_hidden=None
        layers=getattr(base,'attn_layers_st',None)
        if layers is None or len(layers)==0:raise ValueError('STLA source has no attn_layers_st')
        # Collect the hidden state from the same native forward pass, no second encoder run.
        self._hook=layers[-1].register_forward_hook(self._capture)
        self.dimension=int(base.model_dim)
    def _capture(self,module,args,output):
        self._last_hidden=output[0] if isinstance(output,tuple) else output
    def set_trainable_tail(self,n):
        for p in self.base.parameters():p.requires_grad_(False)
        if n:
            for layer in list(self.base.attn_layers_st)[-int(n):]:
                for p in layer.parameters():p.requires_grad_(True)
            for p in self.base.output_proj.parameters():p.requires_grad_(True)
        self.frozen=not bool(n)
        if self.frozen:self.base.eval()
    def train(self,mode=True):
        super().train(mode)
        if self.frozen:self.base.eval()
        return self
    def forward(self,batch):
        raw=official_input(batch,self.base)
        self._last_hidden=None
        if self.frozen:
            with torch.no_grad():prediction=self.base(raw)
        else:
            prediction=self.base(raw)
        if prediction.ndim==4 and prediction.shape[-1]==1:prediction=prediction.squeeze(-1)
        if prediction.ndim!=3:raise ValueError(f'Official STLA output must be [B,P,N], got {prediction.shape}')
        hidden=self._last_hidden
        if hidden is None or hidden.ndim!=4:raise RuntimeError('Official STLA last attention layer hook did not fire')
        if hidden.shape[:3]!=(raw.shape[0],raw.shape[1],raw.shape[2]):
            raise ValueError(f'Official STLA hidden shape mismatch {hidden.shape}')
        # Node context from genuine native forward; no future target consumed.
        return {'prediction':prediction,'node_features':hidden[:,-1].float(),
                'history_features':hidden.float()}
    def parity(self,batch,atol=1e-5,rtol=1e-5):
        self.eval()
        with torch.no_grad():
            a=self(batch)['prediction']
            b=self.base(official_input(batch,self.base))
            if b.ndim==4:b=b.squeeze(-1)
        if not torch.allclose(a,b,atol=atol,rtol=rtol):
            raise AssertionError(f'STLA parity failed max_diff={(a-b).abs().max().item()}')
        return float((a-b).abs().max().item())
