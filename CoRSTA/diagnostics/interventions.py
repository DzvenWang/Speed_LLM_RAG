"""Exactly-scoped forward hooks for read-only feature-path sensitivity tests.

These are inference interventions on a trained model, not retrained ablations.
"""
from __future__ import annotations
from contextlib import contextmanager
import torch

MODES=('full','no_c2_bridge','no_c1_bridge','no_feedback','no_functional','no_directional')

@contextmanager
def intervene(model,mode):
    if mode not in MODES:raise ValueError(mode)
    handles=[];call=[0]
    def zero_bridge(module,inputs,out):return torch.zeros_like(out)
    def block_graph(module,inputs,out):
        i=call[0];call[0]+=1
        if (mode=='no_functional' and i==0) or (mode=='no_directional' and i in (1,2)):
            return torch.zeros_like(out)
        return out
    try:
        if mode in ('no_c1_bridge','no_feedback'):handles.append(model.bridge1.register_forward_hook(zero_bridge))
        if mode in ('no_c2_bridge','no_feedback'):handles.append(model.bridge2.register_forward_hook(zero_bridge))
        if mode in ('no_functional','no_directional'):handles.append(model.c2.message.register_forward_hook(block_graph))
        yield
    finally:
        for h in handles:h.remove()

class FusionCapture:
    """Capture just-enough tensors for actual feature injection, then release."""
    def __init__(self,model):self.model=model;self.handles=[];self.data={}
    def __enter__(self):
        self.handles=[
            self.model.bridge1.register_forward_hook(lambda m,a,o:self.data.__setitem__('a1',o)),
            self.model.bridge2.register_forward_hook(lambda m,a,o:self.data.__setitem__('a2',o)),
            self.model.fusion.register_forward_pre_hook(lambda m,a:self.data.__setitem__('fusion_inputs',a)),
            self.model.c2.register_forward_hook(lambda m,a,o:self.data.__setitem__('c2_aux',o[2])),
        ]
        return self
    def get(self):
        inputs=self.data['fusion_inputs'];h,a1,a2,r1,r2,mask=inputs
        layer=self.model.fusion;B,L,N,D=h.shape
        r1_hist=r1.mean(1)[:,None,:,None].expand(-1,L,-1,-1)
        r2_hist=r2.mean(1)[:,None,:,None].expand(-1,L,-1,-1)
        cov=mask.float().mean(1)[:,None,:,None].expand(-1,L,-1,-1)
        common=torch.cat((h,r1_hist,r2_hist,cov),-1)
        g1=torch.sigmoid(layer.g1(common))*r1_hist
        g2=torch.sigmoid(layer.g2(common))*r2_hist
        contrib1=layer.scale*g1*a1
        contrib2=layer.scale*g2*a2
        norm=h.square().sum(dim=(-1,-2,-3)).sqrt().clamp_min(1e-8)
        ratio1=contrib1.square().sum(dim=(-1,-2,-3)).sqrt()/norm
        ratio2=contrib2.square().sum(dim=(-1,-2,-3)).sqrt()/norm
        return {'injection_c1':ratio1.detach(),'injection_c2':ratio2.detach(),
                'gate_c1':g1.detach(),'gate_c2':g2.detach(),
                'reliability_c1':r1.detach(),'reliability_c2':r2.detach(),
                'graph_mix':self.data['c2_aux']['graph_mix'].detach(),
                'scale':float(layer.scale.detach())}
    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.data.clear()

@contextmanager
def swap_student(model,other_student):
    """Temporary inference-only checkpoint composition; restore even on error."""
    original=model.student
    try:
        model.student=other_student
        yield
    finally:
        model.student=original
