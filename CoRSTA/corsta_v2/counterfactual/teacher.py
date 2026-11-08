"""Paired full/observed teacher targets from identical frozen STLA weights."""
from __future__ import annotations
import torch
from torch import nn
from corsta_v2.stla.backbone import base_features


class PairedTeacher(nn.Module):
    def __init__(self,frozen_stla):
        super().__init__()
        self.stla=frozen_stla
        self.stla.requires_grad_(False)
        self.stla.eval()

    def train(self,mode=True):
        super().train(False)
        self.stla.eval()
        return self

    @torch.no_grad()
    def forward(self,batch,observed_h0=None,observed_p0=None):
        if 'x_full' not in batch or 'original_valid' not in batch:
            raise ValueError('Paired Teacher forbidden outside TRAIN')
        hf,ctx,pf=base_features(self.stla,batch['x_full'],batch['original_valid'],batch['tod'],batch['dow'])
        if observed_h0 is None or observed_p0 is None:
            hm,_,pm=base_features(self.stla,batch['x'],batch['mask'],batch['tod'],batch['dow'])
        else:hm,pm=observed_h0.detach(),observed_p0.detach()
        y=torch.where(batch['future_valid'].bool(),batch['y_norm'].float(),torch.zeros_like(batch['y_norm'].float()))
        cf=pf-pm;res=y-pf
        total=y-pm
        if not torch.allclose((cf+res),total,atol=2e-5,rtol=2e-5):
            raise RuntimeError('CF/Residual target decomposition broken')
        return {'p_full':pf.detach(),'p_masked':pm.detach(),
                'cf_target':cf.detach(),'residual_target':res.detach(),
                'latent_cf_target':(hf-hm).detach(),
                'cf_utility':((pm-y).abs()-(pf-y).abs()).detach()}
