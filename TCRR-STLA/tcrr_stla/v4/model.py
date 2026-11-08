"""Full TCRR-STLA v4 model, same forecast units as v3.

No free decoder: all corrections are generated from train-only counterfactual
retrieval evidence. The STLAformer and Mask Adapter are both frozen after S0.
"""
from __future__ import annotations
import torch
from torch import nn
from .transfer import CounterfactualResidualTransferV4

class TCRRSTLAV4(nn.Module):
    def __init__(self,stla,cfg,neighbours=None):
        super().__init__()
        self.stla=stla
        d=int(cfg['model'].get('latent_dim',16))
        self.latent_proj=nn.Sequential(nn.Linear(stla.model_dim,d),nn.GELU())
        self.v4=CounterfactualResidualTransferV4(
            int(cfg['experiment']['horizon']),d,int(cfg['model'].get('fusion_hidden',64)),
            float(cfg['model'].get('max_normalized_correction',3.0)),
            float(cfg.get('v4',{}).get('rank_utility_weight',.35)),
            float(cfg.get('v4',{}).get('utility_temperature',1.0)))
        if neighbours is None: neighbours=torch.empty(0,dtype=torch.long)
        self.register_buffer('neighbours',neighbours.long(),persistent=True)
        self.freeze_base()
    def freeze_base(self):
        for p in self.stla.parameters():p.requires_grad_(False)
        self.stla.eval()
    def set_stage(self,stage):
        self.v4.set_stage(stage)
        for p in self.latent_proj.parameters():p.requires_grad_(stage!='controller')
    def train(self,mode=True):
        super().train(mode)
        self.stla.eval()
        return self
    def forward(self,x,mask,tod,dow,candidate_y=None,residual=None,
                candidate_scores=None,candidate_coverage=None,hist_distance=None,
                candidate_valid=None,normal_residual=None,teacher_base=None,
                candidate_histories=None,candidate_ids=None,candidate_source_valid=None,
                base_prediction=None,frozen_context=None,mode='full'):
        if base_prediction is None or frozen_context is None:
            with torch.no_grad():
                base_prediction,frozen_context=self.stla(x,mask,tod,dow)
        base=base_prediction.float()
        ctx=self.latent_proj(frozen_context.float())
        if mode=='no_retrieval':
            zero=torch.zeros_like(base)
            return dict(prediction=base,base_prediction=base,correction=zero,
                gate=zero,candidate_weights=None,candidate_logits=None,
                candidate_residuals=None,candidate_available=None)
        if residual is None or normal_residual is None or candidate_valid is None:
            raise ValueError('v4 requires paired train-only normal/matched residual and valid mask')
        neigh=self.neighbours if self.neighbours.numel() else None
        out=self.v4(x,mask,base,ctx,normal_residual,residual,candidate_y,
                    candidate_scores,candidate_coverage,candidate_valid,
                    candidate_histories=candidate_histories,neighbours=neigh,
                    source_valid=candidate_source_valid,mode=mode)
        if self.training and residual.shape[1]>1 and mode=='full':
            fail=residual.float()-normal_residual.float()
            out['hard_shuffled_candidate']=normal_residual.float()[:,:1]+fail.roll(1,1)[:,:1]
            out['hard_shuffled_gain_pred']=self.v4.c12.hard_negative_gain(
                x,mask,base,ctx,normal_residual,fail,candidate_scores,candidate_coverage,
                candidate_histories,neigh,candidate_source_valid)
        else:
            out['hard_shuffled_candidate']=None
            out['hard_shuffled_gain_pred']=None
        out['candidate_ids']=candidate_ids
        out['retrieval_prediction']=base+out['proposed_correction']
        return out
