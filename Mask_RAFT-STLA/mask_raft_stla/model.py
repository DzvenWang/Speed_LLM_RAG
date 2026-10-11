"""Mask_RAFT-STLA v2: query-mask counterfactual residual transfer.

Prediction corrections are evidence-anchored. The free adapter is isolated
inside the explicitly trained no_retrieval control, never used in full mode.
"""
from __future__ import annotations
import torch
from torch import nn
from .stla import MaskAwareSTLAformer


class MaskRAFTSTLA(nn.Module):
    def __init__(self, stla: MaskAwareSTLAformer, cfg: dict):
        super().__init__()
        mc = cfg['model']
        self.stla = stla
        self.horizon = int(cfg['experiment']['horizon'])
        d = int(mc.get('latent_dim', 16))
        hid = int(mc.get('fusion_hidden', 96))
        self.max_delta = float(mc.get('max_normalized_correction', 3.0))
        self.latent_proj = nn.Sequential(nn.Linear(stla.model_dim, d), nn.GELU())
        self.horizon_embed = nn.Embedding(self.horizon, 8)
        # base, candidate Y, residual, outcome direction, similarity, coverage,
        # node observation fraction, observed historical distance, residual size
        self.candidate_scorer = nn.Sequential(nn.Linear(12 + d + 8, hid), nn.GELU(),
                                              nn.Dropout(float(mc['dropout'])), nn.Linear(hid, 1))
        # Explicitly weigh normal-error and failure-induced-error transfer.
        # 2*sigmoid(0)=1 initially reproduces the full matched historical residual.
        self.component_scale=nn.Sequential(nn.Linear(5+d+8,hid),nn.GELU(),
                    nn.Linear(hid,2))
        nn.init.zeros_(self.component_scale[-1].weight)
        nn.init.zeros_(self.component_scale[-1].bias)
        self.none_scorer = nn.Sequential(nn.Linear(2 + d + 8, hid//2), nn.GELU(),
                                        nn.Linear(hid//2, 1))
        self.control_adapter = nn.Sequential(nn.Linear(2 + d + 8, hid), nn.GELU(),
                                             nn.Linear(hid, 1))
        nn.init.constant_(self.none_scorer[-1].bias, float(mc.get('none_bias', 1.0)))
        nn.init.zeros_(self.control_adapter[-1].weight)
        nn.init.zeros_(self.control_adapter[-1].bias)
        self.register_buffer('hids', torch.arange(self.horizon), persistent=False)

    def set_train_phase(self, phase: str, last_layers: int = 2):
        self.stla.set_train_phase(phase, last_layers)

    def forward(self, x, mask, tod, dow, candidate_y=None, residual=None,
                candidate_scores=None, candidate_coverage=None, hist_distance=None,
                candidate_valid=None,normal_residual=None,teacher_base=None,mode='full'):
        base, z = self.stla(x, mask, tod, dow)
        B, P, N = base.shape
        if P != self.horizon:
            raise ValueError(f'Forecast horizon mismatch: {P} != {self.horizon}')
        ctx = self.latent_proj(z)[:, None].expand(-1, P, -1, -1)
        h = self.horizon_embed(self.hids)[None, :, None].expand(B, -1, N, -1)
        cover = mask.mean(1)[:, None, :, None].expand(B, P, N, 1)
        b = base[..., None]
        cfeat = torch.cat([b, cover, ctx, h], -1)
        if mode == 'no_retrieval':
            d = torch.tanh(self.control_adapter(cfeat).squeeze(-1)) * self.max_delta
            return {'prediction': base+d, 'base_prediction': base, 'retrieval_prediction': base,
                    'gate': torch.zeros_like(base), 'candidate_weights': None,
                    'candidate_logits': None, 'correction': d}
        if candidate_y is None or residual is None:
            raise ValueError('Real train-only retrieved future and residual required')
        if candidate_y.shape != residual.shape or candidate_y.shape[0] != B:
            raise ValueError('Evidence tensor has invalid shape')
        K = residual.shape[1]
        if candidate_y.shape[2:] != (P,N):
            raise ValueError('Candidate future shape does not match STLA output')
        if mode in ('shuffled', 'random_future') and K > 1:
            # Deterministic wrong rank-to-future pairing; independent of DataLoader batch.
            permutation = torch.roll(torch.arange(K, device=x.device), 1)
            candidate_y = candidate_y[:, permutation]
            residual = residual[:, permutation]
            if normal_residual is not None:normal_residual=normal_residual[:,permutation]
            if candidate_valid is not None:candidate_valid=candidate_valid[:,permutation]
        if mode == 'standard_future':
            residual = candidate_y - base[:,None].detach()
        if mode == 'no_evidence':
            residual = torch.zeros_like(residual)
        normal_residual=(normal_residual if normal_residual is not None else residual)
        if normal_residual.shape != residual.shape:raise ValueError('Normal residual shape mismatch')
        failure_residual=residual-normal_residual
        dteacher=base-(teacher_base.detach() if teacher_base is not None else base.detach())
        dfeat=dteacher[:,None,:,:,None].expand(B,K,P,N,1)
        compact=torch.cat([base[:,None,:,:,None].expand(B,K,P,N,1),
           residual[...,None],normal_residual[...,None],failure_residual[...,None],
           dfeat,ctx[:,None].expand(B,K,P,N,-1),h[:,None].expand(B,K,P,N,-1)],-1)
        scales=2*torch.sigmoid(self.component_scale(compact))
        # Corrective evidence depends exclusively on the retrieved historical errors.
        effective_residual=scales[...,0]*normal_residual+scales[...,1]*failure_residual
        bs = base[:, None, :, :, None].expand(-1,K,-1,-1,-1)
        cv = candidate_y[...,None]
        rs = residual[...,None]
        sim = candidate_scores[:, :, None, None, None].expand(B,K,P,N,1)
        cp = candidate_coverage[:, :, None, None, None].expand(B,K,P,N,1)
        cov = cover[:, None].expand(B,K,P,N,1)
        dis = hist_distance[:, :, None, :, None].expand(B,K,P,N,1)
        lat = ctx[:, None].expand(B,K,P,N,-1)
        hef = h[:, None].expand(B,K,P,N,-1)
        cf = torch.cat([bs,cv,rs,cv-bs,sim,cp,cov,dis,rs.abs(),normal_residual[...,None],failure_residual[...,None],dfeat,lat,hef],-1)
        logits = self.candidate_scorer(cf).squeeze(-1)
        # The original similarity influences routing, while differentiable
        # target-oriented reranking is trained by forecast/ranking loss.
        logits = logits + candidate_scores[:, :, None, None]
        none = self.none_scorer(cfeat).squeeze(-1)
        if mode == 'no_rerank':
            logits = candidate_scores[:, :, None, None].expand_as(logits)
        if candidate_valid is None:
            candidate_valid=torch.ones_like(residual,dtype=torch.bool)
        logits=logits.masked_fill(~candidate_valid.bool(),-1e4)
        effective_residual=torch.where(candidate_valid.bool(),torch.nan_to_num(effective_residual),torch.zeros_like(effective_residual))
        scores_all = torch.cat([logits, none[:,None]],dim=1)
        weights_all = torch.softmax(scores_all, dim=1)
        evidence = (weights_all[:,:K] * effective_residual).sum(1)
        correction = self.max_delta * torch.tanh(evidence / self.max_delta)
        if mode == 'no_evidence':
            correction = torch.zeros_like(correction)
        pred = base + correction
        return {'prediction': pred, 'base_prediction': base,
                'retrieval_prediction': base+residual.mean(1),
                'gate': 1-weights_all[:,-1], 'candidate_weights': weights_all,
                'candidate_logits': scores_all, 'correction': correction,
                'candidate_residuals': effective_residual,'normal_component':normal_residual,
                'failure_component':failure_residual,'teacher_discrepancy':dteacher,'candidate_available':candidate_valid.bool()}
