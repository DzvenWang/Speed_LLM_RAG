from __future__ import annotations

import torch
import torch.nn as nn


class RetrievalFusionExpert(nn.Module):
    """Target-aware learned Top-K retrieval fusion.

    Candidate discovery is performed by a learned Context embedding index.
    This module learns per-candidate/per-node reranking and a per-cell external
    support gate. The Context restoration is used as a safe endogenous prior;
    the learned gate determines how much historical memory should refine it.
    """
    def __init__(self, hidden=128, dropout=0.1):
        super().__init__()
        self.rank_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(7, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1),
        )
        # Safe initialization: uniform candidate reranking and conservative
        # external-support gate. Training must earn departure from Context.
        nn.init.zeros_(self.rank_mlp[-1].weight); nn.init.zeros_(self.rank_mlp[-1].bias)
        nn.init.zeros_(self.gate_mlp[-1].weight); nn.init.constant_(self.gate_mlp[-1].bias, -1.5)

    def forward(self, x, observed, artificial, context_restored,
                candidates, candidate_valid, coarse_similarity, a_norm):
        # x/observed/context [B,T,N]; candidates/valid [B,K,T,N]; sim [B,K]
        b,k,t,n = candidates.shape
        obs = observed.float()
        match = obs[:,None,:,:] * candidate_valid.float()
        match_count = match.sum(2).clamp_min(1.0)                           # [B,K,N]
        diff = candidates - x[:,None,:,:]
        local_mse = (diff.pow(2) * match).sum(2) / match_count              # [B,K,N]
        local_mae = (diff.abs() * match).sum(2) / match_count
        coverage = candidate_valid.float().mean(2)                          # [B,K,N]
        neighbor_mse = torch.einsum('ij,bkj->bki', a_norm, local_mse)
        neighbor_cov = torch.einsum('ij,bkj->bki', a_norm, coverage)
        sim = coarse_similarity[:,:,None].expand(-1,-1,n)
        feat = torch.stack([sim, local_mse, local_mae, coverage, neighbor_mse, neighbor_cov], dim=-1)
        logits = self.rank_mlp(feat).squeeze(-1)                            # [B,K,N]
        # discourage candidates with no support for the target node
        logits = logits.masked_fill(coverage <= 0.0, -1e4)
        weights = torch.softmax(logits, dim=1)

        # time-cell renormalisation protects against raw-invalid candidate cells.
        w = weights[:,:,None,:] * candidate_valid.float()
        den = w.sum(1).clamp_min(1e-6)
        retrieved = (w * candidates).sum(1) / den                           # [B,T,N]
        valid_any = candidate_valid.any(1)
        retrieved = torch.where(valid_any, retrieved, context_restored)

        # agreement/confidence features for per-cell external-support gating.
        mean = retrieved[:,None,:,:]
        var_num = (w * (candidates - mean).pow(2)).sum(1)
        agreement_std = torch.sqrt(var_num / den + 1e-6)
        max_weight = weights.max(1).values[:,None,:].expand(-1,t,-1)
        node_coverage = coverage.mean(1)[:,None,:].expand(-1,t,-1)
        query_obs_fraction = obs.mean(1)[:,None,:].expand(-1,t,-1)
        gate_feat = torch.stack([
            x, context_restored, retrieved, retrieved-context_restored,
            agreement_std, max_weight, query_obs_fraction * node_coverage,
        ], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_feat).squeeze(-1))
        estimate = (1.0-gate) * context_restored + gate * retrieved
        restored = torch.where(artificial.bool(), estimate, x)
        return restored, {
            'weights': weights,
            'gate': gate,
            'retrieved': retrieved,
            'agreement_std': agreement_std,
        }
