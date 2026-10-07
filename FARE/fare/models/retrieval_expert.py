from __future__ import annotations

import torch
import torch.nn as nn


class RetrievalFusionExpert(nn.Module):
    """Target-aware historical support fusion with two distinct controls.

    support_gate: context-vs-memory restoration source selection.
    control_gate: how strongly the resulting dense restoration may intervene
                  in the frozen forecasting Backbone.
    Candidate ranking is additionally supervised by offline downstream
    forecast-utility labels generated from formal-train/val truth only.
    """
    def __init__(self, hidden=128, dropout=0.1, support_gate_init_bias=-1.5,
                 control_gate_init_bias=-3.0):
        super().__init__()
        self.rank_mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        self.support_gate = nn.Sequential(
            nn.Linear(7, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1),
        )
        self.control_gate = nn.Sequential(
            nn.Linear(8, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1),
        )
        nn.init.zeros_(self.rank_mlp[-1].weight); nn.init.zeros_(self.rank_mlp[-1].bias)
        nn.init.zeros_(self.support_gate[-1].weight); nn.init.constant_(self.support_gate[-1].bias,float(support_gate_init_bias))
        nn.init.zeros_(self.control_gate[-1].weight); nn.init.constant_(self.control_gate[-1].bias,float(control_gate_init_bias))

    def forward(self, x, observed, artificial, context_restored,
                candidates, candidate_valid, coarse_similarity, a_norm):
        b,k,t,n = candidates.shape
        obs = observed.float()
        match = obs[:,None,:,:] * candidate_valid.float()
        match_count = match.sum(2).clamp_min(1.0)
        diff = candidates - x[:,None,:,:]
        local_mse = (diff.pow(2) * match).sum(2) / match_count
        local_mae = (diff.abs() * match).sum(2) / match_count
        coverage = candidate_valid.float().mean(2)
        neighbor_mse = torch.einsum('ij,bkj->bki', a_norm, local_mse)
        neighbor_cov = torch.einsum('ij,bkj->bki', a_norm, coverage)
        sim = coarse_similarity[:,:,None].expand(-1,-1,n)
        feat = torch.stack([sim, local_mse, local_mae, coverage, neighbor_mse, neighbor_cov], dim=-1)
        logits = self.rank_mlp(feat).squeeze(-1)
        logits = logits.masked_fill(coverage <= 0.0, -1e4)
        weights = torch.softmax(logits, dim=1)

        w = weights[:,:,None,:] * candidate_valid.float()
        den = w.sum(1).clamp_min(1e-6)
        retrieved = (w * candidates).sum(1) / den
        valid_any = candidate_valid.any(1)
        retrieved = torch.where(valid_any, retrieved, context_restored)

        mean = retrieved[:,None,:,:]
        var_num = (w * (candidates - mean).pow(2)).sum(1)
        agreement_std = torch.sqrt(var_num / den + 1e-6)
        max_weight = weights.max(1).values[:,None,:].expand(-1,t,-1)
        node_coverage = coverage.mean(1)[:,None,:].expand(-1,t,-1)
        query_obs_fraction = obs.mean(1)[:,None,:].expand(-1,t,-1)
        support_feat = torch.stack([
            x, context_restored, retrieved, retrieved-context_restored,
            agreement_std, max_weight, query_obs_fraction * node_coverage,
        ], dim=-1)
        sg = torch.sigmoid(self.support_gate(support_feat).squeeze(-1))
        estimate = (1.0-sg) * context_restored + sg * retrieved

        missing_node = artificial.float().mean(1)[:,None,:].expand(-1,t,-1)
        control_feat = torch.stack([
            x, context_restored, estimate, estimate-x,
            agreement_std, max_weight, query_obs_fraction, missing_node,
        ], dim=-1)
        cg = torch.sigmoid(self.control_gate(control_feat).squeeze(-1))
        controlled = x + cg * (estimate-x)
        restored = torch.where(artificial.bool(), controlled, x)
        raw_restored = torch.where(artificial.bool(), estimate, x)

        target_nodes = artificial.any(1).float()
        denom = target_nodes.sum(1,keepdim=True).clamp_min(1.0)
        query_candidate_scores = (logits * target_nodes[:,None,:]).sum(2) / denom
        return restored, {
            'weights': weights,
            'candidate_logits': logits,
            'query_candidate_scores': query_candidate_scores,
            'support_gate': sg,
            'control_gate': cg,
            'estimate': estimate,
            'raw_restored': raw_restored,
            'retrieved': retrieved,
            'agreement_std': agreement_std,
        }
