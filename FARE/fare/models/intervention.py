from __future__ import annotations

import torch
import torch.nn as nn


class HierarchicalInterventionController(nn.Module):
    """Forecast-intervention landscape + cell reliability controller.

    The query branch predicts downstream utility for each allowed intervention
    strength alpha.  The cell branch predicts where the raw restoration is
    locally reliable.  Their roles are deliberately separated:

      query landscape -> how much this query should intervene
      cell reliability -> where that intervention should be allocated

    The final gate is alpha * normalized reliability on artificial-missing
    cells.  Normalization keeps the query-level mean intervention close to the
    selected alpha while still allowing spatial/temporal selectivity.
    """

    def __init__(self, cell_dim: int, alpha_grid, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        alpha = torch.as_tensor(list(alpha_grid), dtype=torch.float32)
        if alpha.ndim != 1 or len(alpha) < 2:
            raise ValueError("alpha_grid must contain at least two values")
        if float(alpha.min()) < 0.0 or float(alpha.max()) > 1.0:
            raise ValueError("alpha_grid must be inside [0,1]")
        self.register_buffer("alpha_grid", alpha, persistent=True)
        self.cell_dim = int(cell_dim)
        qdim = self.cell_dim * 2 + 6
        self.query_encoder = nn.Sequential(
            nn.Linear(qdim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.landscape_head = nn.Linear(hidden, len(alpha))
        self.reliability_head = nn.Sequential(
            nn.Linear(self.cell_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, max(hidden // 2, 16)), nn.GELU(), nn.Linear(max(hidden // 2, 16), 1),
        )
        # Safe start: flat landscape includes alpha=0 and therefore selects 0;
        # reliability starts at 0.5 and is normalized before use.
        nn.init.zeros_(self.landscape_head.weight)
        nn.init.zeros_(self.landscape_head.bias)
        nn.init.zeros_(self.reliability_head[-1].weight)
        nn.init.zeros_(self.reliability_head[-1].bias)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x [B,T,N,D], mask [B,T,N]
        w = mask.float()[..., None]
        num = (x * w).sum((1, 2))
        den = w.sum((1, 2)).clamp_min(1.0)
        return num / den

    @staticmethod
    def _masked_scalar_stats(delta: torch.Tensor, artificial: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        m = artificial.float()
        den = m.flatten(1).sum(1).clamp_min(1.0)
        d = delta * m
        absd = delta.abs() * m
        mean = d.flatten(1).sum(1) / den
        abs_mean = absd.flatten(1).sum(1) / den
        sq_mean = (delta.pow(2) * m).flatten(1).sum(1) / den
        std = (sq_mean - mean.pow(2)).clamp_min(0.0).sqrt()
        max_abs = absd.flatten(1).max(1).values
        obs_frac = observed.float().flatten(1).mean(1)
        miss_frac = artificial.float().flatten(1).mean(1)
        return torch.stack([obs_frac, miss_frac, mean, abs_mean, std, max_abs], dim=-1)

    def initialize_landscape_bias(self, mean_utility_scaled: torch.Tensor) -> None:
        v = torch.as_tensor(mean_utility_scaled, dtype=self.landscape_head.bias.dtype,
                            device=self.landscape_head.bias.device).reshape(-1)
        if len(v) != len(self.alpha_grid):
            raise ValueError("mean utility curve length mismatch")
        with torch.no_grad():
            self.landscape_head.weight.zero_()
            self.landscape_head.bias.copy_(v)

    def forward(self, cell_features: torch.Tensor, artificial: torch.Tensor,
                observed: torch.Tensor, delta: torch.Tensor,
                alpha_override: torch.Tensor | None = None):
        if cell_features.ndim != 4:
            raise ValueError("cell_features must be [B,T,N,D]")
        all_pool = cell_features.mean((1, 2))
        miss_pool = self._masked_mean(cell_features, artificial)
        stats = self._masked_scalar_stats(delta, artificial, observed)
        q = torch.cat([all_pool, miss_pool, stats], dim=-1)
        hq = self.query_encoder(q)
        landscape = self.landscape_head(hq)  # utility in scaled percentage points
        idx = torch.argmax(landscape, dim=-1)
        predicted_alpha = self.alpha_grid[idx]
        alpha = predicted_alpha if alpha_override is None else alpha_override.float().reshape_as(predicted_alpha)

        reliability = torch.sigmoid(self.reliability_head(cell_features).squeeze(-1))
        m = artificial.float()
        den = m.flatten(1).sum(1).clamp_min(1.0)
        rmean = (reliability * m).flatten(1).sum(1) / den
        normalized = reliability / rmean[:, None, None].clamp_min(1e-3)
        normalized = normalized.clamp(0.10, 4.0)
        # Re-normalize after clipping so query mean stays aligned with alpha.
        nmean = (normalized * m).flatten(1).sum(1) / den
        normalized = normalized / nmean[:, None, None].clamp_min(1e-3)
        gate = (alpha[:, None, None] * normalized).clamp(0.0, 1.0)
        return {
            "landscape": landscape,
            "selected_index": idx,
            "selected_alpha": predicted_alpha,
            "applied_alpha": alpha,
            "reliability": reliability,
            "normalized_reliability": normalized,
            "gate": gate,
        }
