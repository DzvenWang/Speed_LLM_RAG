from __future__ import annotations

from collections import defaultdict
import numpy as np

from ..cache import CacheSplit


class CausalTensorArchive:
    """Builds [N, support_days+1, H] tensors from formal-train support only."""
    def __init__(self, ctx, historical_days: int=28, day_steps: int=288):
        self.ctx=ctx; self.historical_days=int(historical_days); self.day_steps=int(day_steps)
        self.train=CacheSplit(ctx.formal_cache,"train")
        starts=np.asarray(self.train.meta.window_start,np.int64)
        # one complete ground-truth view per chronological window start; training truth is permitted support.
        first={}
        for i,s in enumerate(starts.tolist()): first.setdefault(int(s),int(i))
        self.by_slot=defaultdict(list)
        for s,i in sorted(first.items()): self.by_slot[s % self.day_steps].append((s,i))

    def support_rows(self, query_start: int):
        slot=int(query_start)%self.day_steps
        cand=[(s,i) for s,i in self.by_slot.get(slot,[]) if s+int(self.ctx.cfg['experiment']['history']) <= int(query_start)]
        if len(cand)<self.historical_days:
            raise RuntimeError(f"Structural archive has {len(cand)} same-slot causal days; need {self.historical_days} for query_start={query_start}")
        return cand[-self.historical_days:]

    def build(self, query_start: int, current_x_norm: np.ndarray, current_observed: np.ndarray):
        H,N=current_x_norm.shape; rows=self.support_rows(int(query_start)); D=self.historical_days+1
        values=np.zeros((N,D,H),np.float32); mask=np.zeros((N,D,H),bool)
        for d,(_s,i) in enumerate(rows):
            full=np.asarray(self.train.x_full[i],np.float32)
            valid=(np.asarray(self.train.history_observed[i],bool)|np.asarray(self.train.artificial_missing[i],bool))
            raw=full*self.ctx.std+self.ctx.mean
            values[:,d,:]=np.where(valid,raw,0.0).T; mask[:,d,:]=valid.T
        cur_raw=current_x_norm*self.ctx.std+self.ctx.mean
        values[:,-1,:]=np.where(current_observed,cur_raw,0.0).T; mask[:,-1,:]=current_observed.T
        return values,mask,[s for s,_ in rows]+[int(query_start)]
