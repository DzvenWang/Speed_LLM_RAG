"""Pairwise observable support for query-conditioned residual transport.

Never compares unobserved query values or source-invalid positions. All feature
statistics are based on the historical input window, with no future labels.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class PairSupport(nn.Module):
    FEATURE_NAMES=('query_coverage','source_coverage','overlap','intervention',
                   'recent_overlap','recent_intervention','observed_agreement',
                   'has_overlap','query_missing_age','query_longest_run',
                   'query_neighbour_coverage','source_neighbour_coverage',
                   'overlap_neighbour','query_recent_coverage')
    def __init__(self, hidden=32):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(len(self.FEATURE_NAMES),hidden),nn.GELU(),
                               nn.Linear(hidden,hidden),nn.GELU())
        self.out_dim=hidden

    @staticmethod
    def _neighbour_mean(values,neighbours):
        # values [B,K?,N] and neighbours [N,L]; do not hardcode node IDs.
        if neighbours is None or neighbours.numel()==0 or neighbours.shape[0]!=values.shape[-1]:
            return values
        ids=neighbours.to(device=values.device,dtype=torch.long)
        return values[...,ids].mean(-1)

    def forward(self,x,mask,candidate_x,source_valid,neighbours=None,control="correct"):
        # query x/mask [B,H,N]; candidates/source_valid [B,K,H,N]
        if candidate_x is None or source_valid is None:
            raise ValueError('v5 requires candidate histories and original source validity')
        B,H,N=mask.shape
        if candidate_x.shape != source_valid.shape or candidate_x.shape[0]!=B or candidate_x.shape[2:]!=(H,N):
            raise ValueError('Invalid paired support dimensions')
        K=candidate_x.shape[1]
        q=mask[:,None].float().clamp(0,1)
        s=source_valid.float().clamp(0,1)
        overlap=q*s
        intervention=(1-q)*s
        count=overlap.sum(2)
        observed_agreement=((candidate_x.float()-x[:,None].float()).abs()*overlap).sum(2)/count.clamp_min(1)
        observed_agreement=torch.exp(-observed_agreement.clamp(0,15))
        observed_agreement=torch.where(count>0,observed_agreement,torch.zeros_like(observed_agreement))
        qc=q.mean(2).expand(B,K,N)
        sc=s.mean(2)
        oc=overlap.mean(2)
        ic=intervention.mean(2)
        rec=max(1,H//4)
        recent_oc=overlap[:,:,-rec:].mean(2)
        recent_ic=intervention[:,:,-rec:].mean(2)
        qrecent=q[:,:,-rec:].mean(2).expand(B,K,N)
        # At a node: number of steps elapsed since last observed sample, scaled.
        missing=(1-q[:,0]).float()
        age=torch.zeros((B,N),device=x.device,dtype=torch.float32)
        run=torch.zeros_like(age)
        longest=torch.zeros_like(age)
        for t in range(H):
            run=(run+1)*missing[:,t]
            longest=torch.maximum(longest,run)
            age=run
        age=(age/H)[:,None].expand(B,K,N)
        longest=(longest/H)[:,None].expand(B,K,N)
        qneigh=self._neighbour_mean(qc,neighbours)
        sneigh=self._neighbour_mean(sc,neighbours)
        oneigh=self._neighbour_mean(oc,neighbours)
        raw=torch.stack([qc,sc,oc,ic,recent_oc,recent_ic,observed_agreement,
                         (count>0).float(),age,longest,qneigh,sneigh,oneigh,qrecent],dim=-1)
        raw=torch.nan_to_num(raw,nan=0.0,posinf=0.0,neginf=0.0)
        if control=='coverage':
            # Retain only query/source aggregate coverage, no detailed pairing.
            raw=torch.cat((raw[...,:2],torch.zeros_like(raw[...,2:])),dim=-1)
        elif control=='shuffled':
            # Break query-candidate support correspondence while preserving the
            # support feature marginals. This is diagnostic, never a test label.
            if K>1:raw=raw.roll(1,dims=1)
            elif B>1:raw=raw.roll(1,dims=0)
            else:raw=raw.roll(1,dims=2)
        elif control!='correct':
            raise ValueError(f'Unknown support control {control}')
        return {'raw':raw,'embedding':self.net(raw),
                'overlap_count':count,'has_overlap':count>0,
                'compatibility_base':raw[...,2], 'intervention_exposure':raw[...,3]}
