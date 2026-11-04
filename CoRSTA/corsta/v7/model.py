"""C1 failure-compatible counterfactual support and C2 horizon-conditioned transfer.

No C3. The model takes only one target's observed history and legal,
TRAIN-derived historical support responses. Query future never enters forward.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .counterfactual import StructuredCounterfactual
from .transport import SupportConditionedTransport
from .support import FEATURE_NAMES,mask_temporal_features


def safe_neighbours(values, valid=None):
    ids=torch.as_tensor(values,dtype=torch.long)
    if ids.ndim!=2:raise ValueError('Neighbor ids must be [N,K]')
    if valid is None:valid=torch.ones_like(ids,dtype=torch.bool)
    else:valid=torch.as_tensor(valid,dtype=torch.bool)
    if ids.shape!=valid.shape:raise ValueError('Neighbor validity mismatch')
    if ids.numel() and (ids.min()<0 or ids.max()>=ids.shape[0]):
        raise ValueError('Invalid graph neighbor index')
    return ids,valid

class SupportAwareTCRR(nn.Module):
    def __init__(self,model_dim,horizon,history,neighbours,road_graph,functional_valid=None,hidden=64,
                 bound=3.,top_k=12,graph_mode='dual',support_control='full',
                 horizon_graph=True,reliability_graph=True):
        super().__init__()
        self.horizon=int(horizon);self.history=int(history);self.bound=float(bound)
        self.top_k=int(top_k);self.graph_mode=graph_mode
        self.support_control=support_control
        self.horizon_graph=bool(horizon_graph);self.reliability_graph=bool(reliability_graph)
        if graph_mode not in ('dual','functional','directional'):raise ValueError(graph_mode)
        if support_control not in ('full','coverage','shuffled'):raise ValueError(support_control)
        ids,valid=safe_neighbours(neighbours,functional_valid)
        self.register_buffer('neighbours',ids)
        self.register_buffer('functional_valid',valid)
        N=ids.shape[0]
        for direction in ('upstream','downstream'):
            for key in ('ids','valid','gps_log','road_log','road_valid'):
                raw=torch.as_tensor(road_graph[f'{direction}_{key}'])
                if raw.ndim!=2 or raw.shape[0]!=N:raise ValueError(f'Road graph mismatch {direction}_{key}')
                data=(raw.long() if key=='ids' else raw.bool() if key in ('valid','road_valid') else raw.float())
                self.register_buffer(f'{direction}_{key}',data)
            if getattr(self,direction+'_ids').shape != getattr(self,direction+'_valid').shape:
                raise ValueError('Road ids/valid mismatch')
        latent=min(16,hidden)
        self.context_proj=nn.Sequential(nn.Linear(model_dim,latent),nn.GELU())
        self.c1=StructuredCounterfactual(horizon,latent,hidden,16)
        self.pair_projection=nn.Sequential(nn.Linear(len(FEATURE_NAMES),32),nn.GELU(),
                                           nn.Linear(32,32),nn.GELU())
        self.transport=SupportConditionedTransport(horizon,latent,32,hidden,bound)
        self.normal_logit=nn.Parameter(torch.tensor(.6190392084))
        # Exact initial fallback to the best Normal stage, then learn C1 activation.
        self.c1_activation=nn.Parameter(torch.tensor(0.0))
        # Target-only, horizon-conditioned spatial correction. All newly-created
        # heads are independent of previous versions' checkpoint layouts.
        self.response_encoder=nn.Sequential(
            nn.Linear(model_dim+4*horizon+2*history+3,hidden),nn.GELU(),
            nn.Linear(hidden,hidden),nn.GELU())
        att_dim=16
        self.query_proj=nn.Linear(hidden,att_dim,bias=False)
        self.key_proj=nn.Linear(hidden,att_dim,bias=False)
        self.edge_proj=nn.Linear(4,att_dim)
        self.horizon_proj=nn.Embedding(horizon,att_dim)
        self.source_reliability=nn.Sequential(nn.Linear(7,16),nn.GELU(),nn.Linear(16,1))
        self.spatial_fusion=nn.Sequential(nn.Linear(hidden*4+5+7,hidden),nn.GELU(),
                                          nn.Linear(hidden,hidden),nn.GELU())
        self.c2_delta=nn.Linear(hidden,1)
        self.c2_gate=nn.Linear(hidden,1)
        nn.init.zeros_(self.c2_delta.weight);nn.init.zeros_(self.c2_delta.bias)
        nn.init.zeros_(self.c2_gate.weight);nn.init.constant_(self.c2_gate.bias,-2.)

    def freeze_stage(self,stage):
        owners={
            'normal':('normal_logit',),
            'c1':('context_proj','c1','pair_projection','transport','c1_activation'),
            'c2':('response_encoder','query_proj','key_proj','edge_proj','horizon_proj',
                  'source_reliability','spatial_fusion','c2_delta','c2_gate'),
            'joint_c12':('context_proj','c1','pair_projection','transport','c1_activation',
                  'response_encoder','query_proj','key_proj','edge_proj','horizon_proj',
                  'source_reliability','spatial_fusion','c2_delta','c2_gate'),
        }[stage]
        for name,p in self.named_parameters():p.requires_grad_(name.split('.')[0] in owners)
        if not any(p.requires_grad for p in self.parameters()):raise RuntimeError('No trainable params')

    def _message(self,z,rel,ids,valid,gps=None,road=None,road_valid=None):
        """No full [B,P,N,K,D] materialization. Separate graph call per branch."""
        B,N,D=z.shape;P=rel.shape[1];K=ids.shape[1]
        if K==0 or not valid.any():return z.new_zeros(B,P,N,D)
        src=z[:,ids,:]  # [B,N,K,D]
        score=(self.query_proj(z)[:,:,None,:]*self.key_proj(src)).sum(-1)/(16**.5)
        # Existing edges can be weighted with fixed GPS/road descriptors.
        def edge(x):
            if x is None:return z.new_zeros(N,K)
            return x.float()
        edge_desc=torch.stack((edge(gps),edge(road),edge(road_valid),valid.float()),-1)
        projected=self.edge_proj(edge_desc)  # [N,K,16]
        h=self.horizon_proj.weight if self.horizon_graph else self.horizon_proj.weight.detach()*0
        temporal=torch.einsum('pd,nkd->pnk',h,projected)
        att=score[:,None]+temporal[None]
        if self.reliability_graph:
            r=rel[:,:,ids]   # [B,P,N,K]
            att=att+r
        valid4=valid[None,None]
        att=att.masked_fill(~valid4,-1e4)
        weights=torch.softmax(att.float(),-1)*valid4.float()
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
        return torch.einsum('bpnk,bnkd->bpnd',weights,src)

    def forward_progressive(self,base,latent,x,mask,normal,cf,*,
                            candidate_scores,candidate_coverage,candidate_valid,
                            candidate_support_raw,max_stage=2,shuffled_negatives=False):
        if max_stage not in (0,1,2):raise ValueError(max_stage)
        B,H,N=x.shape
        if H!=self.history or base.shape!=(B,self.horizon,N) or latent.shape[:2]!=(B,N):
            raise ValueError('Model input shapes mismatch')
        if normal.ndim!=4 or normal.shape!=(B,self.top_k,self.horizon,N) or cf.shape!=normal.shape:
            raise ValueError('Candidate residual [B,K,P,N] mismatch')
        if candidate_support_raw.shape!=(B,self.top_k,N,len(FEATURE_NAMES)):
            raise ValueError('Candidate PairSupport shape mismatch')
        if candidate_scores.shape!=(B,self.top_k) or candidate_coverage.shape!=(B,self.top_k):
            raise ValueError('Candidate metadata shape mismatch')
        if candidate_valid.shape!=normal.shape:raise ValueError('Candidate valid mismatch')
        p0=base.float();x=x.float()*mask.float();mask=mask.float()
        normal_simple=torch.where(candidate_valid,normal.float(),torch.zeros_like(normal.float())).sum(1) / candidate_valid.float().sum(1).clamp_min(1)
        pN=p0+torch.sigmoid(self.normal_logit)*normal_simple
        zero=torch.zeros_like(p0)
        if max_stage==0:
            return {'backbone':p0,'normal':pN,'c1':pN,'c12':pN,'delta1':zero,'delta2':zero}
        context=self.context_proj(latent.float())
        raw=candidate_support_raw.float()
        if self.support_control=='coverage':
            raw=torch.cat((raw[...,:2],torch.zeros_like(raw[...,2:])),dim=-1)
        elif self.support_control=='shuffled':
            raw=raw.roll(1,1)
        support={'embedding':self.pair_projection(raw),'compatibility_base':raw[...,2]}
        c1=self.c1(normal.float(),cf.float(),p0,context,candidate_scores.float(),
                   candidate_coverage.float(),enabled=True,shuffled_negatives=shuffled_negatives)
        tr=self.transport(p0,context,c1['normal_component'],c1['cf_increment'],
                         c1['cf_reliability'],support,candidate_scores.float(),
                         candidate_coverage.float(),candidate_valid.bool(),
                         cf_enabled=True,support_enabled=True,cf_utility=c1['cf_incremental_utility'])
        p1=pN+torch.tanh(self.c1_activation)*(p0+tr['proposed_correction']-pN)
        out={'backbone':p0,'normal':pN,'c1':p1,'c12':p1,
             'delta1':p1-pN,'delta2':zero,'base_prediction':p0,'residual_bound':self.bound,
             'candidate_residuals':tr['candidate_residuals'],
             'candidate_available':tr['candidate_available'],
             'candidate_logits':tr['candidate_logits'],
             'candidate_weights':tr['candidate_weights'],
             'candidate_transfer_gain':tr['candidate_transfer_gain'],
             'candidate_transfer_harm':tr['candidate_transfer_harm'],
             'candidate_marginal_utility_pred':tr['candidate_marginal_utility_pred'],
             'candidate_loo_proposal':tr['candidate_loo_proposal'],
             'marginal_reference_proposal':tr['marginal_reference_proposal'],
             'cf_incremental_utility':c1['cf_incremental_utility'],
             'cf_gain_logit':c1['cf_gain_logit'],
             'normal_component':c1['normal_component'],
             'cf_increment':c1['cf_increment'],
             'none_weight':tr['none_weight'],
             'candidate_coverage':candidate_coverage,
             'normal_response':tr['proposed_normal'],
             'cf_response':tr['proposed_cf'],
             'support_compatibility':tr['support_compatibility_aggregated'],
             'support_variance':tr['support_variance']}
        if 'cf_wrong_increment' in c1:
            out.update(cf_wrong_increment=c1['cf_wrong_increment'],cf_wrong_utility_pred=c1['cf_wrong_utility_pred'])
        if max_stage==1:return out
        tm=mask_temporal_features(mask)
        miss=1-tm[...,0]
        xmean=x.sum(1)/mask.sum(1).clamp_min(1)
        weighted_util=(tr['candidate_weights'][:,:self.top_k]*c1['cf_incremental_utility']).sum(1)
        node=self.response_encoder(torch.cat((latent.float(),p0.transpose(1,2),
             tr['proposed_normal'].transpose(1,2),tr['proposed_cf'].transpose(1,2),
             weighted_util.transpose(1,2),x.transpose(1,2),mask.transpose(1,2),
             miss.unsqueeze(-1),xmean.unsqueeze(-1),tm[...,0:1]),-1))
        # [B,P,N,7] strictly legal C1 reliability features; none is explicit.
        rel_features=torch.stack((
            1-tr['none_weight'],tr['normal_agreement'],tr['cf_agreement'],
            tr['cf_reliability_aggregated'],tr['support_compatibility_aggregated'],
            torch.log1p(tr['support_variance'].clamp_min(0)),
            weighted_util.tanh()),-1)
        reliability=self.source_reliability(rel_features).squeeze(-1)
        if self.graph_mode=='directional':
            func=node.new_zeros(B,self.horizon,N,node.shape[-1])
        else:
            func=self._message(node,reliability,self.neighbours,self.functional_valid)
        if self.graph_mode=='functional':
            up=down=torch.zeros_like(func)
        else:
            up=self._message(node,reliability,self.upstream_ids,self.upstream_valid,
                             self.upstream_gps_log,self.upstream_road_log,self.upstream_road_valid)
            down=self._message(node,reliability,self.downstream_ids,self.downstream_valid,
                               self.downstream_gps_log,self.downstream_road_log,self.downstream_road_valid)
        nodep=node[:,None].expand(-1,self.horizon,-1,-1)
        feat=torch.cat((nodep,func,up,down,
             tm[:,None].expand(-1,self.horizon,-1,-1),rel_features),-1)
        fusion=self.spatial_fusion(feat)
        delta=self.bound*torch.tanh(self.c2_delta(fusion).squeeze(-1)/self.bound)
        gate=torch.sigmoid(self.c2_gate(fusion).squeeze(-1))
        d2=(delta*gate).float()
        out.update(c12=p1+d2,delta2=d2,graph_gate=gate,
                   support_reliability=reliability)
        return out

@torch.no_grad()
def make_neighbours(x_reference, count=8, valid_reference=None, min_overlap=64):
    """TRAIN-only pairwise-valid functional graph with shrinkage.

    Returns (ids,valid), never ranks zero-imputed missing pairs as correlated.
    """
    if x_reference.ndim!=3:raise ValueError('Reference must be [Q,H,N]')
    Q,H,N=x_reference.shape
    if N<2:return (torch.zeros((N,1),dtype=torch.long),torch.zeros((N,1),dtype=torch.bool))
    x=torch.nan_to_num(x_reference.float()).reshape(-1,N)
    if valid_reference is None:m=torch.isfinite(x_reference).float().reshape(-1,N)
    else:
        if valid_reference.shape!=x_reference.shape:raise ValueError('Reference validity mismatch')
        m=valid_reference.float().reshape(-1,N)
    v=x*m
    cnt=m.T@m
    sx=v.T@m
    sx2=(v*v).T@m
    xy=v.T@v
    nxy=xy-sx*sx.T/cnt.clamp_min(1)
    ssx=(sx2-sx.square()/cnt.clamp_min(1)).clamp_min(0)
    sy=(ssx*ssx.T).sqrt()
    corr=nxy/sy.clamp_min(1e-6)
    corr=(corr*cnt/(cnt+min_overlap)).clamp(-1,1)
    corr= corr.masked_fill(cnt<min_overlap,-1e9)
    corr.diagonal().fill_(-1e9)
    k=min(int(count),N-1)
    values,indices=torch.topk(corr,k,1)
    valid=values>-1e8
    return indices.long().cpu(),valid.cpu()
