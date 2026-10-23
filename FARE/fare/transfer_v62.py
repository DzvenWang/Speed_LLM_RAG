"""FARE v6.2 Experiment 1: reproducible T0/T1/T2/T3 value transfer.

T1 is the unmodified legacy PropagationTransport. T2 changes only the
source-relative evolution definition. T3 additionally uses candidate-dependent
spatial and temporal alignment. No query-only additive correction is permitted.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .transfer import PropagationTransport
from .numerics import mask_invalid_logits


def source_relative_evolution(history, observed, future, future_valid):
    """[M,T,N] -> [M,P,N]. A source without observed history is invalid.

    No artificial zero level is treated as a real observation; future values
    must be compared with the SAME source node's last real history observation.
    """
    if history.ndim != 3 or future.ndim != 3 or observed.shape != history.shape:
        raise ValueError('Expected [episodes,time,nodes] history/future')
    if history.shape[0] != future.shape[0] or history.shape[-1] != future.shape[-1]:
        raise ValueError('Episode/node dimensions disagree')
    time = torch.arange(history.shape[1],device=history.device)[None,:,None]
    last_idx=torch.where(observed.bool(),time,-1).amax(1)
    gather=history.gather(1,last_idx.clamp_min(0)[:,None,:]).squeeze(1)
    valid=future_valid.bool() & (last_idx[:,None,:]>=0)
    values=torch.where(valid, future-gather[:,None,:],0.)
    age=torch.where(last_idx>=0,history.shape[1]-1-last_idx,history.shape[1])
    return values,valid,age


class TransferEvidenceCache:
    """Adds derived source evolution to a live schema-6 EpisodeMemory in RAM.

    Does not change or rewrite the user's original OOF storage.
    """
    @staticmethod
    def attach(memory):
        if not hasattr(memory,'_v62_source_evolution'):
            v,valid,age=source_relative_evolution(memory.x,memory.mask,memory.future,memory.future_valid)
            memory._v62_source_evolution=v.contiguous()
            memory._v62_source_evolution_valid=valid.contiguous()
            memory._v62_source_age=age
        return memory


class SourceAwareTransport(PropagationTransport):
    """T2 source-aligned evolution; T3 adds candidate-specific space/lag.

    Shape convention: historical paths [B,N,K,L,H], output [B,N,K,3,P].
    No candidate future at query time; all memory episodes have already ended.
    """
    def __init__(self,d=64,horizon=12,max_lag=3,candidate_specific=False):
        super().__init__(d,horizon,max_lag)
        self.candidate_specific=bool(candidate_specific)
        self.candidate_space=nn.Sequential(nn.Linear(10,24),nn.GELU(),nn.Linear(24,1))
        self.candidate_time=nn.Sequential(nn.Linear(d+5,32),nn.GELU(),nn.Linear(32,max_lag+1))
        nn.init.zeros_(self.candidate_space[-1].weight)
        nn.init.zeros_(self.candidate_space[-1].bias)
        nn.init.zeros_(self.candidate_time[-1].weight)
        nn.init.zeros_(self.candidate_time[-1].bias)

    def forward(self,short,memory,graph,support,base_prediction,intervention=None):
        TransferEvidenceCache.attach(memory)
        ids=short['ids']; legal=short['valid'];device=ids.device
        b,n,k=ids.shape;p=base_prediction.shape[1];L=graph.path_nodes.shape[1]
        src=graph.path_nodes
        residual=memory.fetch_paths(ids,src,'residual_complete',device)
        residual_valid=memory.fetch_paths(ids,src,'residual_valid',device).bool()
        evo=memory.fetch_paths(ids,src,'_v62_source_evolution',device)
        evo_valid=memory.fetch_paths(ids,src,'_v62_source_evolution_valid',device).bool()
        emb=support['embedding'];forecast=base_prediction.permute(0,2,1)
        lag_est=support['lag_mean'][:,:,None,None]
        hop=graph.path_hops[None,:,None,:]
        direction=graph.path_type.float()[None,:,None,:]
        geometry=torch.stack([
            hop.expand(b,-1,k,-1),
            graph.path_confidence[None,:,None,:].expand(b,-1,k,-1),
            graph.path_weight[None,:,None,:].expand(b,-1,k,-1),
            direction.expand(b,-1,k,-1)/2,
            lag_est.expand(-1,-1,k,L).clamp_max(24)/24,
        ],-1)
        context=emb.mean(2)[:,:,None,None,:].expand(-1,-1,k,L,-1)
        logits=self.space(torch.cat([context,geometry],-1)).squeeze(-1)
        if self.candidate_specific:
            # Candidate-dependent alignment features, all taken from *completed*
            # historical source nodes or current query context, never query Y.
            history=memory.fetch_paths(ids,src,'x',device)
            obs=memory.fetch_paths(ids,src,'mask',device).bool()
            htime=torch.arange(history.shape[-1],device=device)[None,None,None,None,:]
            latest=torch.where(obs,htime,-1).amax(-1)
            last=history.gather(-1,latest.clamp_min(0)[...,None]).squeeze(-1)
            first=torch.where(obs,htime,history.shape[-1]).amin(-1)
            initial=history.gather(-1,first.clamp_max(history.shape[-1]-1)[...,None]).squeeze(-1)
            trend=torch.where((latest>=0)&(first<history.shape[-1]),last-initial,0.)
            cnt_r=residual_valid.float().sum(-1).clamp_min(1)
            cnt_e=evo_valid.float().sum(-1).clamp_min(1)
            r_mean=(residual*residual_valid).sum(-1)/cnt_r
            e_mean=(evo*evo_valid).sum(-1)/cnt_e
            age=torch.where(latest>=0,(history.shape[-1]-1-latest).float()/history.shape[-1],1.)
            candidate_stats=torch.stack([last,trend,r_mean,e_mean,age],-1)
            path_stats=torch.cat([geometry,candidate_stats],-1)
            logits=logits+self.candidate_space(path_stats).squeeze(-1)
        path_valid=graph.path_valid[None,:,None,:].expand(b,-1,k,-1)
        logits=logits+graph.path_weight.clamp_min(1e-7).log()[None,:,None,:]
        logits=logits+graph.path_confidence.clamp_min(.01).log()[None,:,None,:]
        logits=mask_invalid_logits(logits,~path_valid)
        missing=~path_valid.any(-1)
        logits=torch.where(missing[...,None],torch.zeros_like(logits),logits)
        w=logits.float().softmax(-1)*path_valid.float()
        w=w/w.sum(-1,keepdim=True).clamp_min(1e-7)
        def spatial(values,valid):
            ww=w[...,None]*valid.float()
            den=ww.sum(3)
            return (values*ww).sum(3)/den.clamp_min(1e-7),den>0
        res,rv=spatial(residual,residual_valid)
        evolution,ev_valid=spatial(evo,evo_valid)
        self_r=memory.fetch_node(ids,'residual_complete',device)
        self_rv=memory.fetch_node(ids,'residual_valid',device).bool()
        self_e=memory.fetch_node(ids,'_v62_source_evolution',device)
        self_ev=memory.fetch_node(ids,'_v62_source_evolution_valid',device).bool()
        query_features=torch.cat([emb,support['demand'][...,None],
                                  support['forward_coverage'][...,None]],-1)
        gate=self.self_gate(query_features).squeeze(-1).sigmoid()[:,:,None,:]
        def blend(direct,dv,neighbor,nv):
            wd=gate*dv.float();wn=(1-gate)*nv.float();den=wd+wn
            return (direct*wd+neighbor*wn)/den.clamp_min(1e-7),den>1e-7
        res,rv=blend(self_r,self_rv,res,rv)
        evolution,ev_valid=blend(self_e,self_ev,evolution,ev_valid)
        temporal_in=torch.cat([emb,support['forward_coverage'][...,None],
                               support['backward_coverage'][...,None],
                               support['demand'][...,None]],-1)
        lag_logits=self.time(temporal_in)[:,:,None,:,:].expand(-1,-1,k,-1,-1)
        if self.candidate_specific:
            # Per-episode temporal offset, no candidate-independent lag sharing.
            count=rv.float().sum(-1).clamp_min(1)
            rc=(res*rv).sum(-1)/count
            ec=(evolution*ev_valid).sum(-1)/ev_valid.float().sum(-1).clamp_min(1)
            epidx=ids.detach().cpu().clamp(0,memory.m-1)
            nodidx=torch.arange(n,dtype=torch.long)[None,:,None]
            historic_age=memory._v62_source_age[epidx,nodidx].to(device).float()/memory.x.shape[1]
            features=torch.cat([emb.mean(2)[:,:,None,:].expand(-1,-1,k,-1),
                                rc[...,None],ec[...,None],(rc-ec).abs()[...,None],
                                rv.float().mean(-1)[...,None],historic_age[...,None]],-1)
            lag_logits=lag_logits+self.candidate_time(features)[:,:,:,None,:]
        lag_probs=lag_logits.float().softmax(-1)
        def lagged(values,valid):
            out=torch.zeros_like(values);den=torch.zeros_like(values)
            for lag in range(self.max_lag+1):
                shift=values if lag==0 else F.pad(values[...,:-lag],(lag,0))
                available=valid if lag==0 else F.pad(valid[...,:-lag],(lag,0),value=False)
                wt=lag_probs[...,lag]
                out=out+shift*wt*available
                den=den+wt*available
            return out/den.clamp_min(1e-7),den>1e-7
        r,rv=lagged(res,rv)
        evolution,ev_valid=lagged(evolution,ev_valid)
        delta_base=forecast-forecast[:,:,:1]
        ctx=emb[:,:,None,:,:].expand(-1,-1,k,-1,-1)
        trend=delta_base[:,:,None,:].expand(-1,-1,k,-1)
        zr=torch.cat([ctx,r[...,None],trend[...,None],r.abs()[...,None]],-1)
        ze=torch.cat([ctx,evolution[...,None],trend[...,None],(evolution-trend)[...,None]],-1)
        r_delta=r*(1+.25*self.residual_map(zr).squeeze(-1).tanh())
        e_delta=(evolution-trend)*(1+.25*self.evolution_map(ze).squeeze(-1).tanh())
        v_r=legal[...,None]&rv
        v_e=legal[...,None]&ev_valid
        failure=memory.fetch_node(ids,'failure_delta',device)
        failure_valid=memory.fetch_node(ids,'failure_valid',device).bool()&legal[...,None]
        delta=torch.stack([r_delta,e_delta,failure],3)
        avail=torch.stack([v_r,v_e,failure_valid],3)
        agreement=1-(r_delta-e_delta).abs()/(1+r_delta.abs()+e_delta.abs())
        comp_in=torch.cat([ctx,(rv.float()+ev_valid.float())[...,None]/2,
                           agreement[...,None],support['demand'][:,:,None,:,None].expand(-1,-1,k,-1,-1),
                           (1-support['forward_coverage'])[:,:,None,:,None].expand(-1,-1,k,-1,-1)],-1)
        compat=self.confidence(comp_in).squeeze(-1).sigmoid()
        return {'deltas':delta,'available':avail,'agreement':agreement,
                'compatibility':compat,'alignment_confidence':lag_probs[...,0],
                'expected_lag':(lag_probs*torch.arange(self.max_lag+1,device=device)).sum(-1),
                'path_weight':w,'raw_residual':r,'evolution':evolution,'transport_valid':v_r|v_e}


class Experiment1Transfer(nn.Module):
    """Same candidate aggregation and gain parameterization across T0--T3.

    The only trainable values in this module act multiplicatively on historical
    evidence; there is no current-query-only additive prediction shortcut.
    """
    def __init__(self,variant,d=64,horizon=12,max_lag=3,temperature=.25):
        super().__init__()
        if variant not in ('direct','legacy','source_aligned','candidate_aligned'):
            raise ValueError(f'Unknown Transfer variant {variant}')
        self.variant=variant;self.horizon=horizon;self.temperature=float(temperature)
        self.transport=(PropagationTransport(d,horizon,max_lag) if variant=='legacy' else
                        SourceAwareTransport(d,horizon,max_lag,candidate_specific=variant=='candidate_aligned')
                        if variant!='direct' else None)
        self.channel_logits=nn.Parameter(torch.tensor([1.,0.,-.5]))
        self.logit_strength=nn.Parameter(torch.full((horizon,),-1.5))

    def forward(self,short,memory,graph,support,base_prediction,reference_prediction=None):
        ids=short['ids'];legal=short['valid'].bool();dev=ids.device
        if self.variant=='direct':
            residual=memory.fetch_node(ids,'residual_complete',dev)
            avail=memory.fetch_node(ids,'residual_valid',dev).bool()&legal[...,None]
            candidate=torch.where(avail,residual,0.)
        else:
            proposal=self.transport(short,memory,graph,support,base_prediction)
            v=proposal['available'].bool()&legal[...,None,None]
            channels=self.channel_logits.float().softmax(0)[None,None,None,:,None]*v
            den=channels.sum(3)
            candidate=(proposal['deltas']*channels).sum(3)/den.clamp_min(1e-8)
            avail=v.any(3)
            candidate=torch.where(avail,candidate,0.)
        scores=short['scores'].float().clamp(-100,100)
        logits=(scores/self.temperature).masked_fill(~legal,-1e7)
        nolegal=~legal.any(-1)
        logits=torch.where(nolegal[...,None],torch.zeros_like(logits),logits)
        # Per-horizon normalisation includes only histories with valid evidence.
        weights=logits[...,None].expand_as(candidate).masked_fill(~avail,-1e7).softmax(2)
        weights=weights*avail
        weights=weights/weights.sum(2,keepdim=True).clamp_min(1e-8)
        transferred=(candidate*weights).sum(2)
        learned_alpha=self.logit_strength.sigmoid()[None,None,:]
        # Transport uses frozen-backbone trend; correction is applied to the
        # *fixed Support-only reference*, NEVER silently to raw STLA output.
        reference=base_prediction if reference_prediction is None else reference_prediction
        pred=reference+learned_alpha.permute(0,2,1)*transferred.permute(0,2,1)
        return {'prediction':pred,'candidate_deltas':candidate,'candidate_valid':avail,
                'candidate_weights':weights,'transfer_delta':transferred,
                'learned_alpha':learned_alpha,'support_prediction':reference}
