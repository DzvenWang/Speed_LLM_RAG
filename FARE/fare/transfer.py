"""Sparse directional space×time historical Residual/Evolution transport.

Residual and Evolution VALUE branches are information isolated.  Both use shared
query-state and static propagation-path context.  Estimated lag is a learned
relative alignment in 5-minute steps, not a measured traffic shockwave speed.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .numerics import mask_invalid_logits


class PropagationTransport(nn.Module):
    def __init__(self,d=64,horizon=12,max_lag=3):
        super().__init__();self.horizon=horizon;self.max_lag=int(max_lag)
        self.space=nn.Sequential(nn.Linear(d+5,d),nn.GELU(),nn.Linear(d,1))
        self.time=nn.Sequential(nn.Linear(d+3,d),nn.GELU(),nn.Linear(d,max_lag+1))
        self.residual_map=nn.Sequential(nn.Linear(d+3,d),nn.GELU(),nn.Linear(d,1))
        self.evolution_map=nn.Sequential(nn.Linear(d+3,d),nn.GELU(),nn.Linear(d,1))
        self.confidence=nn.Sequential(nn.Linear(d+4,d),nn.GELU(),nn.Linear(d,1))
        self.self_gate=nn.Sequential(nn.Linear(d+2,d),nn.GELU(),nn.Linear(d,1))

    def forward(self,short,memory,graph,support,base_prediction,intervention=None):
        ids=short['ids'];legal=short['valid'];device=ids.device
        b,n,k=ids.shape;p=base_prediction.shape[1];L=graph.path_nodes.shape[1]
        src=graph.path_nodes
        # Historical values are gathered only for the bounded sparse path set.
        residual=memory.fetch_paths(ids,src,'residual' if intervention=='ordinary_residual' else 'residual_complete',device)
        residual_valid=memory.fetch_paths(ids,src,'residual_valid',device).bool()
        future=memory.fetch_paths(ids,src,'future',device)
        future_valid=memory.fetch_paths(ids,src,'future_valid',device).bool()
        # Query-independent historic level for Evolution: last actually observed history.
        hist=memory.fetch_node(ids,'x',device)
        hvalid=memory.fetch_node(ids,'mask',device).bool()
        t=torch.arange(hist.shape[-1],device=device)
        last=torch.where(hvalid,t[None,None,None,:],-1).amax(-1)
        level=hist.gather(-1,last.clamp_min(0)[...,None]).squeeze(-1)
        level=torch.where(last>=0,level,0)
        emb=support['embedding'];forecast=base_prediction.permute(0,2,1)
        lag_est=support['lag_mean'][:,:,None,None]
        hop=graph.path_hops[None,:,None,:]
        direction=graph.path_type.float()[None,:,None,:]
        geometry=torch.stack([hop.expand(b,-1,k,-1),
                              graph.path_confidence[None,:,None,:].expand(b,-1,k,-1),
                              graph.path_weight[None,:,None,:].expand(b,-1,k,-1),
                              direction.expand(b,-1,k,-1)/2,
                              lag_est.expand(-1,-1,k,L).clamp_max(24)/24],-1)
        # Low-dimensional per-node path scoring; [B,N,K,L,D] not [B,N,K,P,L,D].
        context=emb.mean(2)[:,:,None,None,:].expand(-1,-1,k,L,-1)
        logits=self.space(torch.cat([context,geometry],-1)).squeeze(-1)
        conf=graph.path_confidence[None,:,None,:]
        log_weight=graph.path_weight.clamp_min(1e-7).log()[None,:,None,:]
        path_valid=graph.path_valid[None,:,None,:].expand(b,-1,k,-1)
        if intervention=='no_direction':
            log_weight=log_weight*0
            logits=logits*0
        logits=logits+log_weight+conf.clamp_min(.01).log()
        # Softmax masked logits computed in FP32: -1e7 overflows FP16.
        logits=mask_invalid_logits(logits,~path_valid)
        all_invalid=~path_valid.any(-1)
        logits=torch.where(all_invalid[...,None],torch.zeros_like(logits),logits)
        w=logits.softmax(-1)*path_valid.float()
        w=w/w.sum(-1,keepdim=True).clamp_min(1e-7)
        # Weighted valid support separately for each VALUE branch; do not mix values.
        def spatial(values,valid):
            ww=w[...,None]*valid.float()
            numerator=(values*ww).sum(3)
            den=ww.sum(3)
            return numerator/den.clamp_min(1e-7),den>0
        res,rv=spatial(residual,residual_valid)
        fut,fv=spatial(future,future_valid)
        # Preserve the historical target-node Episode evidence. A node with no
        # incoming physical path still has a legitimate same-node residual/future.
        self_r=memory.fetch_node(ids,'residual' if intervention=='ordinary_residual' else 'residual_complete',device)
        self_rv=memory.fetch_node(ids,'residual_valid',device).bool()
        self_y=memory.fetch_node(ids,'future',device)
        self_yv=memory.fetch_node(ids,'future_valid',device).bool()
        query_features=torch.cat([emb,support['demand'][...,None],
                                  support['forward_coverage'][...,None]],-1)
        gate=self.self_gate(query_features).squeeze(-1).sigmoid()[:,:,None,:]
        def blend(direct,dv,neighbor,nv):
            w_direct=gate*dv.float()
            w_neigh=(1-gate)*nv.float()
            den=w_direct+w_neigh
            return (direct*w_direct+neighbor*w_neigh)/den.clamp_min(1e-7),den>1e-7
        res,rv=blend(self_r,self_rv,res,rv)
        fut,fv=blend(self_y,self_yv,fut,fv)
        ev=fut-level[...,None]
        # Candidate-independent temporal alignment weights; bounded non-wrapping lags.
        fwd=support['forward_coverage'];bwd=support['backward_coverage']
        temporal_in=torch.cat([emb,fwd[...,None],bwd[...,None],support['demand'][...,None]],-1)
        lag_logits=self.time(temporal_in)
        if intervention=='no_propagation_alignment':
            lag_probs=torch.zeros_like(lag_logits);lag_probs[...,0]=1
        else:lag_probs=lag_logits.float().softmax(-1)
        def lagged(values,valid):
            out=torch.zeros_like(values);den=torch.zeros_like(values)
            for lag in range(self.max_lag+1):
                if lag==0:
                    shift=values;available=valid
                else:
                    shift=F.pad(values[...,:-lag],(lag,0))
                    available=F.pad(valid[...,:-lag],(lag,0),value=False)
                weight=lag_probs[:,:,None,:,lag]
                out=out+shift*weight*available
                den=den+weight*available
            return out/den.clamp_min(1e-7),den>1e-7
        r,rv=lagged(res,rv)
        evolution,ev_valid=lagged(ev,fv)
        delta_base=forecast-forecast[:,:,:1]
        ctx=emb[:,:,None,:,:].expand(-1,-1,k,-1,-1)
        trend=delta_base[:,:,None,:].expand(-1,-1,k,-1)
        # No cross-Evidence VALUE reads in either mapper.
        zr=torch.cat([ctx,r[...,None],trend[...,None],r.abs()[...,None]],-1)
        ze=torch.cat([ctx,evolution[...,None],trend[...,None],
                      (evolution-trend)[...,None]],-1)
        r_delta=r*(1+0.25*self.residual_map(zr).squeeze(-1).tanh())
        e_delta=(evolution-trend)*(1+0.25*self.evolution_map(ze).squeeze(-1).tanh())
        v_r=legal[...,None]&rv
        v_e=legal[...,None]&ev_valid&(last[...,None]>=0)
        if intervention in ('no_residual','no_residual_value','no_residual_information'):v_r=v_r*False
        if intervention in ('no_evolution','no_evolution_value','no_evolution_information'):v_e=v_e*False
        if intervention in ('no_residual_information',):r_delta=r_delta*0
        if intervention in ('no_evolution_information',):e_delta=e_delta*0
        failure=memory.fetch_node(ids,'failure_delta',device)
        failure_valid=memory.fetch_node(ids,'failure_valid',device).bool() & legal[...,None]
        # Paired OOF predictions isolate the support that the sensing failure removed.
        delta=torch.stack([r_delta,e_delta,failure],3)
        avail=torch.stack([v_r,v_e,failure_valid],3)
        agreement=1-(r_delta-e_delta).abs()/(1+r_delta.abs()+e_delta.abs())
        support_weight=w.sum(-1).clamp(0,1)
        # Compatibility combines path confidence, evidence validity, directional support.
        comp_in=torch.cat([ctx,(rv.float()+ev_valid.float())[...,None]/2,
                           agreement[...,None],support['demand'][:,:,None,:,None].expand(-1,-1,k,-1,-1),
                           (1-support['forward_coverage'])[:,:,None,:,None].expand(-1,-1,k,-1,-1)],-1)
        compat=self.confidence(comp_in).squeeze(-1).sigmoid()
        return {'deltas':delta,'available':avail,'agreement':agreement,
                'compatibility':compat,'alignment_confidence':lag_probs[...,0],
                'expected_lag':(lag_probs*torch.arange(self.max_lag+1,device=device)).sum(-1),
                'path_weight':w,'raw_residual':r,'evolution':evolution,
                'transport_valid':v_r|v_e}
