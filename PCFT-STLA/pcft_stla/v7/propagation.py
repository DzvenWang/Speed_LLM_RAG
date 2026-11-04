"""C2: sparse causal source changes, directional messages, explicit null source."""
from __future__ import annotations
import torch
from torch import nn

class SparsePropagation(nn.Module):
    def __init__(self,d,graph,target_idx,channels=3,physical=True,functional=False):
        super().__init__()
        self.target_idx=int(target_idx)
        self.register_buffer('edge_index',graph.edge_index.long().clone())
        self.register_buffer('edge_attr',graph.edge_attr.float().clone())
        self.register_buffer('functional_index',graph.functional_index.long().clone())
        self.register_buffer('functional_attr',graph.functional_attr.float().clone())
        # Mean and valid-pair coverage for three explicitly observed historical scales.
        self.source_mlp=nn.Sequential(nn.Linear(channels*11+d,d),nn.GELU(),nn.Linear(d,d))
        self.key=nn.Linear(d,8,bias=False)
        self.query=nn.Linear(d,8,bias=False)
        self.edge_score=nn.Sequential(nn.Linear(2*d+10,d),nn.GELU(),nn.Linear(d,1))
        self.step_score=nn.Linear(d,1)
        self.null_score=nn.Linear(d,1)
        self.physical=physical;self.functional=functional
        if functional:
            self.functional_score=nn.Sequential(nn.Linear(2*d+6,d),nn.GELU(),nn.Linear(d,1))
    def forward(self,batch,support,use_direction=True,use_null_source=True,*,return_diagnostics=False,edge_keep_mask=None,trace_edges=False):
        from .support import summarize_history
        x,mask,age=(batch[k] for k in ('x_multi','observed_mask','missing_age'))
        B,H,N,C=x.shape
        z=support['node_context']
        scales=[]
        for lag in (1,3,6):
            both=(mask[:,lag:]>0.5)&(mask[:,:-lag]>0.5)
            dx=torch.where(both,x[:,lag:]-x[:,:-lag],0.)
            counts=both.float().sum(1)
            scales += [dx.sum(1)/counts.clamp_min(1.), both.float().mean(1)]
        src_enc=self.source_mlp(torch.cat((summarize_history(x,mask,age),*scales,z),-1))
        kvec=self.key(src_enc)
        P=support['forecast_context'].shape[1]
        # Distinguish genuinely observed consecutive target samples from zero-filled differences.
        target=mask[...,self.target_idx]>0.5
        diff_pairs=(target[:,1:]&target[:,:-1])
        diff_valid=diff_pairs.any(dim=1)
        time_ids=torch.arange(1,H,device=target.device)[None,:,None]
        latest_valid_time=torch.where(diff_pairs,time_ids,-1).amax(dim=1)
        age_steps=torch.where(diff_valid,H-1-latest_valid_time,H)
        if self.physical:
            ei=self.edge_index;ea=self.edge_attr
        else:
            ei=self.edge_index[:,:0];ea=self.edge_attr[:0]
        src,dst=ei
        physical_count=src.numel()
        K=src.numel()
        if K:
            if not use_direction:
                ea=ea.clone();ea[:,6:8]=0.
            ed=ea[None].expand(B,-1,-1)
            base_score=self.edge_score(torch.cat((z[:,src],z[:,dst],ed),-1)).squeeze(-1)
            available=diff_valid[:,src].float()
            quality=support['source_quality'][:,src,0]
            values=src_enc[:,src].float()
        else:
            base_score=z.new_zeros(B,0);available=base_score;quality=base_score;values=z.new_zeros(B,0,z.shape[-1])
        # Optional functional edges represent statistical context, not asserted physical propagation.
        if self.functional and self.functional_index.numel():
            fs,fd=self.functional_index
            fe=self.functional_attr[None].expand(B,-1,-1)
            fscore=self.functional_score(torch.cat((z[:,fs],z[:,fd],fe),-1)).squeeze(-1)
            fi_available=diff_valid[:,fs].float()
            base_score=torch.cat((base_score,fscore),dim=1)
            available=torch.cat((available,fi_available),dim=1)
            quality=torch.cat((quality,support['source_quality'][:,fs,0]),dim=1)
            values=torch.cat((values,src_enc[:,fs].float()),dim=1)
            src=torch.cat((src,fs));dst=torch.cat((dst,fd))
            K=src.numel()
        # Diagnostic source interventions only change availability, not learned parameters.
        if edge_keep_mask is not None:
            keep=torch.as_tensor(edge_keep_mask,device=z.device)
            if keep.shape==(K,):keep=keep[None].expand(B,-1)
            if keep.shape!=(B,K):raise ValueError(f'edge_keep_mask shape {tuple(keep.shape)} expected [{K}] or [{B},{K}]')
            available=available*keep.float()
        trace=None
        if return_diagnostics:
            trace={'src':src.detach(),'dst':dst.detach(),
                   'physical_edge_count':physical_count,
                   'source_age_steps':age_steps.detach(),
                   'source_valid':diff_valid.detach(),
                   'source_available':available.detach(),
                   'edge_quality':quality.detach(),
                   'edge_is_functional':(torch.arange(K,device=z.device)>=physical_count),
                   'edge_forward':torch.cat((self.edge_attr[:,6]>0.5,
                       torch.zeros(K-physical_count,device=z.device,dtype=torch.bool))),
                   'edge_reverse':torch.cat((self.edge_attr[:,7]>0.5,
                       torch.zeros(K-physical_count,device=z.device,dtype=torch.bool))),
                   'null_fraction':[],'weighted_source_age':[],
                   'incoming_available_count':[],'source_mass':[],'message_l2':[]}
            if trace_edges:trace['edge_weights']=[]
        out=[];coverage=[]
        for h in range(P):
            # Scalar edge weights per horizon, values projected ONCE at node level.
            ctx=support['forecast_context'][:,h]
            # Source ranking can change with horizon/anchor-conditioned query.
            key=kvec[:,src]
            query=self.query(ctx[:,dst])
            interaction=(key*query).sum(-1)/(key.shape[-1]**0.5)
            weights=torch.sigmoid(base_score+interaction+self.step_score(ctx[:,dst]).squeeze(-1))                    *available*quality*support['support_score'][:,h,dst,0]
            num=torch.zeros(B,N,z.shape[-1],device=z.device,dtype=torch.float32)
            den=torch.zeros(B,N,1,device=z.device,dtype=torch.float32)
            if K:
                num.index_add_(1,dst,values*weights[...,None].float())
                den.index_add_(1,dst,weights[...,None].float())
            # Real null source: weighted information can be rejected altogether.
            null=(torch.exp(self.null_score(ctx).float().clamp(-6,6)) if use_null_source else torch.zeros_like(den))
            total=den+null
            msg=num/total.clamp_min(1e-6)
            frac=den/total.clamp_min(1e-6)
            if trace is not None:
                incoming=torch.zeros(B,N,1,device=z.device,dtype=torch.float32)
                age_weighted=torch.zeros(B,N,1,device=z.device,dtype=torch.float32)
                if K:
                    incoming.index_add_(1,dst,available[...,None].float())
                    age_weighted.index_add_(1,dst,weights[...,None].float()*age_steps[:,src,None].float())
                trace['incoming_available_count'].append(incoming[...,0])
                trace['weighted_source_age'].append(torch.where(den>0,age_weighted/den.clamp_min(1e-6),torch.full_like(den,H))[...,0])
                trace['null_fraction'].append(torch.where(total>0,null/total.clamp_min(1e-6),torch.ones_like(total))[...,0])
                trace['source_mass'].append(den[...,0])
                trace['message_l2'].append(msg.norm(dim=-1))
                if trace_edges:trace['edge_weights'].append(weights.detach())
            out.append(msg);coverage.append(frac)
        result={'message':torch.stack(out,1),'coverage':torch.stack(coverage,1),
                'valid_source_count':diff_valid.sum(dim=1)}
        if trace is not None:
            for name in ('null_fraction','weighted_source_age','incoming_available_count','source_mass','message_l2'):
                trace[name]=torch.stack(trace[name],dim=1)
            if trace_edges:trace['edge_weights']=torch.stack(trace['edge_weights'],dim=1)
            result['trace']=trace
        return result
