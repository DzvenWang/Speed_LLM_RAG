"""Mask_RAST-STLA v2: target-region support and train-future evolution transfer.

Unlike v1, all correction evidence comes from ACTUAL historical train futures,
not a learned second forecast head. Every future tensor is train-split only.
"""
from __future__ import annotations
import copy,math,torch
from torch import nn
from .encoders import StateEncoder
from .stla import MaskSupportSTLAformer


class MaskRASTSTLA(nn.Module):
    def __init__(self,stla:MaskSupportSTLAformer,num_nodes,history,horizon,cfg):
        super().__init__()
        ec=cfg['encoder'];mc=cfg['model'];re=cfg['retrieval']
        self.H=int(history);self.P=int(horizon)
        self.top_k=int(re['top_k']);self.region_k=int(re.get('regional_top_k',2));self.temperature=float(re['temperature'])
        self.max_delta=float(mc.get('max_normalized_correction',3.0))
        td=int(ec['temporal_dim']);sd=int(ec['spatial_dim']);qd=int(ec['query_dim']);rd=int(ec['retrieval_dim'])
        graph_k=min(int(re.get('graph_neighbors',8)),max(int(num_nodes)-1,1))
        self.query_encoder=StateEncoder(num_nodes,history,td,sd,qd,rd,int(ec['spatial_heads']),float(ec['dropout']),graph_k)
        self.memory_encoder=copy.deepcopy(self.query_encoder)
        self.stla=stla
        mid=int(mc.get('node_hidden',16));hidden=int(mc.get('fusion_hidden',96))
        self.qsummary=nn.Sequential(nn.Linear(qd,mid),nn.GELU())
        self.horizon_embed=nn.Embedding(self.P,8)
        self.scorer=nn.Sequential(nn.Linear(10+mid+8,hidden),nn.GELU(),
                                  nn.Dropout(float(mc['dropout'])),nn.Linear(hidden,1))
        self.none=nn.Sequential(nn.Linear(3+mid+8,hidden//2),nn.GELU(),nn.Linear(hidden//2,1))
        self.control=nn.Sequential(nn.Linear(3+mid+8,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.constant_(self.none[-1].bias,float(mc.get('none_bias',0.8)))
        nn.init.zeros_(self.control[-1].weight);nn.init.zeros_(self.control[-1].bias)
        self.register_buffer('hids',torch.arange(self.P),persistent=False)

    def set_graph(self,neighbours):
        self.query_encoder.set_graph(neighbours)
        self.memory_encoder.set_graph(neighbours)

    @torch.no_grad()
    def sync_region_projector(self):
        """Match regional query/memory embedding spaces before index rebuild."""
        self.memory_encoder.region_proj.load_state_dict(self.query_encoder.region_proj.state_dict())

    def set_train_phase(self,phase,last_layers=2):
        self.stla.set_train_phase(phase,last_layers)

    def forward(self,batch,memory_index,repo,mode='full'):
        x=batch['x'].float();mask=batch['mask'].float();tod=batch['tod'].long();dow=batch['dow'].long()
        B,H,N=x.shape
        q=self.query_encoder(x,mask,tod,dow)
        base,z,_=self.stla(x,mask,tod,dow,support=None,reliability=None)
        P=base.shape[1]
        n_cov=mask.mean(1);neighcov=q['neighbour_coverage']
        qsum=self.qsummary(q['node_query'])[:,None].expand(B,P,N,-1)
        h=self.horizon_embed(self.hids)[None,:,None].expand(B,P,N,-1)
        ctx=torch.cat([base[...,None],n_cov[:,None,:,None].expand(B,P,N,1),
                       neighcov[:,None,:,None].expand(B,P,N,1),qsum,h],-1)
        if mode=='no_retrieval':
            delta=torch.tanh(self.control(ctx).squeeze(-1))*self.max_delta
            return {'prediction':base+delta,'base_prediction':base,'retrieval_prediction':base,
                    'forecast_gate':torch.zeros_like(base),'candidate_weights':None,
                    'candidate_residuals':None,'correction':delta,'router':None}
        memory_index.to(x.device)
        idx_list=[];score_list=[]
        for kind in ('temporal','spatial'):
            ind,sc,_,_=memory_index.search(q[f'{kind}_key'].detach(),kind,self.top_k,
                         batch['window_start'].long(),self.H+self.P,self.temperature)
            idx_list.append(ind);score_list.append(sc)
        idx=torch.cat(idx_list,1);scores=torch.cat(score_list,1)
        G=idx.shape[1]
        gx,gm,gt,gd=repo.fetch(idx,x.device)
        gf=repo.fetch_future(idx,x.device)
        gv=repo.fetch_future_valid(idx,x.device)
        # Memory Encoder gradients exist only for the original global candidates.
        mem=self.memory_encoder(gx.reshape(B*G,H,N),gm.reshape(B*G,H,N),
                                gt.reshape(B*G,H),gd.reshape(B*G,H))
        mnode=mem['node_query'].reshape(B,G,N,-1)
        gsim=torch.nn.functional.cosine_similarity(q['node_query'][:,None],mnode,dim=-1)
        # v2.2: first-stage region-specific retrieval, NOT merely regional
        # reranking inside the global Top-K pool.
        region_idx,region_scores=memory_index.search_regions(q['region_keys'],
            self.region_k,batch['window_start'].long(),self.H+self.P)
        R,K=region_idx.shape[1:]
        rx,_,_,_=repo.fetch(region_idx,x.device)  # [B,R,K,H,N]
        rf=repo.fetch_future(region_idx,x.device)
        rv=repo.fetch_future_valid(region_idx,x.device)
        region_for_node=self.query_encoder.region_assignment
        chosen=region_for_node[None,None,None,None,:].expand(B,1,K,H,N)
        region_x=rx.gather(1,chosen).squeeze(1)
        chosen_future=region_for_node[None,None,None,None,:].expand(B,1,K,self.P,N)
        region_f=rf.gather(1,chosen_future).squeeze(1)
        region_v=rv.gather(1,chosen_future).squeeze(1)
        # Node-specific first-stage region similarity, unlike global scores.
        rs=region_scores[:,region_for_node,:].permute(0,2,1)
        xs=torch.cat([gx,region_x],1)
        futures=torch.cat([gf,region_f],1)
        future_valid=torch.cat([gv,region_v],1)
        C=G+K
        local_sim=torch.cat([gsim,rs],1)
        score_node=torch.cat([scores[:,:,None].expand(B,G,N),rs],1)
        # Transfer candidate FUTURE EVOLUTION, not an imputed historical sequence.
        # Align each candidate's historical reference to the actual most recent
        # observed query timestamp, not always the final history step.
        rev=torch.argmax(mask.flip(1),dim=1);ix=(H-1-rev).clamp_min(0)
        observed=mask.any(1)
        aligned_idx=ix[:,None,None,:].expand(B,C,1,N)
        xk_last=torch.gather(xs,2,aligned_idx).squeeze(2)
        future_delta=futures - xk_last[:,:,None,:]
        # Anchor with last OBSERVED query value; for a fully unobserved sensor,
        # use observed graph neighbours, otherwise fall back to STLA first-step.
        last=torch.gather(x,1,ix[:,None]).squeeze(1)

        neigh=self.query_encoder.neighbors
        neigh_cov=observed[:,neigh].float()
        local_last=last[:,neigh]
        local_anchor=(local_last*neigh_cov).sum(2)/neigh_cov.sum(2).clamp_min(1.)
        anchor=torch.where(observed,last,torch.where(neigh_cov.sum(2)>0,local_anchor,base[:,0].detach()))
        candidate_pred=anchor[:,None,None,:]+future_delta
        # No actual observation exists for a fully missing node. Avoid treating
        # a neighbouring sensor reading as the target node's observed anchor.
        no_obs=~observed[:,None,None,:]
        candidate_pred=torch.where(no_obs,base[:,None]+(futures-futures[:,:,:1,:]),candidate_pred)
        residual=candidate_pred-base[:,None]
        if mode in ('shuffled','random_future') and C>1:
            # Same-query candidate-rank mismatch is deterministic across batch sizes.
            residual=residual[:,torch.arange(C,device=x.device).roll(1)]
        if mode=='no_future':residual=torch.zeros_like(residual)
        d_obs=(xs-x[:,None]).abs()*mask[:,None]
        distance=d_obs.sum(2)/mask[:,None].sum(2).clamp_min(1.)
        branch=torch.cat([torch.zeros(B,self.top_k,device=x.device),torch.ones(B,self.top_k,device=x.device),torch.full((B,K),2.,device=x.device)],1)
        bs=base[:,None,:,:,None].expand(B,C,P,N,1)
        ep=candidate_pred[...,None]
        resid=residual[...,None]
        delta=future_delta[...,None]
        def exp(v):return v[:, :, None,:,None].expand(B,C,P,N,1)
        sco=score_node[:,:,None,:,None].expand(B,C,P,N,1)
        sim=exp(local_sim)
        dis=exp(distance)
        cov=n_cov[:,None,None,:,None].expand(B,C,P,N,1)
        nb=neighcov[:,None,None,:,None].expand(B,C,P,N,1)
        br=branch[:,:,None,None,None].expand(B,C,P,N,1)
        summ=qsum[:,None].expand(B,C,P,N,-1)
        hor=h[:,None].expand(B,C,P,N,-1)
        feats=torch.cat([bs,ep,resid,delta,dis,sco,sim,cov,nb,br,summ,hor],-1)
        logits=self.scorer(feats).squeeze(-1)+score_node[:,:,None,:]
        if mode=='no_regional':
            logits=score_node[:,:,None,:].expand_as(logits).clone()
            logits[:,G:]=-1e4
        if mode=='global_only':
            logits[:,G:]=-1e4
        null=self.none(ctx).squeeze(-1)
        if mode=='no_none':null=torch.full_like(null,-100.)
        logits=logits.masked_fill(~future_valid,-1e4)
        residual=torch.where(future_valid,torch.nan_to_num(residual),torch.zeros_like(residual))
        all_scores=torch.cat([logits,null[:,None]],1)
        weights=torch.softmax(all_scores,1)
        evidence=(weights[:,:C]*residual).sum(1)
        correction=self.max_delta*torch.tanh(evidence/self.max_delta)
        if mode=='no_future':correction=torch.zeros_like(correction)
        pred=base+correction
        # Three genuine competing actions: temporal, regional, no retrieval.
        router=torch.stack([weights[:,:self.top_k].sum(1),
                            weights[:,self.top_k:2*self.top_k].sum(1),
                            weights[:,G:G+K].sum(1),weights[:,-1]],-1)
        return {'prediction':pred,'base_prediction':base,
                'retrieval_prediction':base+residual.mean(1),
                'forecast_gate':1-weights[:,-1], 'candidate_weights':weights,
                'candidate_residuals':residual,'candidate_available':future_valid,'correction':correction,
                'router':router,'temporal_keys':q['temporal_key'],
                'spatial_keys':q['spatial_key']}
