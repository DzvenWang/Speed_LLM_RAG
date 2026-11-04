from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from .graph import DirectedGraph


def load_original_stla(repo: Path, checkpoint: Path, device='cpu'):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck.get('model')!='STLAformer':raise RuntimeError('Not an official STLAformer Backbone checkpoint')
    mc=ck['model_config']; src=repo/'Backbone/vendor/STLAformer/model/STLformer.py'
    if not src.exists():raise FileNotFoundError(f'Official vendor source missing: {src}. Run Backbone/scripts/setup_official_models.sh')
    if str(src.parent) not in sys.path:sys.path.insert(0,str(src.parent))
    spec=importlib.util.spec_from_file_location('pcft5_original_stla',src)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    base=mod.STLformer(num_nodes=int(ck['num_nodes']),in_steps=12,out_steps=12,
        steps_per_week=int(mc.get('steps_per_week',2016)),input_dim=int(mc.get('input_dim',2)),output_dim=1,
        input_embedding_dim=int(mc.get('input_embedding_dim',32)),tow_embedding_dim=int(mc.get('tow_embedding_dim',32)),
        feed_forward_dim=int(mc.get('feed_forward_dim',256)),num_heads=int(mc.get('num_heads',4)),
        num_layers=int(mc.get('num_layers',6)),dropout=float(mc.get('dropout',0.3)))
    base.load_state_dict(ck['state_dict'],strict=True)
    return base.to(device),ck


class STLAEncoderBridge(nn.Module):
    def __init__(self,base,n_channels=3):
        super().__init__()
        self.base=base;d=int(base.model_dim)
        # Side input: normalized 3-channel history, observed mask and missing age.
        self.side=nn.Sequential(nn.Linear(3*n_channels,d),nn.GELU(),nn.Linear(d,d))
        nn.init.zeros_(self.side[-1].weight);nn.init.zeros_(self.side[-1].bias)
        self.mask_proj=nn.Sequential(nn.Linear(4,d),nn.GELU(),nn.Linear(d,d))
        nn.init.zeros_(self.mask_proj[-1].weight);nn.init.zeros_(self.mask_proj[-1].bias)
    def forward(self,x_target,x_multi,mask,age,tod,dow):
        B,H,N=x_target.shape
        if N!=int(self.base.num_nodes) or H!=int(self.base.in_steps):raise ValueError('STLA checkpoint node/history mismatch')
        slot=dow.long()*288+tod.long()
        tow=torch.nextafter(slot.float()/float(self.base.steps_per_week),torch.full_like(slot.float(),float('inf')))
        tow=tow[:,:,None].expand(B,H,N)
        raw=torch.stack((x_target,tow),dim=-1)
        if self.base.input_dim>2:raise ValueError('Unexpected original STLA input_dim > 2')
        z=self.base.input_proj(raw[...,:self.base.input_dim])
        features=[z]
        if self.base.tow_embedding_dim>0:
            idx=(tow*self.base.steps_per_week).long()
            # This mirrors the current v4 wrapper precisely.
            features.append(self.base.tow_embedding(idx))
        z=torch.cat(features,-1)
        target_mask=mask[...,self._target_index]
        nvalid=target_mask.mean(1,keepdim=True).expand(-1,H,-1)
        tvalid=target_mask.mean(2,keepdim=True).expand(-1,-1,N)
        z=z+self.mask_proj(torch.stack([target_mask,nvalid,tvalid,age[...,self._target_index]],-1))
        z=z+self.side(torch.cat([x_multi,mask,age],dim=-1))
        node_emb=self.base.node_position_encoding(z)
        time_emb=self.base.time_position_encoding(z)
        bx=self.base.MLP(z)
        h=z-bx
        for layer in self.base.attn_layers_st:h=layer(h,node_emb,time_emb)
        h=h+bx
        return h
    _target_index=0
    def set_target_index(self,i):self._target_index=int(i)
    def load_mask_adapter(self,path,dataset,target,seed,backbone_mode,backbone_sha):
        pkg=torch.load(path,map_location='cpu',weights_only=False)
        if pkg.get('schema')!='pcft_shared_stla_adapter_v1':raise ValueError('Invalid shared adapter schema')
        expected={'dataset':dataset,'target':target,'seed':int(seed),'backbone_mode':backbone_mode,'backbone_sha256':backbone_sha}
        for key,value in expected.items():
            if pkg.get(key)!=value:raise ValueError(f'Shared adapter {key} mismatch {pkg.get(key)} vs {value}')
        self.mask_proj.load_state_dict(pkg['mask_proj'],strict=True)


class EdgeAwareGraphAggregate(nn.Module):
    """Sparse edge-conditioned message aggregation; FP32 scatter under autocast."""
    def __init__(self,d,edge_dim):
        super().__init__()
        self.msg=nn.Linear(d,d,bias=False)
        self.gate=nn.Sequential(nn.Linear(d*2+edge_dim,d),nn.GELU(),nn.Linear(d,1))
        self.out=nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.Linear(d,d))
    def forward(self,z,ei,ea):
        B,N,D=z.shape
        src,dst=ei
        if not src.numel():return z.new_zeros(z.shape)
        features=torch.cat([z[:,src],z[:,dst],ea[None].expand(B,-1,-1)],-1)
        weight=torch.sigmoid(self.gate(features).float())
        msg=self.msg(z[:,src]).float()*weight
        out=torch.zeros((B,N,D),device=z.device,dtype=torch.float32)
        den=torch.zeros((B,N,1),device=z.device,dtype=torch.float32)
        out.index_add_(1,dst,msg)
        den.index_add_(1,dst,weight)
        agg=out/den.clamp_min(1e-6)
        return self.out(torch.cat([z.float(),agg],-1))


class OCPR(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.temporal=nn.Conv1d(d,d,kernel_size=3,padding=1)
        self.graph=EdgeAwareGraphAggregate(d,10)
        self.functional=EdgeAwareGraphAggregate(d,6)
        self.delta=nn.Sequential(nn.Linear(6,d),nn.GELU(),nn.Linear(d,d))
        self.context_gate=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,1))
        self.combine=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU(),nn.Linear(d,d))
        self.reliability=nn.Sequential(nn.Linear(d+2,32),nn.GELU(),nn.Linear(32,1),nn.Sigmoid())
    def forward(self,h,x,mask,age,pe,pa,fe,fa,intervention=None):
        B,H,N,D=h.shape
        temp=self.temporal(h.permute(0,2,3,1).reshape(B*N,D,H)).mean(-1).reshape(B,N,D)
        v=mask[:,1:]*mask[:,:-1]
        delta=(x[:,1:]-x[:,:-1])*v
        stats=torch.cat([delta.mean(1),v.mean(1)],-1)
        physical=self.graph(h[:,-1],pe,pa)
        functional=self.functional(h[:,-1],fe,fa) if intervention!='remove_functional' else torch.zeros_like(physical)
        fgate=torch.sigmoid(self.context_gate(torch.cat([physical,functional],-1)))
        z=self.combine(h[:,-1]+temp+self.delta(stats)+physical+fgate*functional)
        coverage=mask.mean((1,3)).unsqueeze(-1)
        freshness=(1-age[:,-1].mean(-1,keepdim=True))
        rel=self.reliability(torch.cat([z,coverage,freshness],-1))
        return {'state':z,'reliability':rel,'physical_context_norm':physical.detach().norm(dim=-1).mean(),
                'functional_context_norm':functional.detach().norm(dim=-1).mean()}


class LaggedEvidenceReachability(nn.Module):
    """Forecast-step indexed, causal, physical-edge support; no future targets."""
    def __init__(self,max_lag=6,decay=.94):
        super().__init__()
        self.max_lag=max_lag;self.decay=float(decay)
        self.edge_lag=nn.Linear(10,max_lag)
    def forward(self,observed_mask,reliability,ei,ea,horizon):
        B,H,N,C=observed_mask.shape;src,dst=ei
        lag=torch.softmax(self.edge_lag(ea).float(),dim=-1)
        # Historic source support is observable only where sensor values exist.
        bank=[(observed_mask[:,t].any(-1).float()*reliability.squeeze(-1).float()) for t in range(H)]
        all_h=[]
        for h in range(horizon):
            num=torch.zeros((B,N),device=observed_mask.device,dtype=torch.float32)
            den=torch.zeros((B,N),device=observed_mask.device,dtype=torch.float32)
            for k in range(self.max_lag):
                t=len(bank)-1-k
                if t<0:continue
                weights=lag[:,k].unsqueeze(0).expand(B,-1)
                num.index_add_(1,dst,bank[t][:,src]*weights)
                den.index_add_(1,dst,weights)
            incoming=(num/den.clamp_min(1e-6)).clamp(0,1)
            nxt=self.decay*torch.maximum(bank[-1],incoming)
            nxt=nxt.clamp(0,1)
            bank.append(nxt)
            all_h.append(nxt)
        return torch.stack(all_h,dim=1).unsqueeze(-1)

class HorizonPrior(nn.Module):
    def __init__(self,d,history=12,horizon=12):
        super().__init__();self.history=history;self.horizon=horizon
        self.project=nn.Sequential(nn.Linear(history*d,d*2),nn.GELU(),nn.Linear(d*2,horizon*d))
        self.step_embed=nn.Embedding(horizon,d)
        self.c1=nn.Sequential(nn.Linear(d+2,d),nn.GELU(),nn.Linear(d,d))
        self.c1_scale=nn.Parameter(torch.tensor(-2.0))
    def forward(self,h,c1=None,reach=None):
        B,H,N,D=h.shape
        prior=self.project(h.transpose(1,2).reshape(B,N,H*D)).reshape(B,N,self.horizon,D).transpose(1,2)
        prior=prior+self.step_embed.weight[None,:,None,:]
        if c1 is not None:
            if reach is None:raise ValueError('Reachability must exist with C1')
            z=c1['state'][:,None].expand(-1,self.horizon,-1,-1)
            reliability=c1['reliability'][:,None].expand(-1,self.horizon,-1,-1)
            prior=prior+torch.sigmoid(self.c1_scale)*self.c1(torch.cat([z,reliability,reach],-1))
        return prior


class DynamicLagTransport(nn.Module):
    def __init__(self,d,edge_dim=10,max_lag=6,horizon=12):
        super().__init__();self.max_lag=max_lag;self.horizon=horizon
        self.edge_net=nn.Sequential(nn.Linear(d*2+edge_dim+d,d),nn.GELU(),nn.Linear(d,max_lag+1))
        self.horizon_embed=nn.Embedding(horizon,d)
        self.msg=nn.Linear(d,d,bias=False)
        self.out=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.Tanh())
        self.scale=nn.Parameter(torch.tensor(-2.0))
    def kernel(self,state,ei,ea,step,intervention=None):
        src,dst=ei;h=self.horizon_embed.weight[step].view(1,1,-1)
        inp=torch.cat([state[:,src],state[:,dst],ea[None].expand(state.shape[0],-1,-1),h.expand(state.shape[0],src.numel(),-1)],-1)
        raw=self.edge_net(inp).float()
        lag=torch.softmax(raw[...,:self.max_lag],dim=-1)
        edge=torch.sigmoid(raw[...,-1])
        if intervention=='uniform_lag':lag=torch.ones_like(lag)/self.max_lag
        elif intervention=='reverse_lag':lag=lag.flip(-1)
        return lag,edge
    def forward(self,bank,support,edge_index,edge_attr,state,step,intervention=None,coupled=False):
        B,T,N,D=bank.shape;src,dst=edge_index;E=src.numel()
        if T<self.max_lag+1:raise ValueError('History shorter than max_lag+1')
        if intervention=='no_transport':
            return bank.new_zeros((B,N,D)),support[:,-1].clamp(0,1),{'transport_norm':bank.new_zeros(()),'lag_entropy':bank.new_zeros(())}
        if intervention=='reverse_direction':
            src,dst=dst,src
            edge_attr=edge_attr.clone()
            edge_attr[:,6],edge_attr[:,7]=edge_attr[:,7].clone(),edge_attr[:,6].clone()
            kernel_index=torch.stack((src,dst),dim=0)
        else:kernel_index=edge_index
        lag,edge=self.kernel(state,kernel_index,edge_attr,step,intervention)
        # Reductions stay float32 under CUDA autocast to avoid unstable support/denominator.
        num=torch.zeros((B,N,D),device=bank.device,dtype=torch.float32)
        den=torch.zeros((B,N,1),device=bank.device,dtype=torch.float32)
        evnum=torch.zeros((B,N,1),device=bank.device,dtype=torch.float32)
        for k in range(self.max_lag):
            t=T-k-1
            diff=(bank[:,t,src]-bank[:,t-1,src]).float()
            source_support=support[:,t,src].float().clamp(0,1)
            # When not coupled use strictly observation-derived historical/future support.
            # C1 reliability is applied only in the coupled stage via support bank initialization.
            weight=edge*lag[...,k]
            numerator_weight=weight*source_support
            msg=self.msg(diff.to(self.msg.weight.dtype)).float()*numerator_weight[:,:,None]
            num.index_add_(1,dst,msg)
            den.index_add_(1,dst,weight[:,:,None])
            evnum.index_add_(1,dst,numerator_weight[:,:,None])
        signal=(num/den.clamp_min(1e-6)).to(bank.dtype)
        arriving=(evnum/den.clamp_min(1e-6)).clamp(0,1)
        # Critical invariant: a missing observation source cannot create
        # a propagation update through learned projection biases.
        updated=self.out(signal)*torch.sigmoid(self.scale)*arriving.to(signal.dtype)
        # A bounded update; decay applied even to self-support when observation is absent.
        nxt=(0.92*torch.maximum(support[:,-1].float().unsqueeze(-1),arriving)).clamp(0,1).squeeze(-1)
        entropy=-(lag*lag.clamp_min(1e-8).log()).sum(-1).mean().detach()
        return updated,nxt,{'transport_norm':updated.detach().norm(dim=-1).mean(),'lag_entropy':entropy}


class LatentDynamics(nn.Module):
    def __init__(self,d,horizon=12,max_lag=6):
        super().__init__();self.horizon=horizon
        self.local=nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,d),nn.Tanh())
        self.local_scale=nn.Parameter(torch.tensor(-2.0))
        self.trans=DynamicLagTransport(d,max_lag=max_lag,horizon=horizon)
        self.functional=EdgeAwareGraphAggregate(d,6)
        self.functional_scale=nn.Parameter(torch.tensor(-3.0))
        self.norm=nn.LayerNorm(d)
    def forward(self,h,prior,c1,obs_mask,ei,ea,fe,fa,stage,intervention=None):
        B,H,N,D=h.shape
        obs_support=obs_mask.any(-1).float() # actual historical observation, 0..1
        coupled=stage=='coupled' and intervention not in ('mask_only','disable_c1')
        if coupled and c1 is not None:
            obs_support=obs_support*c1['reliability'].squeeze(-1)[:,None]
        bank=h;support=obs_support
        out=[];su=[];norms=[];ents=[]
        with_transport=stage in ('transport','coupled') and intervention!='no_transport'
        for step in range(self.horizon):
            loc=self.local(torch.cat([bank[:,-1],prior[:,step]],-1))*torch.sigmoid(self.local_scale)
            # Functional associations are predictive context, never physical transport.
            if with_transport and intervention!='remove_functional':
                func=self.functional(prior[:,step],fe,fa)*torch.sigmoid(self.functional_scale)
            else:func=0.
            if with_transport:
                prop,nxt,diag=self.trans(bank,support,ei,ea,c1['state'] if c1 is not None else h[:,-1],step,intervention,coupled)
                norms.append(diag['transport_norm']);ents.append(diag['lag_entropy'])
            else:
                prop=0.;nxt=support[:,-1]*0.92
            z=self.norm(prior[:,step]+loc+prop+func)
            bank=torch.cat([bank,z[:,None]],dim=1)
            support=torch.cat([support,nxt[:,None]],dim=1)
            out.append(z);su.append(nxt)
        return torch.stack(out,dim=1),torch.stack(su,dim=1),{'transport_norm':torch.stack(norms).mean() if norms else h.new_zeros(()),'lag_entropy':torch.stack(ents).mean() if ents else h.new_zeros(())}


class PCFTV6(nn.Module):
    STAGES=('direct','observability','transport','coupled')
    def __init__(self,base,graph:DirectedGraph,target_idx,channels=3,hidden=64,max_lag=6,horizon=12):
        super().__init__();self.encoder=STLAEncoderBridge(base,channels)
        self.encoder.set_target_index(target_idx)
        d=int(base.model_dim)
        self.ocpr=OCPR(d)
        self.horizon_prior=HorizonPrior(d,horizon=horizon)
        self.dynamics=LatentDynamics(d,horizon=horizon,max_lag=max_lag)
        self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,d),nn.GELU(),nn.Linear(d,1))
        self.register_buffer('edge_index',graph.edge_index.long())
        self.register_buffer('edge_attr',graph.edge_attr.float())
        self.register_buffer('functional_index',graph.functional_index.long())
        self.register_buffer('functional_attr',graph.functional_attr.float())
        self.evidence_reach=LaggedEvidenceReachability(max_lag=max_lag)
        self.graph_identity=graph.identity
        self.stage='direct'
        self.target_idx=target_idx
    def set_stage(self,stage):
        if stage not in self.STAGES:raise ValueError(stage)
        self.stage=stage
    def forward(self,batch,intervention=None):
        x=batch['x_multi'];mask=batch['observed_mask'];age=batch['missing_age']
        h=self.encoder(batch['x_target'],x,mask,age,batch['tod'],batch['dow'])
        c1_enabled=self.stage!='direct' and intervention!='disable_c1'
        c1=self.ocpr(h,x,mask,age,self.edge_index,self.edge_attr,
                       self.functional_index,self.functional_attr,intervention) if c1_enabled else None
        # Explicit graph-conditioned horizon reachability. Marginal C1 effect is through the prior.
        reach=None
        if c1 is not None:
            reach=self.evidence_reach(mask,c1["reliability"],self.edge_index,self.edge_attr,self.dynamics.horizon)
        prior=self.horizon_prior(h,c1,reach)
        future,support,diag=self.dynamics(h,prior,c1,mask,self.edge_index,self.edge_attr,self.functional_index,self.functional_attr,self.stage,intervention)
        y=self.head(future).squeeze(-1)
        return {'prediction':y,'future_support':support,'state':c1,'diagnostics':diag}
