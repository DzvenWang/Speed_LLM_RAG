"""Traffic-specific observability, directional influence and boundary support."""
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class TrafficGraph(nn.Module):
    def __init__(self,adj,directed=None,region_size=24,max_neighbors=4):
        super().__init__();a=(torch.as_tensor(adj).float().abs()>0).float();n=a.shape[0]
        if a.shape!=(n,n):raise ValueError('adj must be [N,N]')
        a.fill_diagonal_(0)
        self.n=n;self.metadata_source='directional_neighbors' if directed is not None else 'undirected_topology'
        forward=torch.zeros_like(a);backward=torch.zeros_like(a)
        if directed is None:
            forward=(a+a.T).clamp(max=1);backward=forward.clone()
        else:
            up=directed['upstream_ids'].long();uw=directed['upstream_weights'].float()
            down=directed['downstream_ids'].long();dw=directed['downstream_weights'].float()
            for i in range(n):
                valid=down[i]>=0
                forward[i,down[i,valid]]=dw[i,valid].clamp_min(0)
                valid=up[i]>=0
                backward[i,up[i,valid]]=uw[i,valid].clamp_min(0)
            # documented fallback for isolated directed rows
            forward+=torch.eye(n);backward+=torch.eye(n)
        eye=torch.eye(n)
        def normalized(x):
            x=x+eye;x=x/x.sum(-1,keepdim=True).clamp_min(1e-6)
            return x
        self.register_buffer('forward_adj',normalized(forward))
        self.register_buffer('backward_adj',normalized(backward))
        self.register_buffer('undirected_adj',normalized((a+a.T).clamp(max=1)))
        # BFS partition (fixed, graph-derived; not a propagation boundary).
        nb=((a+a.T)>0).cpu()
        seen=set();owner=torch.empty(n,dtype=torch.long);group=0
        for start in range(n):
            if start in seen:continue
            queue=[start];pending={start};group_nodes=[]
            while queue and len(group_nodes)<region_size:
                u=queue.pop(0)
                if u in seen:continue
                seen.add(u);group_nodes.append(u)
                for v in torch.where(nb[u])[0].tolist():
                    if v not in seen and v not in pending:queue.append(v);pending.add(v)
            owner[group_nodes]=group;group+=1
        self.g=group
        self.register_buffer('owner',owner)
        region=torch.zeros(group,n);region[owner,torch.arange(n)]=1
        self.register_buffer('region_membership',region/region.sum(-1,keepdim=True).clamp_min(1))
        # source neighbors for sparse candidate evidence transport
        k=min(max_neighbors,n)
        vals,ids=self.undirected_adj.topk(k,dim=-1)
        self.register_buffer('neighbor_ids',ids);self.register_buffer('neighbor_weight',vals/vals.sum(-1,keepdim=True).clamp_min(1e-8))
        self.component_neighbors=[torch.where((a[i]+a[:,i])>0)[0].tolist() for i in range(n)]

    def failure_components(self,mask):
        """Exact contiguous missing components on the physical undirected road graph.

        Per-query CPU graph traversal, not a learned substitute for connectivity.
        Feature detached (observability is discrete); no labels or future data.
        """
        b,h,n=mask.shape
        absent=(~mask.any(1)).detach().cpu().numpy()
        sizes=np.zeros((b,n),dtype=np.float32)
        boundary=np.zeros((b,n),dtype=np.float32)
        for row in range(b):
            seen=np.zeros(n,dtype=bool)
            for seed in np.flatnonzero(absent[row]):
                if seen[seed]:continue
                stack=[int(seed)];seen[seed]=True;component=[];observed_boundary=set()
                while stack:
                    v=stack.pop();component.append(v)
                    for u in self.component_neighbors[v]:
                        if absent[row,u]:
                            if not seen[u]:seen[u]=True;stack.append(u)
                        else:observed_boundary.add(u)
                extent=len(component)/max(n,1)
                ratio=len(observed_boundary)/max(len(component),1)
                sizes[row,component]=extent
                boundary[row,component]=min(ratio,1.0)
        return (torch.as_tensor(sizes,device=mask.device),torch.as_tensor(boundary,device=mask.device))

    def support_coverage(self,mask,horizon,steps=3,directional=True):
        # mask [B,H,N], returns per-node / per-horizon two direction supports
        b,h,n=mask.shape
        observation=mask.float()
        base=observation.mean(1) # [B,N]
        def conv(adj):
            x=base;parts=[]
            for i in range(steps):
                x=x@adj.T;parts.append(x)
            return torch.stack(parts,-1)  # [B,N,S]
        a=conv(self.forward_adj if directional else self.undirected_adj)
        d=conv(self.backward_adj if directional else self.undirected_adj)
        # horizon weights prefer longer reach at longer lead times
        p=torch.arange(horizon,device=mask.device).float()
        hop=torch.arange(steps,device=mask.device).float()
        hw=F.softmax(-(hop[None]-(p[:,None]/max(horizon-1,1))*(steps-1)).abs(),-1)
        af=torch.einsum('bns,ps->bnp',a,hw)
        df=torch.einsum('bns,ps->bnp',d,hw)
        # add boundary support by propagating latest observed values over directed edges
        latest=observation[:,-1,:]
        boundary_up=latest@self.backward_adj.T
        boundary_down=latest@self.forward_adj.T
        return af.clamp(0,1),df.clamp(0,1),boundary_up,boundary_down

class PropagationSupportEncoder(nn.Module):
    def __init__(self,d_base=64,d_hidden=64,horizon=12,regimes=4):
        super().__init__();self.p=horizon;self.d=d_hidden
        self.regime=nn.Sequential(nn.Linear(7+d_base,d_hidden),nn.GELU(),nn.Linear(d_hidden,regimes))
        self.horizon_embedding=nn.Embedding(horizon,d_hidden)
        self.mix=nn.Sequential(nn.Linear(d_base+d_hidden+12,d_hidden),nn.GELU(),nn.Linear(d_hidden,d_hidden))
    def forward(self,x,mask,hidden,pred,graph,neutral=False,no_direction=False,no_regime=False):
        b,h,n=x.shape;p=pred.shape[1]
        visible=mask.float();count=visible.sum(1).clamp_min(1)
        avg=(x*visible).sum(1)/count
        last=x[:,-1];first=x[:,0]
        trend=last-first
        local=visible.mean(1)
        fw,bw,ub,db=graph.support_coverage(mask,p,directional=not no_direction)
        missing=1-local
        component_size,boundary_ratio=graph.failure_components(mask)
        feature=torch.stack([avg,trend,local,missing,ub,db,(pred[:,-1]-pred[:,0])],-1)
        logits=self.regime(torch.cat([feature,hidden],-1))
        regime=F.softmax(logits,-1)
        if no_regime:regime=torch.full_like(regime,1/regime.shape[-1])
        regime_embedding=F.pad(regime,(0,self.d-regime.shape[-1]))
        h_emb=self.horizon_embedding(torch.arange(p,device=x.device))[None,None,:,:]
        extra=torch.stack([fw,bw,1-fw,1-bw,ub[:,:,None].expand(-1,-1,p),db[:,:,None].expand(-1,-1,p),
                           local[:,:,None].expand(-1,-1,p),missing[:,:,None].expand(-1,-1,p),
                           trend[:,:,None].expand(-1,-1,p),pred.permute(0,2,1),
                           component_size[:,:,None].expand(-1,-1,p),boundary_ratio[:,:,None].expand(-1,-1,p)],-1)
        embeddings=self.mix(torch.cat([hidden[:,:,None,:].expand(-1,-1,p,-1),
                                        regime_embedding[:,:,None,:].expand(-1,-1,p,-1)+h_emb.expand(b,n,-1,-1),extra],-1))
        if neutral:embeddings=torch.zeros_like(embeddings);fw=torch.zeros_like(fw);bw=torch.zeros_like(bw)
        return {'embedding':embeddings,'forward_coverage':fw,'backward_coverage':bw,
                'boundary_up':ub,'boundary_down':db,'regime':regime,'missingness':missing,
                'component_size':component_size,'missing_boundary_ratio':boundary_ratio,
                'direction_valid':graph.metadata_source=='directional_neighbors'}
