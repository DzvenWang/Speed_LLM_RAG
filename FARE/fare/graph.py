from __future__ import annotations
import math
import numpy as np, torch
from torch import nn

class GraphRegions(nn.Module):
    """Graph-BFS partitions, each node belongs to one target region; static anchor pools."""
    def __init__(self,adj,target_size=24,max_anchors=32,directed=None):
        super().__init__();a=np.asarray(adj.detach().cpu(),dtype=np.float32);n=a.shape[0]
        graph=(np.abs(a)+np.abs(a.T))>0
        np.fill_diagonal(graph,False)
        seen=set(); parts=[]
        while len(seen)<n:
            s=next(i for i in range(n) if i not in seen);q=[s];region=[]; queued={s}
            while q and len(region)<target_size:
                v=q.pop(0)
                if v in seen:continue
                region.append(v);seen.add(v)
                for u in np.where(graph[v])[0]:
                    u=int(u)
                    if u not in seen and u not in queued:q.append(u);queued.add(u)
            parts.append(region)
        # Merge tiny components into nearest/earliest region (only in low-connectivity graphs).
        r=len(parts); own=np.empty(n,dtype=np.int64)
        for g,region in enumerate(parts):own[region]=g
        mask=np.zeros((r,n),dtype=np.float32); anchor=np.zeros((r,max_anchors),dtype=np.int64); av=np.zeros((r,max_anchors),dtype=bool)
        upmask=np.zeros((r,max_anchors),dtype=np.float32);downmask=np.zeros((r,max_anchors),dtype=np.float32)
        for g,region in enumerate(parts):
            mask[g,region]=1
            candidates=[];
            up=set();down=set()
            if directed is not None:
                upids=directed['upstream_ids'].cpu().numpy();downids=directed['downstream_ids'].cpu().numpy()
                up={int(j) for i in region for j in upids[i] if j>=0 and int(j) not in region}
                down={int(j) for i in region for j in downids[i] if j>=0 and int(j) not in region}
                candidates=list(sorted(up))+list(sorted(down-up))
            dist={v:0 for v in region};queue=list(region)
            while queue and len(candidates)<max_anchors*3:
                v=queue.pop(0)
                for u in np.where(graph[v])[0]:
                    u=int(u)
                    if u not in dist:dist[u]=dist[v]+1;queue.append(u)
                    if u not in region and u not in candidates:candidates.append(u)
            # If region exhausts graph neighborhood use same-region nodes as valid fallback anchors.
            if not candidates:candidates=list(region)
            candidates=candidates[:max_anchors]
            anchor[g,:len(candidates)]=candidates;av[g,:len(candidates)]=True
        for g in range(r):
            for j,u in enumerate(anchor[g]):
                if av[g,j]:
                    # Determine direction relative to this target region.
                    if directed is not None:
                        ups=directed['upstream_ids'].cpu().numpy()
                        downs=directed['downstream_ids'].cpu().numpy()
                        upmask[g,j]=float(any(int(u) in ups[k] for k in parts[g]))
                        downmask[g,j]=float(any(int(u) in downs[k] for k in parts[g]))
        # Fixed graph-walk transfer from each anchor to target nodes (node-specific).
        neigh=np.asarray(graph,dtype=np.float32)+np.eye(n,dtype=np.float32)
        neigh=neigh/neigh.sum(-1,keepdims=True).clip(min=1)
        walks=neigh+.7*(neigh@neigh)+.3*(neigh@neigh@neigh)
        transfer=np.zeros((r,n,max_anchors),dtype=np.float32)
        for region_id,region_nodes in enumerate(parts):
            for node_id in region_nodes:
                vals=walks[node_id,anchor[region_id]]*av[region_id]
                # Always allow a small amount of global anchor adaptation if disconnected.
                vals=vals+av[region_id]*.01
                vals=vals/max(vals.sum(),1e-8)
                transfer[region_id,node_id]=vals
        self.register_buffer('graph_transfer',torch.from_numpy(transfer))
        self.register_buffer('up_anchor',torch.from_numpy(upmask))
        self.register_buffer('down_anchor',torch.from_numpy(downmask))
        self.register_buffer('target_mask',torch.from_numpy(mask))
        self.register_buffer('owner',torch.from_numpy(own))
        self.register_buffer('anchor_ids',torch.from_numpy(anchor))
        self.register_buffer('anchor_valid',torch.from_numpy(av))
        # Non-persistent derived indices: no change to checkpoint state_dict.
        max_region = max(map(len, parts))
        region_ids = np.zeros((r, max_region), dtype=np.int64)
        region_valid = np.zeros((r, max_region), dtype=bool)
        for region_id, nodes in enumerate(parts):
            region_ids[region_id, :len(nodes)] = nodes
            region_valid[region_id, :len(nodes)] = True
        self.register_buffer('region_node_ids', torch.from_numpy(region_ids), persistent=False)
        self.register_buffer('region_node_valid', torch.from_numpy(region_valid), persistent=False)
        self.n=n;self.g=r;self.a=max_anchors
    def forward(self,x,obs_mask):
        b,h,n=x.shape
        ids=self.anchor_ids.reshape(-1)
        ax=x[:,:,ids].reshape(b,h,self.g,self.a).permute(0,2,1,3)
        am=obs_mask[:,:,ids].reshape(b,h,self.g,self.a).permute(0,2,1,3)&self.anchor_valid[None,:,None,:]
        missing=(1-obs_mask.float().mean(1))
        target=(missing[:,None,:]*self.target_mask[None]).sum(-1)/self.target_mask.sum(-1)[None].clamp_min(1)
        cover=am.float().mean((-1,-2))
        up=(am.float()*self.up_anchor[None,:,None,:]).sum((-1,-2))/self.up_anchor.sum(-1)[None].clamp_min(1)/h
        down=(am.float()*self.down_anchor[None,:,None,:]).sum((-1,-2))/self.down_anchor.sum(-1)[None].clamp_min(1)/h
        return {'anchor_values':ax,'anchor_mask':am,'missing':target,'coverage':cover,'valid':(am.sum((-1,-2))>0),'upstream_coverage':up,'downstream_coverage':down,'target_mask':self.target_mask}

class ObservabilityEncoder(nn.Module):
    def __init__(self,hidden=64):
        super().__init__();self.net=nn.Sequential(nn.Linear(7,hidden),nn.GELU(),nn.Linear(hidden,hidden))
    def forward(self,ctx):
        x=ctx['anchor_values'];m=ctx['anchor_mask'];den=m.sum((-1,-2)).clamp_min(1)
        mean=(x*m).sum((-1,-2))/den
        centered=((x-mean[...,None,None])*m)
        std=torch.sqrt((centered.square().sum((-1,-2))/den).clamp_min(1e-6))
        feat=torch.stack([mean,std,ctx['missing'],ctx['coverage'],ctx['valid'].float(),ctx['upstream_coverage'],ctx['downstream_coverage']],-1)
        return self.net(feat)
