from __future__ import annotations
from pathlib import Path
import json, numpy as np, torch

class LatentMemoryIndex:
    def __init__(self,temporal_keys,spatial_keys,starts):
        self.temporal_np=np.asarray(temporal_keys,np.float32);self.spatial_np=np.asarray(spatial_keys,np.float32);self.starts_np=np.asarray(starts,np.int64);self._device=None;self.tk=None;self.sk=None;self.starts=None
    def to(self,device):
        if self._device!=str(device): self.tk=torch.from_numpy(self.temporal_np).to(device);self.sk=torch.from_numpy(self.spatial_np).to(device);self.starts=torch.from_numpy(self.starts_np).to(device);self._device=str(device)
        return self
    def search(self,q,key_type,top_k,query_starts,span,temperature=0.1):
        keys=self.tk if key_type=='temporal' else self.sk;sim=q@keys.t();qs=query_starts.long();overlap=(self.starts.unsqueeze(0)<(qs+span).unsqueeze(1))&(qs.unsqueeze(1)<(self.starts+span));sim=sim.masked_fill(overlap,-float('inf'));scores,idx=torch.topk(sim,k=min(int(top_k),keys.shape[0]),dim=1);
        if not torch.isfinite(scores).all():raise RuntimeError('No temporally nonoverlapping retrieval evidence available')
        log=scores/max(float(temperature),1e-6);w=torch.softmax(log,dim=1);ent=-(w.clamp_min(1e-12)*w.clamp_min(1e-12).log()).sum(1)/np.log(max(w.shape[1],2));margin=scores[:,0]-scores[:,1] if scores.shape[1]>1 else torch.zeros_like(scores[:,0]);conf=torch.stack([scores[:,0],margin,1-ent],-1);return idx,scores,w,conf
    def save(self,path,metadata=None):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,temporal_keys=self.temporal_np,spatial_keys=self.spatial_np,starts=self.starts_np);path.with_suffix('.json').write_text(json.dumps(metadata or {},indent=2),encoding='utf-8')
    @classmethod
    def load(cls,path):
        z=np.load(path,allow_pickle=False);return cls(z['temporal_keys'],z['spatial_keys'],z['starts'])

def build_memory_index(encoder,repo,device,batch_size=64):
    encoder.eval();t=[];s=[]
    with torch.no_grad():
        for st in range(0,len(repo),batch_size):
            en=min(st+batch_size,len(repo));x,m,tod,dow=repo.batch(st,en,device);e=encoder(x,m,tod,dow);t.append(e['temporal_key'].cpu());s.append(e['spatial_key'].cpu())
    return LatentMemoryIndex(torch.cat(t).numpy(),torch.cat(s).numpy(),repo.starts.copy())
