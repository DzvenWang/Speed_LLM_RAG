from __future__ import annotations
import numpy as np
import torch
from .history import CausalTensorArchive
from .tmcp import recover_batch

class StructuralExpert:
    def __init__(self,ctx):
        self.ctx=ctx; c=ctx.cfg['structural']; self.cfg=c
        self.archive=CausalTensorArchive(ctx,c['historical_days'],c['day_steps'])
    def restore_batch(self,x,observed,artificial,window_starts,smoke=False):
        vals=[]; masks=[]
        for q,st in enumerate(window_starts):
            v,m,_=self.archive.build(int(st),np.asarray(x[q],np.float32),np.asarray(observed[q],bool)); vals.append(v); masks.append(m)
        device=self.ctx.device; out=[]; infos=[]; bs=int(self.cfg['batch_size'])
        raw_hi=max(self.ctx.mean+8.0*self.ctx.std,1.0)
        for s in range(0,len(vals),bs):
            vt=torch.from_numpy(np.stack(vals[s:s+bs])).to(device); mt=torch.from_numpy(np.stack(masks[s:s+bs])).to(device)
            rec,info=recover_batch(vt,mt,rho=self.cfg['rho'],theta0=self.cfg['theta0'],beta=self.cfg['beta'],gamma=self.cfg['gamma'],factor=self.cfg['factor'],tol=self.cfg['tol'],max_iter=self.cfg['smoke_max_iter'] if smoke else self.cfg['max_iter'],clamp_min=0.,clamp_max=raw_hi,decomposition=self.cfg['decomposition']); infos.append(info)
            cur=rec[:,:,-1,:].transpose(1,2).cpu().numpy(); out.append((cur-self.ctx.mean)/self.ctx.std)
        est=np.concatenate(out,0).astype(np.float32); restored=np.where(np.asarray(artificial,bool),est,np.asarray(x,np.float32))
        return restored,infos
