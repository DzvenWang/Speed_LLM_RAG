from __future__ import annotations

import numpy as np
import torch

from .backbone import FrozenBackbone
from .factory import load_context, load_retrieval, load_backbone_aware
from .retrieval import LearnedRetrievalIndex
from .structural import StructuralExpert


class ExpertRuntime:
    def __init__(self, ctx):
        self.ctx=ctx; self.context=load_context(ctx); self.retrieval=load_retrieval(ctx); self.backbone_aware=load_backbone_aware(ctx)
        self.index=LearnedRetrievalIndex(ctx,ctx.artifact_root/'retrieval_index'); self.structural=StructuralExpert(ctx)
        self.bb=FrozenBackbone(ctx.build_backbone(),ctx.cfg['runtime']['inference_batch_size'])

    @torch.inference_mode()
    def neural_restore(self,x,observed,artificial,tod,dow,window_starts,batch_size=16):
        names={"context":[],"retrieval":[],"backbone_aware":[]}; retrieval_diag=[]
        for s in range(0,len(x),int(batch_size)):
            sl=slice(s,min(s+int(batch_size),len(x)))
            xt=torch.from_numpy(np.asarray(x[sl],np.float32).copy()).to(self.ctx.device)
            ot=torch.from_numpy(np.asarray(observed[sl],bool).copy()).to(self.ctx.device)
            at=torch.from_numpy(np.asarray(artificial[sl],bool).copy()).to(self.ctx.device)
            tt=torch.from_numpy(np.asarray(tod[sl],np.int64).copy()).to(self.ctx.device)
            dt=torch.from_numpy(np.asarray(dow[sl],np.int64).copy()).to(self.ctx.device)
            st=torch.from_numpy(np.asarray(window_starts[sl],np.int64).copy()).to(self.ctx.device)
            cr,_=self.context(xt,ot,at,tt,dt)
            q=self.context.global_embedding(xt,ot,tt,dt)
            inds,sim=self.index.search(q,st,self.ctx.cfg['training']['retrieval']['top_k'],self.ctx.cfg['training']['retrieval']['search_k'])
            cn,cv=self.index.materialize(inds); cnt=torch.from_numpy(cn).to(self.ctx.device); cvt=torch.from_numpy(cv).to(self.ctx.device)
            rr,diag=self.retrieval(xt,ot,at,cr,cnt,cvt,sim,self.context.encoder.a_norm)
            base=self.bb.predict_norm_tensor(xt,tt,dt); br,_=self.backbone_aware(xt,ot,at,tt,dt,cr,base)
            names['context'].append(cr.cpu().numpy().astype(np.float32)); names['retrieval'].append(rr.cpu().numpy().astype(np.float32)); names['backbone_aware'].append(br.cpu().numpy().astype(np.float32))
            retrieval_diag.append({"mean_gate":float(diag['gate'][at].mean().cpu()) if bool(at.any()) else 0.0,"mean_max_weight":float(diag['weights'].max(1).values.mean().cpu())})
        return {k:np.concatenate(v,0) for k,v in names.items()},retrieval_diag

    def restore_all(self,x,observed,artificial,tod,dow,window_starts,smoke=False):
        out,diag=self.neural_restore(x,observed,artificial,tod,dow,window_starts)
        structural,info=self.structural.restore_batch(x,observed,artificial,window_starts,smoke=smoke)
        out['structural']=structural
        return out,{"retrieval":diag,"structural":[i.__dict__ for i in info]}
