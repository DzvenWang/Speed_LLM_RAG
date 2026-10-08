from __future__ import annotations

import json
import numpy as np
import torch

from .backbone import FrozenBackbone
from .factory import load_context, load_retrieval, load_backbone_aware
from .retrieval import LearnedRetrievalIndex
from .structural import StructuralExpert


class ExpertRuntime:
    def __init__(self, ctx):
        self.ctx=ctx
        self.context=load_context(ctx)
        self.retrieval=load_retrieval(ctx)
        self.backbone_aware=load_backbone_aware(ctx)
        self.index=LearnedRetrievalIndex(ctx,ctx.artifact_root/'retrieval_index')
        self.structural=StructuralExpert(ctx)
        self.bb=FrozenBackbone(ctx.build_backbone(),ctx.cfg['runtime']['inference_batch_size'])
        cal=ctx.artifact_root/'structural'/'calibration.json'
        if not cal.exists(): raise FileNotFoundError(f'Structural calibration missing: {cal}')
        self.structural_alpha=float(json.loads(cal.read_text(encoding='utf-8'))['alpha'])

    @staticmethod
    def _query_diag(info, artificial, extra=None):
        art=artificial.bool(); B=art.shape[0]; out=[]
        for q in range(B):
            m=art[q]
            gate=float(info['gate'][q][m].mean().detach().cpu()) if bool(m.any()) else 0.0
            rel=float(info['reliability'][q][m].mean().detach().cpu()) if bool(m.any()) else 0.0
            rec={
                'selected_alpha':float(info['selected_alpha'][q].detach().cpu()),
                'mean_gate':gate,
                'mean_reliability':rel,
                'predicted_landscape_pp':[float(v) for v in info['landscape'][q].detach().cpu().tolist()],
            }
            if extra is not None: rec.update(extra(q))
            out.append(rec)
        return out

    @torch.inference_mode()
    def neural_restore(self,x,observed,artificial,tod,dow,window_starts,batch_size=16):
        controlled={"context":[],"retrieval":[],"backbone_aware":[]}
        raw={"context":[],"retrieval":[],"backbone_aware":[]}
        diagnostics={"context":[],"retrieval":[],"backbone_aware":[]}
        for s in range(0,len(x),int(batch_size)):
            sl=slice(s,min(s+int(batch_size),len(x)))
            xt=torch.from_numpy(np.asarray(x[sl],np.float32).copy()).to(self.ctx.device)
            ot=torch.from_numpy(np.asarray(observed[sl],bool).copy()).to(self.ctx.device)
            at=torch.from_numpy(np.asarray(artificial[sl],bool).copy()).to(self.ctx.device)
            tt=torch.from_numpy(np.asarray(tod[sl],np.int64).copy()).to(self.ctx.device)
            dt=torch.from_numpy(np.asarray(dow[sl],np.int64).copy()).to(self.ctx.device)
            st=torch.from_numpy(np.asarray(window_starts[sl],np.int64).copy()).to(self.ctx.device)

            cr,ci=self.context(xt,ot,at,tt,dt)
            c_raw=ci['raw_restored']
            q=self.context.global_embedding(xt,ot,tt,dt)
            inds,sim=self.index.search(q,st,self.ctx.cfg['training']['retrieval']['top_k'],self.ctx.cfg['training']['retrieval']['search_k'])
            cn,cv=self.index.materialize(inds)
            cnt=torch.from_numpy(cn).to(self.ctx.device); cvt=torch.from_numpy(cv).to(self.ctx.device)
            rr,ri=self.retrieval(xt,ot,at,c_raw,cnt,cvt,sim,self.context.encoder.a_norm)
            base=self.bb.predict_norm_tensor(xt,tt,dt)
            br,bi=self.backbone_aware(xt,ot,at,tt,dt,c_raw,base)

            controlled['context'].append(cr.cpu().numpy().astype(np.float32)); raw['context'].append(c_raw.cpu().numpy().astype(np.float32))
            controlled['retrieval'].append(rr.cpu().numpy().astype(np.float32)); raw['retrieval'].append(ri['raw_restored'].cpu().numpy().astype(np.float32))
            controlled['backbone_aware'].append(br.cpu().numpy().astype(np.float32)); raw['backbone_aware'].append(bi['raw_restored'].cpu().numpy().astype(np.float32))

            diagnostics['context'].extend(self._query_diag(ci,at))
            diagnostics['retrieval'].extend(self._query_diag(ri,at,lambda i:{
                'mean_support_gate':float(ri['support_gate'][i][at[i]].mean().detach().cpu()) if bool(at[i].any()) else 0.0,
                'mean_max_weight':float(ri['weights'][i].max(0).values.mean().detach().cpu()),
            }))
            diagnostics['backbone_aware'].extend(self._query_diag(bi,at))
        return ({k:np.concatenate(v,0) for k,v in controlled.items()}, {k:np.concatenate(v,0) for k,v in raw.items()}, diagnostics)

    def restore_all(self,x,observed,artificial,tod,dow,window_starts,smoke=False):
        controlled,raw,diag=self.neural_restore(x,observed,artificial,tod,dow,window_starts)
        sraw,info=self.structural.restore_batch(x,observed,artificial,window_starts,smoke=smoke)
        alpha=self.structural_alpha
        sctrl=np.where(np.asarray(artificial,bool),np.asarray(x,np.float32)+alpha*(sraw-np.asarray(x,np.float32)),np.asarray(x,np.float32)).astype(np.float32)
        controlled['structural']=sctrl; raw['structural']=sraw
        diag['structural']={'alpha':alpha,'tmcp':[i.__dict__ for i in info]}
        return controlled,raw,diag
