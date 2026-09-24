from __future__ import annotations
from collections import deque
import math
import numpy as np
import torch
import torch.nn as nn


FEATURE_NAMES = [
    'top_score','mean_score','score_margin','score_std',
    'mean_latent_sim','max_latent_sim','mean_observed_match','max_observed_match',
    'mean_complementarity','mean_recency','mean_context',
    'completion_coverage','mean_completion_confidence','max_completion_confidence',
    'missing_fraction','memory_fraction',
    'latent_observed_product','observed_complementarity_product','triple_quality_product','coverage_confidence_product',
]


def _finite(x, default=0.0):
    try:
        x=float(x)
    except Exception:
        return float(default)
    return x if np.isfinite(x) else float(default)



def _arr_mean(a):
    a=np.asarray(a,dtype=np.float32); a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else 0.0

def _arr_max(a):
    a=np.asarray(a,dtype=np.float32); a=a[np.isfinite(a)]
    return float(a.max()) if len(a) else 0.0

def _arr_std(a):
    a=np.asarray(a,dtype=np.float32); a=a[np.isfinite(a)]
    return float(a.std()) if len(a) else 0.0

def visible_gate_features(retrieved, completion, query, memory_size, memory_capacity):
    """Build current-query visible features only.

    No x_full/current future label is permitted here. Features describe retrieval
    quality, complementarity and completion confidence available at prediction time.
    """
    score=np.asarray(retrieved.get('score',[]),np.float32) if retrieved is not None else np.asarray([],np.float32)
    lat=np.asarray(retrieved.get('latent_sim',[]),np.float32) if retrieved is not None else np.asarray([],np.float32)
    obs=np.asarray(retrieved.get('observed_match',[]),np.float32) if retrieved is not None else np.asarray([],np.float32)
    comp=np.asarray(retrieved.get('complementarity',[]),np.float32) if retrieved is not None else np.asarray([],np.float32)
    rec=np.asarray(retrieved.get('recency',[]),np.float32) if retrieved is not None else np.asarray([],np.float32)
    ctx=np.asarray(retrieved.get('context',[]),np.float32) if retrieved is not None else np.asarray([],np.float32)
    node_conf=np.asarray(completion.get('node_confidence',[]),np.float32) if completion is not None else np.asarray([],np.float32)
    miss=np.asarray(query['artificial_missing'].numpy(),bool)
    score_sorted=np.sort(score[np.isfinite(score)])[::-1]
    margin=float(score_sorted[0]-score_sorted[1]) if len(score_sorted)>1 else (float(score_sorted[0]) if len(score_sorted)==1 else 0.0)
    mean_lat=_arr_mean(lat); mean_obs=_arr_mean(obs); mean_comp=_arr_mean(comp)
    coverage=_finite(completion.get('coverage',0.0) if completion is not None else 0.0); mean_conf=_arr_mean(node_conf)
    feats=np.asarray([
        _finite(score_sorted[0] if len(score_sorted) else 0.0),
        _arr_mean(score),
        _finite(margin),
        _arr_std(score),
        mean_lat,
        _arr_max(lat),
        mean_obs,
        _arr_max(obs),
        mean_comp,
        _arr_mean(rec),
        _arr_mean(ctx),
        coverage,
        mean_conf,
        _arr_max(node_conf),
        _finite(miss.mean() if miss.size else 0.0),
        _finite(float(memory_size)/max(float(memory_capacity),1.0)),
        mean_lat*mean_obs,
        mean_obs*mean_comp,
        mean_lat*mean_obs*mean_comp,
        coverage*mean_conf,
    ],np.float32)
    return feats


class _GateNet(nn.Module):
    def __init__(self, d, init_probability=0.85):
        super().__init__()
        self.linear=nn.Linear(int(d),1)
        nn.init.zeros_(self.linear.weight)
        p=float(np.clip(init_probability,1e-4,1-1e-4))
        nn.init.constant_(self.linear.bias,math.log(p/(1-p)))
    def forward(self,x):
        return torch.sigmoid(self.linear(x)).squeeze(-1)


class OnlineCompletionArbitrator:
    """Causal learned gate for retrieval-conditioned completion.

    Prediction-time input is visible retrieval/completion metadata only. A training
    target is generated only after that historical query's future has fully matured.
    The target is the best blend-strength multiplier on a small causal diagnostic
    grid for the *already produced* slow and fixed-completion predictions.
    """
    def __init__(self,cfg,seed=10):
        self.cfg=cfg
        torch.manual_seed(int(seed)+1701)
        self.net=_GateNet(len(FEATURE_NAMES),float(cfg.get('initial_probability',0.85)))
        self.opt=torch.optim.AdamW(self.net.parameters(),lr=float(cfg.get('learning_rate',1e-2)),weight_decay=float(cfg.get('weight_decay',1e-4)))
        self.buffer=deque(maxlen=int(cfg.get('buffer_size',512)))
        self.examples_seen=0; self.updates=0
        self.rng=np.random.default_rng(int(seed)+1701)

    def multiplier(self,features):
        if not bool(self.cfg.get('enabled',True)):
            return 1.0
        if self.examples_seen < int(self.cfg.get('min_examples',64)):
            return float(self.cfg.get('warmup_multiplier',1.0))
        x=torch.from_numpy(np.asarray(features,np.float32)).view(1,-1)
        self.net.eval()
        with torch.no_grad(): p=float(self.net(x).item())
        return float(np.clip(p,float(self.cfg.get('min_multiplier',0.0)),float(self.cfg.get('max_multiplier',1.0))))

    def add_example(self,features,target_strength,utility_fixed_pct=0.0):
        target=float(np.clip(target_strength,0.0,1.0))
        weight=1.0+float(self.cfg.get('utility_weight',1.0))*min(abs(float(utility_fixed_pct))/max(float(self.cfg.get('utility_scale_pct',3.0)),1e-6),float(self.cfg.get('utility_weight_cap',3.0)))
        self.buffer.append((np.asarray(features,np.float32),target,float(weight)))
        self.examples_seen+=1
        return self._train()

    def _train(self):
        if len(self.buffer)<int(self.cfg.get('min_examples',64)):
            return {'updated':False,'loss':float('nan'),'examples_seen':self.examples_seen}
        self.net.train(); losses=[]
        for _ in range(int(self.cfg.get('steps_per_update',2))):
            bs=min(int(self.cfg.get('batch_size',64)),len(self.buffer))
            sel=self.rng.choice(np.arange(len(self.buffer)),size=bs,replace=False)
            x=np.stack([self.buffer[int(i)][0] for i in sel])
            y=np.asarray([self.buffer[int(i)][1] for i in sel],np.float32)
            w=np.asarray([self.buffer[int(i)][2] for i in sel],np.float32)
            tx=torch.from_numpy(x); ty=torch.from_numpy(y); tw=torch.from_numpy(w)
            self.opt.zero_grad(set_to_none=True)
            p=self.net(tx)
            # BCE accepts soft targets in [0,1]; weighting emphasizes large historical harms/benefits.
            loss=nn.functional.binary_cross_entropy(p,ty,reduction='none')
            loss=(loss*tw).sum()/torch.clamp(tw.sum(),min=1e-6)
            loss.backward(); torch.nn.utils.clip_grad_norm_(self.net.parameters(),float(self.cfg.get('grad_clip',5.0))); self.opt.step()
            losses.append(float(loss.item())); self.updates+=1
        self.net.eval()
        return {'updated':True,'loss':float(np.mean(losses)),'examples_seen':self.examples_seen}

    def state_dict(self):
        return {'model':self.net.state_dict(),'optimizer':self.opt.state_dict(),'examples_seen':self.examples_seen,'updates':self.updates,'feature_names':FEATURE_NAMES}
