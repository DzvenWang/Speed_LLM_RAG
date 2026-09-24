from __future__ import annotations
from collections import deque
import numpy as np
import torch
import torch.nn as nn

from .evidence import node_shift_alignment, softmax_weights

FEATURE_NAMES = [
    'latent_sim','global_observed_match','global_complementarity','recency','context',
    'node_observed_match','node_overlap_fraction','patch_observed_match','patch_overlap_fraction',
    'node_complementarity','patch_complementarity','patch_missing_fraction','node_missing_fraction',
    'candidate_patch_observed_fraction','candidate_consensus','global_rank_norm','global_score','age_scaled',
]


def temporal_patches(history: int, patch_size: int):
    h=int(history); p=max(1,int(patch_size))
    return [(s,min(s+p,h)) for s in range(0,h,p)]


def _sim_from_overlap(qv, cv, qo, co, scale, min_count=1):
    qv=np.asarray(qv,np.float32); cv=np.asarray(cv,np.float32)
    qo=np.asarray(qo,bool); co=np.asarray(co,bool)
    m=qo & co & np.isfinite(qv) & np.isfinite(cv)
    cnt=int(m.sum()); qcnt=max(int((qo & np.isfinite(qv)).sum()),1)
    frac=float(cnt)/float(qcnt)
    if cnt<int(min_count): return 0.0,frac
    mae=float(np.mean(np.abs(qv[m]-cv[m])))
    return float(np.exp(-mae/max(float(scale),1e-6))),frac


def prepare_candidate_pool(rows, pool, qx, qobs, qmissing, completion_cfg):
    """Align the current causal global candidate pool once for patch-wise scoring/completion."""
    if pool is None or len(pool.get('local',[]))==0:
        return None
    qx=np.asarray(qx,np.float32); qobs=np.asarray(qobs,bool); qmissing=np.asarray(qmissing,bool)
    local=np.asarray(pool['local'],np.int64); K=len(local)
    cx=np.stack([np.asarray(rows[int(li)].x_obs,np.float32) for li in local],axis=0)
    cobs=np.stack([np.asarray(rows[int(li)].history_observed,bool) for li in local],axis=0)
    aligned=np.empty_like(cx,dtype=np.float32)
    shifts=np.empty((K,qx.shape[1]),np.float32)
    for j,li in enumerate(local):
        sh=node_shift_alignment(qx,qobs,rows[int(li)].x_obs,rows[int(li)].history_observed,
                                completion_cfg.get('min_node_overlap_steps',2))
        shifts[j]=sh; aligned[j]=np.asarray(rows[int(li)].x_obs,np.float32)+sh[None,:]
    return {'local':local,'case_index':np.asarray(pool['case_index'],np.int64),'cx':cx,'cobs':cobs,'aligned':aligned,'shifts':shifts,
            'latent_sim':np.asarray(pool['latent_sim'],np.float32),'observed_match':np.asarray(pool['observed_match'],np.float32),
            'global_complementarity':np.asarray(pool['complementarity'],np.float32),'recency':np.asarray(pool['recency'],np.float32),
            'context':np.asarray(pool['context'],np.float32),'age':np.asarray(pool['age'],np.float32),
            'global_score':np.asarray(pool['score'],np.float32),'qx':qx,'qobs':qobs,'qmissing':qmissing}


def build_patch_candidate_tables(prepared, utility_cfg, retrieval_cfg):
    """Build visible feature tables for every missing node x temporal patch.

    Hidden current truth/future is intentionally absent from this function.
    """
    if prepared is None: return {}
    qx=prepared['qx']; qobs=prepared['qobs']; qm=prepared['qmissing']
    cobs=prepared['cobs']; aligned=prepared['aligned']; K=len(prepared['local']); T,N=qx.shape
    patches=temporal_patches(T,utility_cfg.get('temporal_patch_size',3))
    node_scale=float(retrieval_cfg.get('nodewise',{}).get('local_observed_match_scale',retrieval_cfg.get('observed_match_scale',.5)))
    patch_scale=float(utility_cfg.get('patch_observed_match_scale',node_scale))
    min_patch_overlap=int(utility_cfg.get('min_patch_overlap_steps',1))
    rank_norm=np.arange(K,dtype=np.float32)/max(K-1,1)
    score=prepared['global_score']; smin=float(np.min(score)) if K else 0.; smax=float(np.max(score)) if K else 1.
    score_norm=(score-smin)/max(smax-smin,1e-6)
    age_scale=max(float(utility_cfg.get('age_scale_steps',retrieval_cfg.get('recency_tau_steps',288.0))),1.)
    age_scaled=np.minimum(prepared['age']/age_scale,4.0).astype(np.float32)
    tables={}
    for n in np.where(np.any(qm,axis=0))[0]:
        n=int(n)
        nsim,nfrac=np.zeros(K,np.float32),np.zeros(K,np.float32)
        node_comp=np.zeros(K,np.float32)
        denom_node=max(int(qm[:,n].sum()),1)
        node_miss_frac=float(qm[:,n].mean())
        for j in range(K):
            nsim[j],nfrac[j]=_sim_from_overlap(qx[:,n],prepared['cx'][j,:,n],qobs[:,n],cobs[j,:,n],node_scale,1)
            node_comp[j]=float((qm[:,n]&cobs[j,:,n]).sum())/denom_node
        for pid,(s,e) in enumerate(patches):
            target=qm[s:e,n]; miss_count=int(target.sum())
            if miss_count<=0: continue
            feats=[]; eligible=[]; psim=[]; pfrac=[]; pcomp=[]; consensus=[]
            # Pool median is visible because it uses only matured candidate observations.
            vals=aligned[:,s:e,n].copy(); av=cobs[:,s:e,n]&np.isfinite(vals)
            med=np.full(e-s,np.nan,np.float32)
            for tt in range(e-s):
                z=vals[:,tt][av[:,tt]]
                if len(z): med[tt]=float(np.median(z))
            for j in range(K):
                sim,frac=_sim_from_overlap(qx[s:e,n],prepared['cx'][j,s:e,n],qobs[s:e,n],cobs[j,s:e,n],patch_scale,min_patch_overlap)
                cov=float((target & cobs[j,s:e,n]).sum())/max(miss_count,1)
                cand_obs_frac=float(cobs[j,s:e,n].mean())
                if cov<=0: continue
                m=target & cobs[j,s:e,n] & np.isfinite(vals[j]) & np.isfinite(med)
                if np.any(m):
                    disagreement=float(np.mean(np.abs(vals[j,m]-med[m])))
                    cons=float(np.exp(-disagreement/max(float(utility_cfg.get('candidate_consensus_scale',.5)),1e-6)))
                else: cons=0.0
                f=np.asarray([
                    prepared['latent_sim'][j],prepared['observed_match'][j],prepared['global_complementarity'][j],prepared['recency'][j],prepared['context'][j],
                    nsim[j],nfrac[j],sim,frac,node_comp[j],cov,float(target.mean()),node_miss_frac,
                    cand_obs_frac,cons,1.0-rank_norm[j],score_norm[j],age_scaled[j]
                ],np.float32)
                eligible.append(j); feats.append(f); psim.append(sim); pfrac.append(frac); pcomp.append(cov); consensus.append(cons)
            if not eligible: continue
            feats=np.stack(feats).astype(np.float32); eligible=np.asarray(eligible,np.int64)
            # Visible similarity fallback for warm-up / baseline.
            pw=utility_cfg.get('similarity_fallback_weights',{})
            sim_score=(float(pw.get('latent',.10))*feats[:,0] + float(pw.get('global_observed',.10))*feats[:,1]
                       +float(pw.get('node_observed',.15))*feats[:,5] + float(pw.get('patch_observed',.25))*feats[:,7]
                       +float(pw.get('patch_complementarity',.25))*feats[:,10] + float(pw.get('recency',.10))*feats[:,3]
                       +float(pw.get('context',.05))*feats[:,4])
            tables[(n,pid)]={'node':n,'patch_id':pid,'start':s,'end':e,'eligible':eligible,'features':feats,
                             'similarity_score':sim_score.astype(np.float32),'patch_observed_match':np.asarray(psim,np.float32),
                             'patch_overlap_fraction':np.asarray(pfrac,np.float32),'patch_complementarity':np.asarray(pcomp,np.float32),
                             'candidate_consensus':np.asarray(consensus,np.float32)}
    return tables


class _UtilityMLP(nn.Module):
    def __init__(self, dim, hidden=(128,64)):
        super().__init__()
        layers=[]; d=int(dim)
        for h in hidden:
            layers += [nn.Linear(d,int(h)),nn.ReLU(),nn.LayerNorm(int(h))]; d=int(h)
        layers.append(nn.Linear(d,3)); self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)


class OnlineUtilityScorer:
    """Strictly causal online Helpful/Neutral/Harmful scorer."""
    def __init__(self,cfg,device,seed=10):
        self.cfg=cfg; self.device=device; torch.manual_seed(int(seed)+330)
        hidden=tuple(cfg.get('hidden_dims',[128,64])); self.model=_UtilityMLP(len(FEATURE_NAMES),hidden).to(device)
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=float(cfg.get('learning_rate',1e-3)),weight_decay=float(cfg.get('weight_decay',1e-4)))
        self.x=deque(maxlen=int(cfg.get('buffer_size',20000))); self.y=deque(maxlen=int(cfg.get('buffer_size',20000)))
        self.utility=deque(maxlen=int(cfg.get('buffer_size',20000))); self.rng=np.random.default_rng(int(seed)+331)
        self.new_since_train=0; self.train_updates=0; self.train_steps=0; self.last_loss=float('nan')
    def __len__(self): return len(self.y)
    def class_counts(self):
        if not self.y: return [0,0,0]
        return np.bincount(np.asarray(self.y,np.int64),minlength=3).tolist()
    @property
    def ready(self):
        c=self.class_counts(); return len(self)>=int(self.cfg.get('min_examples',256)) and c[0]>=int(self.cfg.get('min_class_examples',8)) and c[2]>=int(self.cfg.get('min_class_examples',8))
    def utility_to_label(self,u_pct):
        if float(u_pct)>=float(self.cfg.get('helpful_threshold_pct',0.05)): return 2
        if float(u_pct)<=float(self.cfg.get('harmful_threshold_pct',-0.05)): return 0
        return 1
    def add(self,features,utility_pct):
        f=np.asarray(features,np.float32)
        u=np.asarray(utility_pct,np.float32).reshape(-1)
        if f.ndim==1: f=f[None,:]
        for i in range(min(len(f),len(u))):
            if np.all(np.isfinite(f[i])) and np.isfinite(u[i]):
                self.x.append(f[i].copy()); self.y.append(self.utility_to_label(float(u[i]))); self.utility.append(float(u[i])); self.new_since_train+=1
    def maybe_train(self):
        if not self.ready or self.new_since_train<int(self.cfg.get('train_every_new_labels',64)): return None
        X=np.stack(self.x).astype(np.float32); Y=np.asarray(self.y,np.int64)
        counts=np.maximum(np.bincount(Y,minlength=3).astype(np.float32),1.0)
        cw=(counts.sum()/counts); cw=cw/np.mean(cw); cw=np.clip(cw,0.4,3.0)
        weight=torch.tensor(cw,dtype=torch.float32,device=self.device)
        self.model.train(); losses=[]
        for _ in range(int(self.cfg.get('train_steps_per_update',4))):
            bs=min(int(self.cfg.get('batch_size',256)),len(Y)); sel=self.rng.choice(len(Y),size=bs,replace=False)
            xb=torch.from_numpy(X[sel]).to(self.device); yb=torch.from_numpy(Y[sel]).to(self.device)
            self.opt.zero_grad(set_to_none=True); logits=self.model(xb); loss=nn.functional.cross_entropy(logits,yb,weight=weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(),float(self.cfg.get('grad_clip',5.0))); self.opt.step()
            losses.append(float(loss.item())); self.train_steps+=1
        self.model.eval(); self.new_since_train=0; self.train_updates+=1; self.last_loss=float(np.mean(losses))
        return self.last_loss
    @torch.no_grad()
    def predict(self,features):
        f=np.asarray(features,np.float32)
        if f.ndim==1: f=f[None,:]
        if not self.ready:
            return None
        self.model.eval(); logits=self.model(torch.from_numpy(f).to(self.device)); p=torch.softmax(logits,dim=-1).cpu().numpy().astype(np.float32)
        penalty=float(self.cfg.get('harmful_penalty',1.5)); score=p[:,2]-penalty*p[:,0]
        return {'probs':p,'utility_score':score.astype(np.float32)}


def select_patch_supports(prepared,tables,scorer,utility_cfg,completion_cfg,mode='utility'):
    """Select supports per node x temporal patch using similarity or learned predictive utility."""
    if prepared is None: return {'patches':{},'scorer_ready':False}
    out={}; topk=int(utility_cfg.get('top_k_per_patch',2)); temp=max(float(completion_cfg.get('score_temperature',.15)),1e-6)
    scorer_ready=bool(scorer is not None and scorer.ready)
    for key,t in tables.items():
        cand=t['eligible']; feats=t['features']; sim=t['similarity_score']
        pred=None
        if mode=='utility' and scorer_ready: pred=scorer.predict(feats)
        if pred is None:
            score=sim.copy(); accepted=np.ones(len(score),bool); phelp=np.full(len(score),np.nan,np.float32); pharm=np.full(len(score),np.nan,np.float32)
            source='similarity_fallback' if mode=='utility' else 'similarity'
        else:
            score=pred['utility_score']; phelp=pred['probs'][:,2]; pharm=pred['probs'][:,0]
            accepted=(phelp>=float(utility_cfg.get('helpful_prob_min',.40))) & (pharm<=float(utility_cfg.get('harmful_prob_max',.40))) & (score>=float(utility_cfg.get('utility_score_min',-0.05)))
            source='utility'
        eligible_idx=np.where(accepted)[0]
        if len(eligible_idx)==0:
            if bool(utility_cfg.get('abstain_when_no_support',True)): continue
            eligible_idx=np.arange(len(score))
        take=min(topk,len(eligible_idx)); chosen=eligible_idx[np.argsort(-score[eligible_idx])[:take]]
        cs=score[chosen].astype(np.float32); z=(cs-cs.max())/temp; ew=np.exp(np.clip(z,-40,40)); w=(ew/max(float(ew.sum()),1e-12)).astype(np.float32)
        out[key]={'node':t['node'],'patch_id':t['patch_id'],'start':t['start'],'end':t['end'],'pool_rank':cand[chosen].astype(np.int16),
                  'local':prepared['local'][cand[chosen]].astype(np.int64),'case_index':prepared['case_index'][cand[chosen]].astype(np.int64),
                  'score':cs,'weights':w,'features':feats[chosen].astype(np.float32),'source':source,
                  'p_helpful':phelp[chosen].astype(np.float32),'p_harmful':pharm[chosen].astype(np.float32),
                  'patch_observed_match':t['patch_observed_match'][chosen].astype(np.float32),
                  'patch_complementarity':t['patch_complementarity'][chosen].astype(np.float32)}
    return {'patches':out,'scorer_ready':scorer_ready}


def sample_probe_specs(tables, utility_cfg, rng):
    """Choose a small diverse set of candidate patch interventions for causal downstream labels."""
    all_specs=[]
    for key,t in tables.items():
        for r in range(len(t['eligible'])):
            all_specs.append((key,r,float(t['similarity_score'][r])))
    if not all_specs: return []
    budget=min(int(utility_cfg.get('probes_per_query',12)),len(all_specs)); n_top=max(1,int(round(budget*float(utility_cfg.get('probe_top_fraction',.5)))))
    order=np.argsort([-x[2] for x in all_specs]); top=[all_specs[int(i)] for i in order[:min(n_top,len(order))]]
    used={(a,b) for a,b,_ in top}; remain=[x for x in all_specs if (x[0],x[1]) not in used]
    n_rand=budget-len(top)
    if n_rand>0 and remain:
        sel=rng.choice(len(remain),size=min(n_rand,len(remain)),replace=False); top += [remain[int(i)] for i in sel]
    return top[:budget]


def build_probe_augmented_histories(query,prepared,tables,specs):
    qx=np.asarray(query['x_obs'].numpy(),np.float32); qm=np.asarray(query['artificial_missing'].numpy(),bool)
    xbatch=[]; features=[]; meta=[]
    for key,row_idx,_ in specs:
        t=tables[key]; pool_rank=int(t['eligible'][int(row_idx)]); n=int(t['node']); s,e=int(t['start']),int(t['end'])
        v=prepared['aligned'][pool_rank,s:e,n]; a=prepared['cobs'][pool_rank,s:e,n] & qm[s:e,n] & np.isfinite(v)
        if not np.any(a): continue
        aug=qx.copy(); loc=np.where(a)[0]+s; aug[loc,n]=v[a]
        xbatch.append(aug); features.append(t['features'][int(row_idx)])
        meta.append({'node':n,'patch_id':int(t['patch_id']),'start':s,'end':e,'case_index':int(prepared['case_index'][pool_rank]),'pool_rank':pool_rank})
    if not xbatch: return None
    return {'x_batch':np.stack(xbatch).astype(np.float32),'features':np.stack(features).astype(np.float32),'meta':meta}
