from __future__ import annotations

import gc
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifacts import save_checkpoint, load_model_checkpoint
from .backbone import FrozenBackbone
from .cache import CacheSplit
from .data import RestorationDataset
from .factory import build_context_model, build_retrieval_model, build_backbone_aware_model, load_context
from .losses import (
    masked_l1, baseline_relative_no_harm, pairwise_utility_ranking_loss,
    per_query_masked_l1, intervention_landscape_loss, reliability_supervision_loss,
    query_gate_mean,
)
from .retrieval import LearnedRetrievalIndex
from .structural import StructuralExpert
from .utils import balanced_indices, log, seed_all, write_json

VERSION="0.2.4"


def _cuda_release():
    gc.collect()
    if torch.cuda.is_available():
        try: torch.cuda.synchronize()
        except Exception: pass
        torch.cuda.empty_cache()


def _embedding_micro_batch(ctx, requested):
    requested=max(1,int(requested))
    if ctx.device.type != 'cuda' or not torch.cuda.is_available(): return requested
    try:
        free_b,_=torch.cuda.mem_get_info(ctx.device); free_gb=float(free_b)/(1024**3)
        if free_gb < 4: cap=2
        elif free_gb < 6: cap=4
        elif free_gb < 10: cap=8
        elif free_gb < 14: cap=16
        elif free_gb < 22: cap=32
        else: cap=64
        return max(1,min(requested,cap))
    except Exception: return min(requested,8)


def _is_oom(exc): return isinstance(exc,torch.cuda.OutOfMemoryError) or 'out of memory' in str(exc).lower()


def _indices(data: CacheSplit, smoke: bool, train: bool, families, seed):
    if not smoke: return np.arange(len(data),dtype=np.int64)
    n=160 if train else 80
    return balanced_indices(data.meta,min(n,len(data)),families,seed+(0 if train else 1000))


def _landscape_indices(ctx,data,split,smoke,seed_offset=0):
    cc=ctx.cfg['intervention_landscape']; train=(split=='train')
    n=int(cc['smoke_train_rows'] if smoke and train else cc['smoke_val_rows'] if smoke else cc['train_rows'] if train else cc['val_rows'])
    return balanced_indices(data.meta,min(n,len(data)),ctx.cfg['experiment']['families'],ctx.seed+seed_offset+(0 if train else 1000))


def _loader(ctx,split,batch_size,smoke,shuffle,min_window_start=None):
    d=CacheSplit(ctx.formal_cache,split); idx=_indices(d,smoke,split=='train',ctx.cfg['experiment']['families'],ctx.seed)
    if min_window_start is not None:
        starts=np.asarray(d.meta.window_start,np.int64); idx=idx[starts[idx] >= int(min_window_start)]
        if len(idx)==0: raise RuntimeError(f'No {split} rows remain after causal retrieval cutoff={min_window_start}')
    return _loader_rows(ctx,split,batch_size,idx,shuffle)


def _loader_rows(ctx,split,batch_size,rows,shuffle):
    rows=np.asarray(rows,dtype=np.int64)
    if len(rows)==0: raise RuntimeError(f'No rows available for {split} loader')
    ds=RestorationDataset(ctx.formal_cache,split,rows)
    g=torch.Generator().manual_seed(ctx.seed+(0 if split=='train' else 1))
    return DataLoader(ds,batch_size=int(batch_size),shuffle=bool(shuffle),num_workers=int(ctx.cfg['runtime']['num_workers']),pin_memory=torch.cuda.is_available(),generator=g)


def _move(batch,device):
    return {k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}


def _meta(ctx,kind):
    return {"version":VERSION,"kind":kind,"target":ctx.target,"dataset":ctx.dataset,"backbone_sha":ctx.checkpoint_sha256,"seed":ctx.seed}


def _bb_forward_chunks(bb,x,tod,dow,chunk=64):
    out=[]
    for s in range(0,len(x),max(1,int(chunk))): out.append(bb.forward_norm_tensor(x[s:s+chunk],tod[s:s+chunk],dow[s:s+chunk]))
    return torch.cat(out,0)


def _set_all_trainable(model, flag: bool):
    for p in model.parameters(): p.requires_grad_(bool(flag))


def _set_module_trainable(module, flag: bool):
    for p in module.parameters(): p.requires_grad_(bool(flag))


def _optimizer(params,lr,wd):
    pp=[p for p in params if p.requires_grad]
    if not pp: raise RuntimeError('No trainable parameters')
    return torch.optim.AdamW(pp,lr=float(lr),weight_decay=float(wd))


def _raw_training_loss(c, estimate, raw_restored, b, bb):
    rec=masked_l1(estimate,b['full'],b['artificial'])
    with torch.no_grad(): base=bb.predict_norm_tensor(b['x'],b['tod'],b['dow'])
    pred=bb.forward_norm_tensor(raw_restored,b['tod'],b['dow'])
    fc=masked_l1(pred,b['y_norm'],b['future_valid'])
    nh,_,_=baseline_relative_no_harm(pred,base,b['y_norm'],b['future_valid'],float(c.get('no_harm_margin',0.0)))
    loss=float(c['reconstruction_weight'])*rec+float(c['raw_forecast_weight'])*fc+float(c.get('raw_utility_weight',0.0))*nh
    return loss,rec,fc,nh


@torch.no_grad()
def _counterfactual_landscape(ctx,b,raw_restored,bb):
    alphas=[float(v) for v in ctx.cfg['intervention_landscape']['alpha_grid']]
    base=bb.predict_norm_tensor(b['x'],b['tod'],b['dow']); bmae=per_query_masked_l1(base,b['y_norm'],b['future_valid'])
    B=len(b['x']); variants=[]
    for alpha in alphas:
        z=torch.where(b['artificial'],b['x']+alpha*(raw_restored-b['x']),b['x']); variants.append(z)
    vx=torch.stack(variants,1).reshape(B*len(alphas),*b['x'].shape[1:])
    vtd=b['tod'][:,None,:].expand(-1,len(alphas),-1).reshape(B*len(alphas),-1)
    vdw=b['dow'][:,None,:].expand(-1,len(alphas),-1).reshape(B*len(alphas),-1)
    py=_bb_forward_chunks(bb,vx,vtd,vdw,ctx.cfg['runtime']['inference_batch_size'])
    yy=b['y_norm'][:,None].expand(-1,len(alphas),-1,-1).reshape(B*len(alphas),*b['y_norm'].shape[1:])
    mm=b['future_valid'][:,None].expand(-1,len(alphas),-1,-1).reshape(B*len(alphas),*b['future_valid'].shape[1:])
    mae=per_query_masked_l1(py,yy,mm).reshape(B,len(alphas))
    utility=(bmae[:,None]-mae)/bmae[:,None].clamp_min(1e-6)
    ai=torch.argmax(utility,dim=1); alpha_tensor=torch.tensor(alphas,device=utility.device,dtype=torch.float32)
    return alpha_tensor[ai],utility


def _landscape_root(ctx,expert): return ctx.artifact_root/'intervention_landscape'/expert

def _load_landscape(ctx,expert,split):
    return np.load(_landscape_root(ctx,expert)/f'{split}_utility_f16.npy',mmap_mode='r')


def _load_best_alpha(ctx,expert,split):
    return np.load(_landscape_root(ctx,expert)/f'{split}_best_alpha_f16.npy',mmap_mode='r')


def _save_landscape_meta(ctx,expert,stats):
    root=_landscape_root(ctx,expert); root.mkdir(parents=True,exist_ok=True)
    write_json(root/'landscape_meta.json',{
        'version':VERSION,'expert':expert,
        'alpha_grid':[float(v) for v in ctx.cfg['intervention_landscape']['alpha_grid']],
        'label':'complete baseline-relative downstream forecast utility curve over intervention alpha',
        'future_truth_policy':'formal train/val labels only; never evaluation',**stats,
    })


def _eligible_landscape_rows(ctx,data,split,smoke,eligible_rows,seed_offset=0):
    eligible=np.asarray(eligible_rows,dtype=np.int64)
    eligible=np.unique(eligible[(eligible>=0)&(eligible<len(data))])
    if len(eligible)==0: return eligible
    cc=ctx.cfg['intervention_landscape']; train=(split=='train')
    n=int(cc['smoke_train_rows'] if smoke and train else cc['smoke_val_rows'] if smoke else cc['train_rows'] if train else cc['val_rows'])
    if n<=0 or len(eligible)<=n: return eligible
    sub=data.meta.iloc[eligible].reset_index(drop=True)
    pos=balanced_indices(sub,n,ctx.cfg['experiment']['families'],ctx.seed+seed_offset+(0 if train else 1000))
    return eligible[np.asarray(pos,dtype=np.int64)]


def _build_landscape_labels(ctx,expert,smoke,provider,bb,seed_offset=0,eligible_rows_by_split=None):
    root=_landscape_root(ctx,expert); root.mkdir(parents=True,exist_ok=True); stats={}
    A=len(ctx.cfg['intervention_landscape']['alpha_grid']); bs=int(ctx.cfg['intervention_landscape']['label_batch_size'])
    for split in ('train','val'):
        d=CacheSplit(ctx.formal_cache,split)
        rows=(_eligible_landscape_rows(ctx,d,split,smoke,eligible_rows_by_split[split],seed_offset)
              if eligible_rows_by_split is not None else _landscape_indices(ctx,d,split,smoke,seed_offset))
        if len(rows)==0: raise RuntimeError(f'No eligible {split} rows for {expert} intervention landscape')
        curves=np.full((len(d),A),np.nan,np.float16); best_alpha=np.full(len(d),np.nan,np.float16)
        loader=_loader_rows(ctx,split,bs,rows,False); done=0
        for batch in loader:
            b=_move(batch,ctx.device)
            with torch.inference_mode(): raw=provider(split,b); alpha,u=_counterfactual_landscape(ctx,b,raw,bb)
            rr=b['row'].detach().cpu().numpy().astype(np.int64)
            curves[rr]=u.cpu().numpy().astype(np.float16); best_alpha[rr]=alpha.cpu().numpy().astype(np.float16); done+=len(rr)
            if done%max(40,bs*10)<len(rr) or done>=len(rows): log(f'{expert.upper()}_INTERVENTION_LANDSCAPE {split} {done}/{len(rows)}')
        np.save(root/f'{split}_utility_f16.npy',curves); np.save(root/f'{split}_best_alpha_f16.npy',best_alpha)
        vv=np.asarray(best_alpha,np.float32); vv=vv[np.isfinite(vv)]
        stats[f'{split}_rows']=int(len(vv)); stats[f'{split}_mean_alpha']=float(vv.mean()) if len(vv) else 0.; stats[f'{split}_positive_alpha_rate']=float((vv>0).mean()) if len(vv) else 0.
        cc=np.asarray(curves,np.float32); good=np.all(np.isfinite(cc),axis=1)
        if np.any(good): stats[f'{split}_mean_utility_curve']=[float(v) for v in cc[good].mean(0)]
    _save_landscape_meta(ctx,expert,stats)
    log(f"{expert.upper()}_LANDSCAPE_SUMMARY train_mean_alpha={stats.get('train_mean_alpha',0.):.3f} train_positive={stats.get('train_positive_alpha_rate',0.):.3f} val_mean_alpha={stats.get('val_mean_alpha',0.):.3f} val_positive={stats.get('val_positive_alpha_rate',0.):.3f}")
    log(f'{expert.upper()}_INTERVENTION_LANDSCAPE_PASS {root}')
    return root


def _landscape_rows(curves):
    a=np.asarray(curves,np.float32)
    return np.flatnonzero(np.all(np.isfinite(a),axis=1)).astype(np.int64)


def _init_landscape_bias(intervention, curves, cfg):
    a=np.asarray(curves,np.float32); good=np.all(np.isfinite(a),axis=1)
    mean=np.mean(a[good],axis=0) if np.any(good) else np.zeros(a.shape[1],np.float32)
    scaled=mean*float(cfg['utility_scale'])
    intervention.initialize_landscape_bias(torch.tensor(scaled,dtype=torch.float32,device=intervention.alpha_grid.device))
    best=int(np.argmax(mean)); alpha=float(intervention.alpha_grid[best].detach().cpu())
    return mean,alpha


def _controller_training_loss(ctx, info, restored, b, bb, target_curve):
    cc=ctx.cfg['intervention_landscape']
    land=intervention_landscape_loss(
        info['landscape'],target_curve,info['alpha_grid'] if 'alpha_grid' in info else torch.tensor(cc['alpha_grid'],device=target_curve.device),
        utility_scale=float(cc['utility_scale']),pairwise_min_gap=float(cc['pairwise_min_utility_gap']),
        choice_temperature=float(cc['choice_temperature']))
    rel,rel_mae,_=reliability_supervision_loss(info['reliability'],info['estimate'],b['full'],b['artificial'])
    pred=bb.forward_norm_tensor(restored,b['tod'],b['dow'])
    with torch.no_grad(): base=bb.predict_norm_tensor(b['x'],b['tod'],b['dow'])
    fc=masked_l1(pred,b['y_norm'],b['future_valid']); nh,_,_=baseline_relative_no_harm(pred,base,b['y_norm'],b['future_valid'],0.0)
    loss=(float(cc['curve_weight'])*land['regression'] + float(cc['choice_weight'])*land['choice'] +
          float(cc['pairwise_weight'])*land['pairwise'] + float(cc['reliability_weight'])*rel +
          float(cc['forecast_weight'])*fc + float(cc['no_harm_weight'])*nh)
    return loss,land,rel,rel_mae,fc,nh


def _target_alpha_tensor(ctx,target_curve):
    grid=torch.tensor(ctx.cfg['intervention_landscape']['alpha_grid'],dtype=torch.float32,device=target_curve.device)
    return grid[torch.argmax(target_curve,dim=1)]


@torch.no_grad()
def _val_context(model,loader,bb,ctx,use_raw=False,landscape_targets=None):
    model.eval(); rs=rn=fs=fn=bs=bn=0.; harmful=nq=0; gates=[]; metrics=[]; relmae=[]; alphas=[]
    cc=ctx.cfg['intervention_landscape']
    for batch in loader:
        b=_move(batch,ctx.device); restored,info=model(b['x'],b['observed'],b['artificial'],b['tod'],b['dow']); z=info['raw_restored'] if use_raw else restored
        rr=(info['estimate']-b['full']).abs()[b['artificial']]; rs+=float(rr.sum()); rn+=int(rr.numel())
        pred=bb.forward_norm_tensor(z,b['tod'],b['dow']); base=bb.predict_norm_tensor(b['x'],b['tod'],b['dow'])
        ff=(pred-b['y_norm']).abs()[b['future_valid']]; be=(base-b['y_norm']).abs()[b['future_valid']]; fs+=float(ff.sum()); fn+=int(ff.numel()); bs+=float(be.sum()); bn+=int(be.numel())
        eq=per_query_masked_l1(pred,b['y_norm'],b['future_valid']); bq=per_query_masked_l1(base,b['y_norm'],b['future_valid']); harmful+=int((eq>bq).sum()); nq+=len(eq)
        if not use_raw and torch.any(b['artificial']):
            gates.append(float(info['gate'][b['artificial']].mean())); alphas.extend(info['selected_alpha'].detach().cpu().tolist())
        if landscape_targets is not None:
            rows=b['row'].detach().cpu().numpy().astype(np.int64); tc=torch.from_numpy(np.asarray(landscape_targets[rows],np.float32).copy()).to(ctx.device)
            lm=intervention_landscape_loss(info['landscape'],tc,model.intervention.alpha_grid,float(cc['utility_scale']),float(cc['pairwise_min_utility_gap']),float(cc['choice_temperature']))
            metrics.append({k:float(lm[k].detach()) for k in ['curve_mae_pp','alpha_acc','alpha_mae','pair_acc','pred_positive_rate','oracle_positive_rate','pred_alpha_mean','oracle_alpha_mean']})
            _,rm,_=reliability_supervision_loss(info['reliability'],info['estimate'],b['full'],b['artificial']); relmae.append(float(rm))
    out={'rec':rs/max(rn,1),'forecast':fs/max(fn,1),'baseline':bs/max(bn,1),'harmful_rate':harmful/max(nq,1),'gate':float(np.mean(gates)) if gates else 1.0 if use_raw else 0.,'selected_alpha_mean':float(np.mean(alphas)) if alphas else 1.0 if use_raw else 0.}
    if metrics:
        for k in metrics[0]: out[k]=float(np.mean([m[k] for m in metrics]))
        out['reliability_mae']=float(np.mean(relmae))
    return out


def train_context(ctx,smoke=False):
    seed_all(ctx.seed); c=ctx.cfg['training']['context']; model=build_context_model(ctx); bb=FrozenBackbone(ctx.build_backbone(),ctx.cfg['runtime']['inference_batch_size'])
    restp=ctx.artifact_root/'context'/'restorer.pt'; bestp=ctx.artifact_root/'context'/'best.pt'; hist={'stage_a':[],'stage_b_landscape':[]}
    _set_all_trainable(model,True); _set_module_trainable(model.intervention,False)
    tr=_loader(ctx,'train',c['batch_size'],smoke,True); va=_loader(ctx,'val',c['batch_size'],smoke,False); opt=_optimizer(model.parameters(),c['learning_rate'],c['weight_decay'])
    epochs=int(c['smoke_stage_a_epochs'] if smoke else c['stage_a_max_epochs']); best=math.inf; bad=0
    for ep in range(1,epochs+1):
        model.train(); ls=0.; nb=0
        for batch in tr:
            b=_move(batch,ctx.device); opt.zero_grad(set_to_none=True); _,info=model(b['x'],b['observed'],b['artificial'],b['tod'],b['dow']); loss,*_= _raw_training_loss(c,info['estimate'],info['raw_restored'],b,bb); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(c['gradient_clip'])); opt.step(); ls+=float(loss.detach()); nb+=1
        v=_val_context(model,va,bb,ctx,True); gain=(v['baseline']-v['forecast'])/max(v['baseline'],1e-12)*100; hist['stage_a'].append({'epoch':ep,'train_loss':ls/max(nb,1),**v}); log(f"CONTEXT_RESTORER_EPOCH {ep:03d}/{epochs} train={ls/max(nb,1):.6f} val_rec={v['rec']:.6f} raw_forecast={v['forecast']:.6f} baseline={v['baseline']:.6f} raw_gain={gain:+.3f}% harmful={v['harmful_rate']:.3f}")
        if v['forecast']<best-1e-6: best=v['forecast']; bad=0; save_checkpoint(restp,model,opt,ep,best,_meta(ctx,'context_restorer'))
        else: bad+=1
        if not smoke and bad>=int(c['patience']): break
    load_model_checkpoint(restp,model,ctx.device,{"version":VERSION,"target":ctx.target,"backbone_sha":ctx.checkpoint_sha256}); model.eval()
    provider=lambda split,b: model(b['x'],b['observed'],b['artificial'],b['tod'],b['dow'])[1]['raw_restored']
    _build_landscape_labels(ctx,'context',smoke,provider,bb,101)
    tr_curve=_load_landscape(ctx,'context','train'); va_curve=_load_landscape(ctx,'context','val'); mean_curve,global_alpha=_init_landscape_bias(model.intervention,tr_curve,ctx.cfg['intervention_landscape']); log(f'CONTEXT_LANDSCAPE_INIT global_best_alpha={global_alpha:.3f} mean_curve_pp={[round(float(v)*100,4) for v in mean_curve]}')
    _set_all_trainable(model,False); _set_module_trainable(model.intervention,True); opt=_optimizer(model.intervention.parameters(),c['controller_learning_rate'],c['weight_decay'])
    trc=_loader_rows(ctx,'train',c['batch_size'],_landscape_rows(tr_curve),True); vac=_loader_rows(ctx,'val',c['batch_size'],_landscape_rows(va_curve),False); epochs=int(c['smoke_stage_b_epochs'] if smoke else c['stage_b_max_epochs']); best=math.inf; bad=0
    for ep in range(1,epochs+1):
        model.eval(); model.intervention.train(); sums={'loss':0.,'curve':0.,'choice':0.,'pair':0.,'rel':0.}; nb=0
        for batch in trc:
            b=_move(batch,ctx.device); rows=b['row'].detach().cpu().numpy().astype(np.int64); tc=torch.from_numpy(np.asarray(tr_curve[rows],np.float32).copy()).to(ctx.device); ta=_target_alpha_tensor(ctx,tc)
            opt.zero_grad(set_to_none=True); restored,info=model(b['x'],b['observed'],b['artificial'],b['tod'],b['dow'],alpha_override=ta); info['alpha_grid']=model.intervention.alpha_grid
            loss,lm,rel,_,_,_=_controller_training_loss(ctx,info,restored,b,bb,tc); loss.backward(); torch.nn.utils.clip_grad_norm_(model.intervention.parameters(),float(c['gradient_clip'])); opt.step(); sums['loss']+=float(loss.detach()); sums['curve']+=float(lm['regression'].detach()); sums['choice']+=float(lm['choice'].detach()); sums['pair']+=float(lm['pairwise'].detach()); sums['rel']+=float(rel.detach()); nb+=1
        v=_val_context(model,vac,bb,ctx,False,va_curve); gain=(v['baseline']-v['forecast'])/max(v['baseline'],1e-12)*100; hist['stage_b_landscape'].append({'epoch':ep,**{f'train_{k}':vv/max(nb,1) for k,vv in sums.items()},**v}); log(f"CONTEXT_LANDSCAPE_EPOCH {ep:03d}/{epochs} train={sums['loss']/max(nb,1):.6f} curve={v['curve_mae_pp']:.3f}pp alpha_acc={v['alpha_acc']:.3f} alpha_mae={v['alpha_mae']:.3f} pred_alpha={v['pred_alpha_mean']:.3f} oracle_alpha={v['oracle_alpha_mean']:.3f} rel_mae={v['reliability_mae']:.3f} val_forecast={v['forecast']:.6f} baseline={v['baseline']:.6f} gain={gain:+.3f}% harmful={v['harmful_rate']:.3f}")
        if v['forecast']<best-1e-6: best=v['forecast']; bad=0; save_checkpoint(bestp,model,opt,ep,best,_meta(ctx,'context'))
        else: bad+=1
        if not smoke and bad>=int(c['patience']): break
    write_json(ctx.artifact_root/'context'/'history.json',{'version':VERSION,'best_val_forecast_mae_norm':best,'history':hist}); log(f'TRAIN_CONTEXT_PASS {bestp}'); return bestp


def _precompute_candidate_assignments(ctx, context, index, split: str, smoke: bool):
    d=CacheSplit(ctx.formal_cache,split); c=ctx.cfg['training']['retrieval']; top_k=int(c['top_k']); requested_bs=int(c['embedding_batch_size']); bs=_embedding_micro_batch(ctx,requested_bs)
    M=len(d); inds_all=np.full((M,top_k),-1,np.int32); sim_all=np.full((M,top_k),-1e4,np.float16)
    cutoff=int(np.sort(np.asarray(index.starts,np.int64))[min(top_k-1,len(index.starts)-1)])+int(ctx.cfg['experiment']['history'])
    rows=np.arange(M,dtype=np.int64) if not smoke else _indices(d,True,split=='train',ctx.cfg['experiment']['families'],ctx.seed)
    context.eval(); _cuda_release(); log(f'CANDIDATE_ASSIGN_PROFILE split={split} requested_batch={requested_bs} initial_micro_batch={bs}')
    done=0
    with torch.inference_mode():
        while done < len(rows):
            cur=min(bs,len(rows)-done); rr=rows[done:done+cur]
            try:
                starts=np.asarray(d.meta.window_start.iloc[rr],np.int64); eligible=starts>=cutoff
                if np.any(eligible):
                    ee=rr[eligible]; x=torch.from_numpy(np.asarray(d.x_obs[ee],np.float32).copy()).to(ctx.device); o=torch.from_numpy(np.asarray(d.history_observed[ee],bool).copy()).to(ctx.device); td=torch.from_numpy(np.asarray(d.tod_hist[ee],np.int64).copy()).to(ctx.device); dw=torch.from_numpy(np.asarray(d.dow_hist[ee],np.int64).copy()).to(ctx.device); st=torch.from_numpy(np.asarray(d.meta.window_start.iloc[ee],np.int64)).to(ctx.device)
                    q=context.global_embedding(x,o,td,dw); ii,ss=index.search(q,st,top_k,int(c['search_k'])); inds_all[ee]=ii.cpu().numpy().astype(np.int32); sim_all[ee]=ss.cpu().numpy().astype(np.float16)
                    del x,o,td,dw,st,q,ii,ss
                done += cur
                if done % max(100,bs*20) < cur or done>=len(rows): log(f'CANDIDATE_ASSIGN {split} {done}/{len(rows)} micro_batch={bs}')
            except RuntimeError as exc:
                if ctx.device.type=='cuda' and _is_oom(exc) and bs>1:
                    new_bs=max(1,bs//2); log(f'CANDIDATE_ASSIGN_OOM split={split} batch={bs} -> retry_batch={new_bs}'); bs=new_bs; _cuda_release(); continue
                raise
    root=ctx.artifact_root/'retrieval_index'; root.mkdir(parents=True,exist_ok=True); np.save(root/f'{split}_candidate_indices.npy',inds_all); np.save(root/f'{split}_candidate_similarity_f16.npy',sim_all)


def build_retrieval_index(ctx,smoke=False):
    _cuda_release(); context=load_context(ctx); c=ctx.cfg['training']['retrieval']; rows=min(4000,int(c['memory_rows'])) if smoke else int(c['memory_rows'])
    p=LearnedRetrievalIndex.build(ctx,context,ctx.artifact_root/'retrieval_index',rows,int(c['embedding_batch_size'])); index=LearnedRetrievalIndex(ctx,p)
    _precompute_candidate_assignments(ctx,context,index,'train',smoke); _precompute_candidate_assignments(ctx,context,index,'val',smoke); log(f'BUILD_RETRIEVAL_INDEX_PASS {p}'); return p


def _load_assignments(ctx, split):
    root=ctx.artifact_root/'retrieval_index'; return np.load(root/f'{split}_candidate_indices.npy',mmap_mode='r'),np.load(root/f'{split}_candidate_similarity_f16.npy',mmap_mode='r')


def _retrieval_batch(ctx,context,index,b,assignments=None):
    with torch.no_grad():
        _,ci=context(b['x'],b['observed'],b['artificial'],b['tod'],b['dow']); cr=ci['raw_restored']
        if assignments is None:
            q=context.global_embedding(b['x'],b['observed'],b['tod'],b['dow']); inds,sim=index.search(q,b['window_start'].long(),ctx.cfg['training']['retrieval']['top_k'],ctx.cfg['training']['retrieval']['search_k'])
        else:
            rows=b['row'].detach().cpu().numpy().astype(np.int64); ii=np.asarray(assignments[0][rows],np.int64); ss=np.asarray(assignments[1][rows],np.float32)
            if np.any(ii<0): raise RuntimeError('Retrieval batch contains rows without causal precomputed candidates')
            inds=torch.from_numpy(ii).to(ctx.device); sim=torch.from_numpy(ss).to(ctx.device)
        cn,cv=index.materialize(inds); candidates=torch.from_numpy(cn).to(ctx.device); cvalid=torch.from_numpy(cv).to(ctx.device)
    return cr,candidates,cvalid,sim


@torch.no_grad()
def build_retrieval_utility_labels(ctx,smoke=False):
    context=load_context(ctx); context.eval(); index=LearnedRetrievalIndex(ctx,ctx.artifact_root/'retrieval_index'); bb=FrozenBackbone(ctx.build_backbone(),ctx.cfg['runtime']['inference_batch_size']); c=ctx.cfg['training']['retrieval']
    k=min(int(c['smoke_utility_label_k'] if smoke else c['utility_label_k']),int(c['top_k']))
    outroot=ctx.artifact_root/'retrieval_utility'; outroot.mkdir(parents=True,exist_ok=True)
    for split in ('train','val'):
        d=CacheSplit(ctx.formal_cache,split); assigns=_load_assignments(ctx,split); labels=np.full((len(d),int(c['top_k'])),np.nan,np.float16)
        rows=_indices(d,smoke,split=='train',ctx.cfg['experiment']['families'],ctx.seed) if smoke else np.arange(len(d),dtype=np.int64)
        usable=rows[np.all(np.asarray(assigns[0][rows,:k])>=0,axis=1)]
        bs=max(1,min(8,int(c['batch_size']))); done=0
        while done<len(usable):
            rr=usable[done:done+bs]
            x=torch.from_numpy(np.asarray(d.x_obs[rr],np.float32).copy()).to(ctx.device); o=torch.from_numpy(np.asarray(d.history_observed[rr],bool).copy()).to(ctx.device); a=torch.from_numpy(np.asarray(d.artificial_missing[rr],bool).copy()).to(ctx.device); td=torch.from_numpy(np.asarray(d.tod_hist[rr],np.int64).copy()).to(ctx.device); dw=torch.from_numpy(np.asarray(d.dow_hist[rr],np.int64).copy()).to(ctx.device); y=torch.from_numpy(np.asarray(d.y_norm[rr],np.float32).copy()).to(ctx.device); fv=torch.from_numpy(np.asarray(d.future_valid[rr],bool).copy()).to(ctx.device)
            _,ci=context(x,o,a,td,dw); cr=ci['raw_restored']; ii=np.asarray(assigns[0][rr,:k],np.int64); cn,cv=index.materialize(torch.from_numpy(ii).to(ctx.device)); cand=torch.from_numpy(cn).to(ctx.device); valid=torch.from_numpy(cv).to(ctx.device)
            base=bb.predict_norm_tensor(x,td,dw); bmae=per_query_masked_l1(base,y,fv)
            variants=[]
            for j in range(k):
                est=torch.where(valid[:,j],cand[:,j],cr); variants.append(torch.where(a,est,x))
            vx=torch.stack(variants,1).reshape(len(rr)*k,*x.shape[1:]); vtd=td[:,None,:].expand(-1,k,-1).reshape(len(rr)*k,-1); vdw=dw[:,None,:].expand(-1,k,-1).reshape(len(rr)*k,-1)
            py=_bb_forward_chunks(bb,vx,vtd,vdw,ctx.cfg['runtime']['inference_batch_size']); yy=y[:,None].expand(-1,k,-1,-1).reshape(len(rr)*k,*y.shape[1:]); mm=fv[:,None].expand(-1,k,-1,-1).reshape(len(rr)*k,*fv.shape[1:])
            cmae=per_query_masked_l1(py,yy,mm).reshape(len(rr),k); util=(bmae[:,None]-cmae)/bmae[:,None].clamp_min(1e-6); labels[rr,:k]=util.cpu().numpy().astype(np.float16)
            done+=len(rr)
            if done%max(50,bs*10)<len(rr) or done>=len(usable): log(f'RETRIEVAL_UTILITY {split} {done}/{len(usable)} k={k}')
        np.save(outroot/f'{split}_candidate_utility_f16.npy',labels)
    write_json(outroot/'utility_meta.json',{'version':VERSION,'label':'relative_downstream_forecast_utility','ranking':'pairwise order supervision','candidate_k':k,'backbone_sha':ctx.checkpoint_sha256,'future_truth_policy':'formal train/val labels only; never evaluation'})
    log(f'BUILD_RETRIEVAL_UTILITY_PASS {outroot}'); return outroot


def _load_utility(ctx,split): return np.load(ctx.artifact_root/'retrieval_utility'/f'{split}_candidate_utility_f16.npy',mmap_mode='r')


@torch.no_grad()
def _val_retrieval(model,loader,context,index,bb,ctx,assignments,use_raw=False,utility_labels=None,landscape_targets=None):
    model.eval(); rs=rn=fs=fn=bs=bn=0.; harmful=nq=0; gates=[]; sg=[]; rloss=[]; racc=[]; metrics=[]; relmae=[]
    c=ctx.cfg['training']['retrieval']; cc=ctx.cfg['intervention_landscape']
    for batch in loader:
        b=_move(batch,ctx.device); cr,cn,cv,sim=_retrieval_batch(ctx,context,index,b,assignments); restored,info=model(b['x'],b['observed'],b['artificial'],cr,cn,cv,sim,context.encoder.a_norm); z=info['raw_restored'] if use_raw else restored
        rr=(info['estimate']-b['full']).abs()[b['artificial']]; pred=bb.forward_norm_tensor(z,b['tod'],b['dow']); base=bb.predict_norm_tensor(b['x'],b['tod'],b['dow']); ff=(pred-b['y_norm']).abs()[b['future_valid']]; be=(base-b['y_norm']).abs()[b['future_valid']]
        rs+=float(rr.sum()); rn+=int(rr.numel()); fs+=float(ff.sum()); fn+=int(ff.numel()); bs+=float(be.sum()); bn+=int(be.numel()); eq=per_query_masked_l1(pred,b['y_norm'],b['future_valid']); bq=per_query_masked_l1(base,b['y_norm'],b['future_valid']); harmful+=int((eq>bq).sum()); nq+=len(eq)
        if torch.any(b['artificial']): gates.append(float(info['gate'][b['artificial']].mean())); sg.append(float(info['support_gate'][b['artificial']].mean()))
        rows=b['row'].detach().cpu().numpy().astype(np.int64)
        if utility_labels is not None:
            uu=torch.from_numpy(np.asarray(utility_labels[rows],np.float32).copy()).to(ctx.device); rl,ra,_=pairwise_utility_ranking_loss(info['query_candidate_scores'],uu,torch.isfinite(uu),float(c['pairwise_min_utility_gap'])); rloss.append(float(rl)); racc.append(float(ra))
        if landscape_targets is not None:
            tc=torch.from_numpy(np.asarray(landscape_targets[rows],np.float32).copy()).to(ctx.device); lm=intervention_landscape_loss(info['landscape'],tc,model.intervention.alpha_grid,float(cc['utility_scale']),float(cc['pairwise_min_utility_gap']),float(cc['choice_temperature'])); metrics.append({k:float(lm[k]) for k in ['curve_mae_pp','alpha_acc','alpha_mae','pair_acc','pred_positive_rate','oracle_positive_rate','pred_alpha_mean','oracle_alpha_mean']}); _,rm,_=reliability_supervision_loss(info['reliability'],info['estimate'],b['full'],b['artificial']); relmae.append(float(rm))
    out={'rec':rs/max(rn,1),'forecast':fs/max(fn,1),'baseline':bs/max(bn,1),'harmful_rate':harmful/max(nq,1),'gate':float(np.mean(gates)) if gates else 1.0 if use_raw else 0.,'support_gate':float(np.mean(sg)) if sg else 0.,'rank_loss':float(np.mean(rloss)) if rloss else None,'rank_acc':float(np.mean(racc)) if racc else None}
    if metrics:
        for k in metrics[0]: out[k]=float(np.mean([m[k] for m in metrics]))
        out['reliability_mae']=float(np.mean(relmae))
    return out


def train_retrieval(ctx,smoke=False):
    seed_all(ctx.seed+11); c=ctx.cfg['training']['retrieval']; context=load_context(ctx); context.eval(); index=LearnedRetrievalIndex(ctx,ctx.artifact_root/'retrieval_index'); model=build_retrieval_model(ctx); bb=FrozenBackbone(ctx.build_backbone(),ctx.cfg['runtime']['inference_batch_size'])
    cutoff=int(np.sort(np.asarray(index.starts,np.int64))[min(int(c['top_k'])-1,len(index.starts)-1)])+int(ctx.cfg['experiment']['history']); tr=_loader(ctx,'train',c['batch_size'],smoke,True,cutoff); va=_loader(ctx,'val',c['batch_size'],smoke,False); train_assign=_load_assignments(ctx,'train'); val_assign=_load_assignments(ctx,'val'); train_u=_load_utility(ctx,'train'); val_u=_load_utility(ctx,'val')
    restp=ctx.artifact_root/'retrieval'/'restorer_ranker.pt'; bestp=ctx.artifact_root/'retrieval'/'best.pt'; hist={'stage_a':[],'stage_b_landscape':[]}
    _set_all_trainable(model,True); _set_module_trainable(model.intervention,False); opt=_optimizer(model.parameters(),c['learning_rate'],c['weight_decay']); epochs=int(c['smoke_stage_a_epochs'] if smoke else c['stage_a_max_epochs']); best=math.inf; bad=0
    for ep in range(1,epochs+1):
        model.train(); ls=ranks=accs=0.; nb=0
        for batch in tr:
            b=_move(batch,ctx.device); cr,cn,cv,sim=_retrieval_batch(ctx,context,index,b,train_assign); opt.zero_grad(set_to_none=True); _,info=model(b['x'],b['observed'],b['artificial'],cr,cn,cv,sim,context.encoder.a_norm)
            base_loss,*_= _raw_training_loss(c,info['estimate'],info['raw_restored'],b,bb); rows=b['row'].detach().cpu().numpy().astype(np.int64); uu=torch.from_numpy(np.asarray(train_u[rows],np.float32).copy()).to(ctx.device); rank,acc,_=pairwise_utility_ranking_loss(info['query_candidate_scores'],uu,torch.isfinite(uu),float(c['pairwise_min_utility_gap'])); loss=base_loss+float(c['ranking_weight'])*rank
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(c['gradient_clip'])); opt.step(); ls+=float(loss.detach()); ranks+=float(rank.detach()); accs+=float(acc.detach()); nb+=1
        v=_val_retrieval(model,va,context,index,bb,ctx,val_assign,True,val_u); gain=(v['baseline']-v['forecast'])/max(v['baseline'],1e-12)*100; hist['stage_a'].append({'epoch':ep,'train_loss':ls/max(nb,1),'train_rank_loss':ranks/max(nb,1),'train_rank_acc':accs/max(nb,1),**v}); log(f"RETRIEVAL_RESTORER_EPOCH {ep:03d}/{epochs} train={ls/max(nb,1):.6f} rank={ranks/max(nb,1):.5f} rank_acc={accs/max(nb,1):.3f} val_rank={v['rank_loss']:.5f} val_rank_acc={v['rank_acc']:.3f} raw_forecast={v['forecast']:.6f} baseline={v['baseline']:.6f} raw_gain={gain:+.3f}% support_gate={v['support_gate']:.3f}")
        score=v['forecast']+0.05*(v['rank_loss'] if v['rank_loss'] is not None else math.log(2.0))
        if score<best-1e-6: best=score; bad=0; save_checkpoint(restp,model,opt,ep,best,_meta(ctx,'retrieval_restorer_ranker'))
        else: bad+=1
        if not smoke and bad>=int(c['patience']): break
    load_model_checkpoint(restp,model,ctx.device,{"version":VERSION,"target":ctx.target,"backbone_sha":ctx.checkpoint_sha256}); model.eval()
    assign_cache={'train':train_assign,'val':val_assign}
    def provider(split,b):
        cr,cn,cv,sim=_retrieval_batch(ctx,context,index,b,assign_cache[split]); return model(b['x'],b['observed'],b['artificial'],cr,cn,cv,sim,context.encoder.a_norm)[1]['raw_restored']
    label_k=int(c['smoke_utility_label_k'] if smoke else c['utility_label_k']); eligible={'train':np.flatnonzero(np.all(np.isfinite(np.asarray(train_u[:,:label_k],np.float32)),axis=1)).astype(np.int64),'val':np.flatnonzero(np.all(np.isfinite(np.asarray(val_u[:,:label_k],np.float32)),axis=1)).astype(np.int64)}
    log(f"RETRIEVAL_LANDSCAPE_ELIGIBLE train={len(eligible['train'])} val={len(eligible['val'])} k={label_k}")
    _build_landscape_labels(ctx,'retrieval',smoke,provider,bb,211,eligible_rows_by_split=eligible)
    tr_curve=_load_landscape(ctx,'retrieval','train'); va_curve=_load_landscape(ctx,'retrieval','val'); mean_curve,global_alpha=_init_landscape_bias(model.intervention,tr_curve,ctx.cfg['intervention_landscape']); log(f'RETRIEVAL_LANDSCAPE_INIT global_best_alpha={global_alpha:.3f} mean_curve_pp={[round(float(v)*100,4) for v in mean_curve]}')
    _set_all_trainable(model,False); _set_module_trainable(model.intervention,True); opt=_optimizer(model.intervention.parameters(),c['controller_learning_rate'],c['weight_decay']); trc=_loader_rows(ctx,'train',c['batch_size'],_landscape_rows(tr_curve),True); vac=_loader_rows(ctx,'val',c['batch_size'],_landscape_rows(va_curve),False); epochs=int(c['smoke_stage_b_epochs'] if smoke else c['stage_b_max_epochs']); best=math.inf; bad=0
    for ep in range(1,epochs+1):
        model.eval(); model.intervention.train(); ls=0.; nb=0
        for batch in trc:
            b=_move(batch,ctx.device); rows=b['row'].detach().cpu().numpy().astype(np.int64); tc=torch.from_numpy(np.asarray(tr_curve[rows],np.float32).copy()).to(ctx.device); ta=_target_alpha_tensor(ctx,tc); cr,cn,cv,sim=_retrieval_batch(ctx,context,index,b,train_assign); opt.zero_grad(set_to_none=True); restored,info=model(b['x'],b['observed'],b['artificial'],cr,cn,cv,sim,context.encoder.a_norm,alpha_override=ta); info['alpha_grid']=model.intervention.alpha_grid; loss,*_=_controller_training_loss(ctx,info,restored,b,bb,tc); loss.backward(); torch.nn.utils.clip_grad_norm_(model.intervention.parameters(),float(c['gradient_clip'])); opt.step(); ls+=float(loss.detach()); nb+=1
        v=_val_retrieval(model,vac,context,index,bb,ctx,val_assign,False,val_u,va_curve); gain=(v['baseline']-v['forecast'])/max(v['baseline'],1e-12)*100; hist['stage_b_landscape'].append({'epoch':ep,'train_loss':ls/max(nb,1),**v}); log(f"RETRIEVAL_LANDSCAPE_EPOCH {ep:03d}/{epochs} train={ls/max(nb,1):.6f} curve={v['curve_mae_pp']:.3f}pp alpha_acc={v['alpha_acc']:.3f} alpha_mae={v['alpha_mae']:.3f} pred_alpha={v['pred_alpha_mean']:.3f} oracle_alpha={v['oracle_alpha_mean']:.3f} rel_mae={v['reliability_mae']:.3f} val_forecast={v['forecast']:.6f} baseline={v['baseline']:.6f} gain={gain:+.3f}% harmful={v['harmful_rate']:.3f} rank_acc={v['rank_acc']:.3f}")
        if v['forecast']<best-1e-6: best=v['forecast']; bad=0; save_checkpoint(bestp,model,opt,ep,best,_meta(ctx,'retrieval'))
        else: bad+=1
        if not smoke and bad>=int(c['patience']): break
    write_json(ctx.artifact_root/'retrieval'/'history.json',{'version':VERSION,'pairwise_random_loss':math.log(2.0),'best_val_forecast_mae_norm':best,'history':hist}); log(f'TRAIN_RETRIEVAL_PASS {bestp}'); return bestp


@torch.no_grad()
def _val_backbone_aware(model,loader,context,bb,ctx,use_raw=False,landscape_targets=None):
    model.eval(); rs=rn=fs=fn=bs=bn=0.; harmful=nq=0; gates=[]; metrics=[]; relmae=[]; cc=ctx.cfg['intervention_landscape']
    for batch in loader:
        b=_move(batch,ctx.device); _,ci=context(b['x'],b['observed'],b['artificial'],b['tod'],b['dow']); cr=ci['raw_restored']; base=bb.predict_norm_tensor(b['x'],b['tod'],b['dow']); restored,info=model(b['x'],b['observed'],b['artificial'],b['tod'],b['dow'],cr,base); z=info['raw_restored'] if use_raw else restored
        rr=(info['estimate']-b['full']).abs()[b['artificial']]; pp=bb.forward_norm_tensor(z,b['tod'],b['dow']); ff=(pp-b['y_norm']).abs()[b['future_valid']]; be=(base-b['y_norm']).abs()[b['future_valid']]
        rs+=float(rr.sum()); rn+=int(rr.numel()); fs+=float(ff.sum()); fn+=int(ff.numel()); bs+=float(be.sum()); bn+=int(be.numel()); eq=per_query_masked_l1(pp,b['y_norm'],b['future_valid']); bq=per_query_masked_l1(base,b['y_norm'],b['future_valid']); harmful+=int((eq>bq).sum()); nq+=len(eq)
        if torch.any(b['artificial']): gates.append(float(info['gate'][b['artificial']].mean()))
        if landscape_targets is not None:
            rows=b['row'].detach().cpu().numpy().astype(np.int64); tc=torch.from_numpy(np.asarray(landscape_targets[rows],np.float32).copy()).to(ctx.device); lm=intervention_landscape_loss(info['landscape'],tc,model.intervention.alpha_grid,float(cc['utility_scale']),float(cc['pairwise_min_utility_gap']),float(cc['choice_temperature'])); metrics.append({k:float(lm[k]) for k in ['curve_mae_pp','alpha_acc','alpha_mae','pair_acc','pred_positive_rate','oracle_positive_rate','pred_alpha_mean','oracle_alpha_mean']}); _,rm,_=reliability_supervision_loss(info['reliability'],info['estimate'],b['full'],b['artificial']); relmae.append(float(rm))
    out={'rec':rs/max(rn,1),'forecast':fs/max(fn,1),'baseline':bs/max(bn,1),'harmful_rate':harmful/max(nq,1),'gate':float(np.mean(gates)) if gates else 1.0 if use_raw else 0.}
    if metrics:
        for k in metrics[0]: out[k]=float(np.mean([m[k] for m in metrics]))
        out['reliability_mae']=float(np.mean(relmae))
    return out


def train_backbone_aware(ctx,smoke=False):
    seed_all(ctx.seed+23); c=ctx.cfg['training']['backbone_aware']; context=load_context(ctx); context.eval(); model=build_backbone_aware_model(ctx); bb=FrozenBackbone(ctx.build_backbone(),ctx.cfg['runtime']['inference_batch_size']); tr=_loader(ctx,'train',c['batch_size'],smoke,True); va=_loader(ctx,'val',c['batch_size'],smoke,False)
    restp=ctx.artifact_root/'backbone_aware'/'restorer.pt'; bestp=ctx.artifact_root/'backbone_aware'/'best.pt'; hist={'stage_a':[],'stage_b_landscape':[]}
    _set_all_trainable(model,True); _set_module_trainable(model.intervention,False); opt=_optimizer(model.parameters(),c['learning_rate'],c['weight_decay']); epochs=int(c['smoke_stage_a_epochs'] if smoke else c['stage_a_max_epochs']); best=math.inf; bad=0
    for ep in range(1,epochs+1):
        model.train(); ls=0.; nb=0
        for batch in tr:
            b=_move(batch,ctx.device)
            with torch.no_grad(): _,ci=context(b['x'],b['observed'],b['artificial'],b['tod'],b['dow']); cr=ci['raw_restored']; base=bb.predict_norm_tensor(b['x'],b['tod'],b['dow'])
            opt.zero_grad(set_to_none=True); _,info=model(b['x'],b['observed'],b['artificial'],b['tod'],b['dow'],cr,base); loss,*_= _raw_training_loss(c,info['estimate'],info['raw_restored'],b,bb); reg=(info['delta'][b['artificial']]**2).mean() if torch.any(b['artificial']) else info['delta'].pow(2).mean()*0; loss=loss+float(c['residual_weight'])*reg; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(c['gradient_clip'])); opt.step(); ls+=float(loss.detach()); nb+=1
        v=_val_backbone_aware(model,va,context,bb,ctx,True); gain=(v['baseline']-v['forecast'])/max(v['baseline'],1e-12)*100; hist['stage_a'].append({'epoch':ep,'train_loss':ls/max(nb,1),**v}); log(f"BACKBONE_AWARE_RESTORER_EPOCH {ep:03d}/{epochs} train={ls/max(nb,1):.6f} val_rec={v['rec']:.6f} raw_forecast={v['forecast']:.6f} baseline={v['baseline']:.6f} raw_gain={gain:+.3f}% harmful={v['harmful_rate']:.3f}")
        if v['forecast']<best-1e-6: best=v['forecast']; bad=0; save_checkpoint(restp,model,opt,ep,best,_meta(ctx,'backbone_aware_restorer'))
        else: bad+=1
        if not smoke and bad>=int(c['patience']): break
    load_model_checkpoint(restp,model,ctx.device,{"version":VERSION,"target":ctx.target,"backbone_sha":ctx.checkpoint_sha256}); model.eval()
    def provider(split,b):
        _,ci=context(b['x'],b['observed'],b['artificial'],b['tod'],b['dow']); cr=ci['raw_restored']; base=bb.predict_norm_tensor(b['x'],b['tod'],b['dow']); return model(b['x'],b['observed'],b['artificial'],b['tod'],b['dow'],cr,base)[1]['raw_restored']
    _build_landscape_labels(ctx,'backbone_aware',smoke,provider,bb,323)
    tr_curve=_load_landscape(ctx,'backbone_aware','train'); va_curve=_load_landscape(ctx,'backbone_aware','val'); mean_curve,global_alpha=_init_landscape_bias(model.intervention,tr_curve,ctx.cfg['intervention_landscape']); log(f'BACKBONE_AWARE_LANDSCAPE_INIT global_best_alpha={global_alpha:.3f} mean_curve_pp={[round(float(v)*100,4) for v in mean_curve]}')
    _set_all_trainable(model,False); _set_module_trainable(model.intervention,True); opt=_optimizer(model.intervention.parameters(),c['controller_learning_rate'],c['weight_decay']); trc=_loader_rows(ctx,'train',c['batch_size'],_landscape_rows(tr_curve),True); vac=_loader_rows(ctx,'val',c['batch_size'],_landscape_rows(va_curve),False); epochs=int(c['smoke_stage_b_epochs'] if smoke else c['stage_b_max_epochs']); best=math.inf; bad=0
    for ep in range(1,epochs+1):
        model.eval(); model.intervention.train(); ls=0.; nb=0
        for batch in trc:
            b=_move(batch,ctx.device); rows=b['row'].detach().cpu().numpy().astype(np.int64); tc=torch.from_numpy(np.asarray(tr_curve[rows],np.float32).copy()).to(ctx.device); ta=_target_alpha_tensor(ctx,tc)
            with torch.no_grad(): _,ci=context(b['x'],b['observed'],b['artificial'],b['tod'],b['dow']); cr=ci['raw_restored']; base=bb.predict_norm_tensor(b['x'],b['tod'],b['dow'])
            opt.zero_grad(set_to_none=True); restored,info=model(b['x'],b['observed'],b['artificial'],b['tod'],b['dow'],cr,base,alpha_override=ta); info['alpha_grid']=model.intervention.alpha_grid; loss,*_=_controller_training_loss(ctx,info,restored,b,bb,tc); loss.backward(); torch.nn.utils.clip_grad_norm_(model.intervention.parameters(),float(c['gradient_clip'])); opt.step(); ls+=float(loss.detach()); nb+=1
        v=_val_backbone_aware(model,vac,context,bb,ctx,False,va_curve); gain=(v['baseline']-v['forecast'])/max(v['baseline'],1e-12)*100; hist['stage_b_landscape'].append({'epoch':ep,'train_loss':ls/max(nb,1),**v}); log(f"BACKBONE_AWARE_LANDSCAPE_EPOCH {ep:03d}/{epochs} train={ls/max(nb,1):.6f} curve={v['curve_mae_pp']:.3f}pp alpha_acc={v['alpha_acc']:.3f} alpha_mae={v['alpha_mae']:.3f} pred_alpha={v['pred_alpha_mean']:.3f} oracle_alpha={v['oracle_alpha_mean']:.3f} rel_mae={v['reliability_mae']:.3f} val_forecast={v['forecast']:.6f} baseline={v['baseline']:.6f} gain={gain:+.3f}% harmful={v['harmful_rate']:.3f}")
        if v['forecast']<best-1e-6: best=v['forecast']; bad=0; save_checkpoint(bestp,model,opt,ep,best,_meta(ctx,'backbone_aware'))
        else: bad+=1
        if not smoke and bad>=int(c['patience']): break
    write_json(ctx.artifact_root/'backbone_aware'/'history.json',{'version':VERSION,'best_val_forecast_mae_norm':best,'history':hist}); log(f'TRAIN_BACKBONE_AWARE_PASS {bestp}'); return bestp


@torch.no_grad()
def calibrate_structural(ctx,smoke=False):
    c=ctx.cfg['structural']; d=CacheSplit(ctx.formal_cache,'val'); n=int(c['smoke_calibration_rows'] if smoke else c['calibration_rows']); idx=balanced_indices(d.meta,min(n,len(d)),ctx.cfg['experiment']['families'],ctx.seed+220)
    x=np.asarray(d.x_obs[idx],np.float32).copy(); o=np.asarray(d.history_observed[idx],bool).copy(); a=np.asarray(d.artificial_missing[idx],bool).copy(); y=np.asarray(d.y_norm[idx],np.float32).copy(); fv=np.asarray(d.future_valid[idx],bool).copy(); tod=np.asarray(d.tod_hist[idx],np.int64).copy(); dow=np.asarray(d.dow_hist[idx],np.int64).copy(); starts=np.asarray(d.meta.window_start.iloc[idx],np.int64)
    expert=StructuralExpert(ctx); raw,infos=expert.restore_batch(x,o,a,starts,smoke=smoke); bb=FrozenBackbone(ctx.build_backbone(),ctx.cfg['runtime']['inference_batch_size']); alphas=[float(v) for v in c['alpha_grid']]; scores={}
    for alpha in alphas:
        z=np.where(a,x+alpha*(raw-x),x).astype(np.float32); zt=torch.from_numpy(z).to(ctx.device); tt=torch.from_numpy(tod).to(ctx.device); dt=torch.from_numpy(dow).to(ctx.device); p=_bb_forward_chunks(bb,zt,tt,dt,ctx.cfg['runtime']['inference_batch_size']); mae=float(per_query_masked_l1(p,torch.from_numpy(y).to(ctx.device),torch.from_numpy(fv).to(ctx.device)).mean().cpu()); scores[f'{alpha:.2f}']=mae
    best=min(alphas,key=lambda aa:scores[f'{aa:.2f}']); base=scores[f'{alphas[0]:.2f}']; utility={k:float((base-v)/max(base,1e-12)) for k,v in scores.items()}; path=ctx.artifact_root/'structural'/'calibration.json'; write_json(path,{'version':VERSION,'alpha':float(best),'validation_rows':int(len(idx)),'forecast_mae_norm_by_alpha':scores,'forecast_utility_landscape_by_alpha':utility,'selection':'minimum formal-validation downstream forecast MAE','leakage_policy':'formal validation only; evaluation truth never used','tmcp_infos':[i.__dict__ for i in infos]}); log(f'STRUCTURAL_CALIBRATION_PASS alpha={best:.2f} {path}'); return path
