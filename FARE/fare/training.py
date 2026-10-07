from __future__ import annotations

from pathlib import Path
import math
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifacts import save_checkpoint
from .backbone import FrozenBackbone
from .cache import CacheSplit
from .data import RestorationDataset
from .factory import build_context_model, build_retrieval_model, build_backbone_aware_model, load_context
from .losses import masked_l1
from .retrieval import LearnedRetrievalIndex
from .utils import balanced_indices, log, seed_all, write_json


def _cuda_release():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def _embedding_micro_batch(ctx, requested):
    requested=max(1,int(requested))
    if ctx.device.type != 'cuda' or not torch.cuda.is_available(): return requested
    try:
        free_b,total_b=torch.cuda.mem_get_info(ctx.device); free_gb=float(free_b)/(1024**3)
        if free_gb < 4: cap=2
        elif free_gb < 6: cap=4
        elif free_gb < 10: cap=8
        elif free_gb < 14: cap=16
        elif free_gb < 22: cap=32
        else: cap=64
        return max(1,min(requested,cap))
    except Exception:
        return min(requested,8)


def _is_oom(exc):
    return isinstance(exc,torch.cuda.OutOfMemoryError) or 'out of memory' in str(exc).lower()


def _indices(data: CacheSplit, smoke: bool, train: bool, families, seed):
    if not smoke: return np.arange(len(data),dtype=np.int64)
    n=160 if train else 80
    return balanced_indices(data.meta,min(n,len(data)),families,seed+(0 if train else 1000))

def _loader(ctx,split,batch_size,smoke,shuffle,min_window_start=None):
    d=CacheSplit(ctx.formal_cache,split); idx=_indices(d,smoke,split=='train',ctx.cfg['experiment']['families'],ctx.seed)
    if min_window_start is not None:
        starts=np.asarray(d.meta.window_start,np.int64); idx=idx[starts[idx] >= int(min_window_start)]
        if len(idx)==0: raise RuntimeError(f'No {split} rows remain after causal retrieval cutoff={min_window_start}')
    ds=RestorationDataset(ctx.formal_cache,split,idx)
    g=torch.Generator().manual_seed(ctx.seed+(0 if split=='train' else 1))
    return DataLoader(ds,batch_size=int(batch_size),shuffle=bool(shuffle),num_workers=int(ctx.cfg['runtime']['num_workers']),pin_memory=torch.cuda.is_available(),generator=g)

def _move(batch,device):
    out={}
    for k,v in batch.items(): out[k]=v.to(device,non_blocking=True) if torch.is_tensor(v) else v
    return out

def _meta(ctx,kind): return {"version":"0.2.0","kind":kind,"target":ctx.target,"dataset":ctx.dataset,"backbone_sha":ctx.checkpoint_sha256,"seed":ctx.seed}

def _run_val_context(model,loader,bb,ctx):
    model.eval(); rec_sum=0.; rec_n=0; fc_sum=0.; fc_n=0
    with torch.no_grad():
        for batch in loader:
            b=_move(batch,ctx.device); restored,_=model(b['x'],b['observed'],b['artificial'],b['tod'],b['dow'])
            rec=(restored-b['full']).abs()[b['artificial']]; rec_sum+=float(rec.sum()); rec_n+=int(rec.numel())
            pred=bb.forward_norm_tensor(restored,b['tod'],b['dow']); e=(pred-b['y_norm']).abs()[b['future_valid']]; fc_sum+=float(e.sum()); fc_n+=int(e.numel())
    return rec_sum/max(rec_n,1),fc_sum/max(fc_n,1)

def train_context(ctx,smoke=False):
    seed_all(ctx.seed); c=ctx.cfg['training']['context']; model=build_context_model(ctx); bb=FrozenBackbone(ctx.build_backbone(),ctx.cfg['runtime']['inference_batch_size'])
    tr=_loader(ctx,'train',c['batch_size'],smoke,True); va=_loader(ctx,'val',c['batch_size'],smoke,False)
    opt=torch.optim.AdamW(model.parameters(),lr=float(c['learning_rate']),weight_decay=float(c['weight_decay']))
    epochs=int(c['smoke_epochs'] if smoke else c['max_epochs']); best=math.inf; bad=0; bestp=ctx.artifact_root/'context'/'best.pt'; hist=[]
    for ep in range(1,epochs+1):
        model.train(); ls=0.; nb=0
        for batch in tr:
            b=_move(batch,ctx.device); opt.zero_grad(set_to_none=True)
            restored,_=model(b['x'],b['observed'],b['artificial'],b['tod'],b['dow'])
            rec=masked_l1(restored,b['full'],b['artificial']); pred=bb.forward_norm_tensor(restored,b['tod'],b['dow']); fc=masked_l1(pred,b['y_norm'],b['future_valid'])
            loss=float(c['reconstruction_weight'])*rec+float(c['forecast_weight'])*fc; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(c['gradient_clip'])); opt.step(); ls+=float(loss.detach()); nb+=1
        vr,vf=_run_val_context(model,va,bb,ctx); score=vf; hist.append({"epoch":ep,"train_loss":ls/max(nb,1),"val_reconstruction_mae_norm":vr,"val_forecast_mae_norm":vf})
        log(f"CONTEXT_EPOCH {ep:03d}/{epochs} train={ls/max(nb,1):.6f} val_rec={vr:.6f} val_forecast={vf:.6f}")
        if score<best-1e-6: best=score; bad=0; save_checkpoint(bestp,model,opt,ep,score,_meta(ctx,'context'))
        else: bad+=1
        if not smoke and bad>=int(c['patience']): break
    write_json(ctx.artifact_root/'context'/'history.json',{"best_val_forecast_mae_norm":best,"history":hist})
    log(f"TRAIN_CONTEXT_PASS {bestp}"); return bestp

def _precompute_candidate_assignments(ctx, context, index, split: str, smoke: bool):
    d=CacheSplit(ctx.formal_cache,split); c=ctx.cfg['training']['retrieval']; top_k=int(c['top_k']); requested_bs=int(c['embedding_batch_size'])
    bs=_embedding_micro_batch(ctx,requested_bs)
    M=len(d); inds_all=np.full((M,top_k),-1,np.int32); sim_all=np.full((M,top_k),-1e4,np.float16)
    cutoff=int(np.sort(np.asarray(index.starts,np.int64))[min(top_k-1,len(index.starts)-1)])+int(ctx.cfg['experiment']['history'])
    rows=np.arange(M,dtype=np.int64)
    if smoke:
        rows=_indices(d,True,split=='train',ctx.cfg['experiment']['families'],ctx.seed)
    context.eval(); _cuda_release()
    log(f'CANDIDATE_ASSIGN_PROFILE split={split} requested_batch={requested_bs} initial_micro_batch={bs}')
    done=0
    with torch.inference_mode():
        while done < len(rows):
            cur=min(bs,len(rows)-done); rr=rows[done:done+cur]
            try:
                starts=np.asarray(d.meta.window_start.iloc[rr],np.int64); eligible=starts>=cutoff
                if np.any(eligible):
                    ee=rr[eligible]
                    x=torch.from_numpy(np.asarray(d.x_obs[ee],np.float32).copy()).to(ctx.device)
                    o=torch.from_numpy(np.asarray(d.history_observed[ee],bool).copy()).to(ctx.device)
                    td=torch.from_numpy(np.asarray(d.tod_hist[ee],np.int64).copy()).to(ctx.device)
                    dw=torch.from_numpy(np.asarray(d.dow_hist[ee],np.int64).copy()).to(ctx.device)
                    st=torch.from_numpy(np.asarray(d.meta.window_start.iloc[ee],np.int64)).to(ctx.device)
                    q=context.global_embedding(x,o,td,dw); ii,ss=index.search(q,st,top_k,int(c['search_k']))
                    inds_all[ee]=ii.cpu().numpy().astype(np.int32); sim_all[ee]=ss.cpu().numpy().astype(np.float16)
                    del x,o,td,dw,st,q,ii,ss
                done += cur
                if done % max(100,bs*20) < cur or done>=len(rows):
                    log(f'CANDIDATE_ASSIGN {split} {done}/{len(rows)} micro_batch={bs}')
            except RuntimeError as exc:
                if ctx.device.type=='cuda' and _is_oom(exc) and bs>1:
                    new_bs=max(1,bs//2)
                    log(f'CANDIDATE_ASSIGN_OOM split={split} batch={bs} -> retry_batch={new_bs}')
                    for _name in ('x','o','td','dw','st','q','ii','ss'):
                        if _name in locals():
                            locals()[_name] = None
                    bs=new_bs; _cuda_release(); continue
                raise
    np.save(ctx.artifact_root/'retrieval_index'/f'{split}_candidate_indices.npy',inds_all)
    np.save(ctx.artifact_root/'retrieval_index'/f'{split}_candidate_similarity_f16.npy',sim_all)

def build_retrieval_index(ctx,smoke=False):
    _cuda_release(); context=load_context(ctx); c=ctx.cfg['training']['retrieval']; rows=min(4000,int(c['memory_rows'])) if smoke else int(c['memory_rows'])
    p=LearnedRetrievalIndex.build(ctx,context,ctx.artifact_root/'retrieval_index',rows,int(c['embedding_batch_size'])); index=LearnedRetrievalIndex(ctx,p)
    _precompute_candidate_assignments(ctx,context,index,'train',smoke); _precompute_candidate_assignments(ctx,context,index,'val',smoke)
    log(f"BUILD_RETRIEVAL_INDEX_PASS {p}"); return p

def _load_assignments(ctx, split):
    root=ctx.artifact_root/'retrieval_index'; return np.load(root/f'{split}_candidate_indices.npy',mmap_mode='r'),np.load(root/f'{split}_candidate_similarity_f16.npy',mmap_mode='r')

def _retrieval_batch(ctx,context,index,b,assignments=None):
    with torch.no_grad():
        cr,_=context(b['x'],b['observed'],b['artificial'],b['tod'],b['dow'])
        if assignments is None:
            q=context.global_embedding(b['x'],b['observed'],b['tod'],b['dow']); inds,sim=index.search(q,b['window_start'].long(),ctx.cfg['training']['retrieval']['top_k'],ctx.cfg['training']['retrieval']['search_k'])
        else:
            rows=b['row'].detach().cpu().numpy().astype(np.int64); ii=np.asarray(assignments[0][rows],np.int64); ss=np.asarray(assignments[1][rows],np.float32)
            if np.any(ii<0): raise RuntimeError('Retrieval batch contains rows without causal precomputed candidates')
            inds=torch.from_numpy(ii).to(ctx.device); sim=torch.from_numpy(ss).to(ctx.device)
        cn,cv=index.materialize(inds); candidates=torch.from_numpy(cn).to(ctx.device); cvalid=torch.from_numpy(cv).to(ctx.device)
    return cr,candidates,cvalid,sim

def _val_retrieval(model,loader,context,index,bb,ctx,assignments):
    model.eval(); rs=rn=fs=fn=0.
    with torch.no_grad():
        for batch in loader:
            b=_move(batch,ctx.device); cr,cn,cv,sim=_retrieval_batch(ctx,context,index,b,assignments)
            restored,_=model(b['x'],b['observed'],b['artificial'],cr,cn,cv,sim,context.encoder.a_norm)
            rr=(restored-b['full']).abs()[b['artificial']]; pp=bb.forward_norm_tensor(restored,b['tod'],b['dow']); ff=(pp-b['y_norm']).abs()[b['future_valid']]
            rs+=float(rr.sum()); rn+=int(rr.numel()); fs+=float(ff.sum()); fn+=int(ff.numel())
    return rs/max(rn,1),fs/max(fn,1)

def train_retrieval(ctx,smoke=False):
    seed_all(ctx.seed+11); c=ctx.cfg['training']['retrieval']; context=load_context(ctx); context.eval(); index=LearnedRetrievalIndex(ctx,ctx.artifact_root/'retrieval_index'); model=build_retrieval_model(ctx); bb=FrozenBackbone(ctx.build_backbone(),ctx.cfg['runtime']['inference_batch_size'])
    cutoff=int(np.sort(np.asarray(index.starts,np.int64))[min(int(c['top_k'])-1,len(index.starts)-1)])+int(ctx.cfg['experiment']['history']); tr=_loader(ctx,'train',c['batch_size'],smoke,True,cutoff); va=_loader(ctx,'val',c['batch_size'],smoke,False); train_assign=_load_assignments(ctx,'train'); val_assign=_load_assignments(ctx,'val'); opt=torch.optim.AdamW(model.parameters(),lr=float(c['learning_rate']),weight_decay=float(c['weight_decay']))
    epochs=int(c['smoke_epochs'] if smoke else c['max_epochs']); best=math.inf; bad=0; bestp=ctx.artifact_root/'retrieval'/'best.pt'; hist=[]
    for ep in range(1,epochs+1):
        model.train(); ls=0.; nb=0
        for batch in tr:
            b=_move(batch,ctx.device); cr,cn,cv,sim=_retrieval_batch(ctx,context,index,b,train_assign); opt.zero_grad(set_to_none=True)
            restored,_=model(b['x'],b['observed'],b['artificial'],cr,cn,cv,sim,context.encoder.a_norm)
            rec=masked_l1(restored,b['full'],b['artificial']); pred=bb.forward_norm_tensor(restored,b['tod'],b['dow']); fc=masked_l1(pred,b['y_norm'],b['future_valid'])
            loss=float(c['reconstruction_weight'])*rec+float(c['forecast_weight'])*fc; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(c['gradient_clip'])); opt.step(); ls+=float(loss.detach()); nb+=1
        vr,vf=_val_retrieval(model,va,context,index,bb,ctx,val_assign); hist.append({"epoch":ep,"train_loss":ls/max(nb,1),"val_reconstruction_mae_norm":vr,"val_forecast_mae_norm":vf}); log(f"RETRIEVAL_EPOCH {ep:03d}/{epochs} train={ls/max(nb,1):.6f} val_rec={vr:.6f} val_forecast={vf:.6f}")
        if vf<best-1e-6: best=vf; bad=0; save_checkpoint(bestp,model,opt,ep,vf,_meta(ctx,'retrieval'))
        else: bad+=1
        if not smoke and bad>=int(c['patience']): break
    write_json(ctx.artifact_root/'retrieval'/'history.json',{"best_val_forecast_mae_norm":best,"history":hist}); log(f"TRAIN_RETRIEVAL_PASS {bestp}"); return bestp

def _val_backbone_aware(model,loader,context,bb,ctx):
    model.eval(); rs=rn=fs=fn=0.
    with torch.no_grad():
        for batch in loader:
            b=_move(batch,ctx.device); cr,_=context(b['x'],b['observed'],b['artificial'],b['tod'],b['dow']); base=bb.predict_norm_tensor(b['x'],b['tod'],b['dow']); restored,_=model(b['x'],b['observed'],b['artificial'],b['tod'],b['dow'],cr,base)
            rr=(restored-b['full']).abs()[b['artificial']]; pp=bb.forward_norm_tensor(restored,b['tod'],b['dow']); ff=(pp-b['y_norm']).abs()[b['future_valid']]
            rs+=float(rr.sum()); rn+=int(rr.numel()); fs+=float(ff.sum()); fn+=int(ff.numel())
    return rs/max(rn,1),fs/max(fn,1)

def train_backbone_aware(ctx,smoke=False):
    seed_all(ctx.seed+23); c=ctx.cfg['training']['backbone_aware']; context=load_context(ctx); context.eval(); model=build_backbone_aware_model(ctx); bb=FrozenBackbone(ctx.build_backbone(),ctx.cfg['runtime']['inference_batch_size'])
    tr=_loader(ctx,'train',c['batch_size'],smoke,True); va=_loader(ctx,'val',c['batch_size'],smoke,False); opt=torch.optim.AdamW(model.parameters(),lr=float(c['learning_rate']),weight_decay=float(c['weight_decay']))
    epochs=int(c['smoke_epochs'] if smoke else c['max_epochs']); best=math.inf; bad=0; bestp=ctx.artifact_root/'backbone_aware'/'best.pt'; hist=[]
    for ep in range(1,epochs+1):
        model.train(); ls=0.; nb=0
        for batch in tr:
            b=_move(batch,ctx.device)
            with torch.no_grad(): cr,_=context(b['x'],b['observed'],b['artificial'],b['tod'],b['dow']); base=bb.predict_norm_tensor(b['x'],b['tod'],b['dow'])
            opt.zero_grad(set_to_none=True); restored,delta=model(b['x'],b['observed'],b['artificial'],b['tod'],b['dow'],cr,base)
            rec=masked_l1(restored,b['full'],b['artificial']); pred=bb.forward_norm_tensor(restored,b['tod'],b['dow']); fc=masked_l1(pred,b['y_norm'],b['future_valid']); reg=(delta[b['artificial']]**2).mean() if torch.any(b['artificial']) else delta.pow(2).mean()*0
            loss=float(c['reconstruction_weight'])*rec+float(c['forecast_weight'])*fc+float(c['residual_weight'])*reg; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(c['gradient_clip'])); opt.step(); ls+=float(loss.detach()); nb+=1
        vr,vf=_val_backbone_aware(model,va,context,bb,ctx); hist.append({"epoch":ep,"train_loss":ls/max(nb,1),"val_reconstruction_mae_norm":vr,"val_forecast_mae_norm":vf}); log(f"BACKBONE_AWARE_EPOCH {ep:03d}/{epochs} train={ls/max(nb,1):.6f} val_rec={vr:.6f} val_forecast={vf:.6f}")
        if vf<best-1e-6: best=vf; bad=0; save_checkpoint(bestp,model,opt,ep,vf,_meta(ctx,'backbone_aware'))
        else: bad+=1
        if not smoke and bad>=int(c['patience']): break
    write_json(ctx.artifact_root/'backbone_aware'/'history.json',{"best_val_forecast_mae_norm":best,"history":hist}); log(f"TRAIN_BACKBONE_AWARE_PASS {bestp}"); return bestp
