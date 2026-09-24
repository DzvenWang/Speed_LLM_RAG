#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,json,numpy as np,torch
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe'); ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml,write_json,stamp,seed_all
from presure.contract import cache_dir
from presure.data import CacheSplit
from presure.metrics import metric_bundle,weighted_mae_loss,per_query_mae_torch
from presure.features import NODE_SENSE_DIM,node_sensing_from_features
from presure.models import NodeM1Model,NodeUsefulnessNet,PairwiseSupportRouter,SensingAwareSafeGate
from presure.pipeline import load_artifacts,run_dir,load_m1,load_m3,load_m4


def t(a,dev,dtype=torch.float32): return torch.as_tensor(np.asarray(a),device=dev,dtype=dtype)
def batches(n,bs,shuffle=False,seed=0):
    ids=np.arange(n)
    if shuffle: np.random.default_rng(seed).shuffle(ids)
    for a in range(0,n,bs): yield ids[a:a+bs]
def save_latest(cfg,var,stage,rd,ck):
    p=Path(cfg['project']['exp_root'])/var/cfg['experiment']['subgraph']/stage; p.mkdir(parents=True,exist_ok=True)
    write_json(p/'LATEST.json',{'run_dir':str(rd),'checkpoint':str(ck),'variant':var,'stage':stage,'implementation_revision':cfg.get('implementation_revision')})
def metrics(cfg,pred,ds): return metric_bundle(cfg,pred,np.asarray(ds.y),np.asarray(ds.valid,bool),np.asarray(ds.direct,bool),np.asarray(ds.support,bool))
def objective(cfg,m):
    w=cfg['oracle']['objective_weights']; return float(w['overall'])*float(m['overall_MAE'])+float(w['masked'])*float(m['masked_MAE'])+float(w['support'])*float(m['support_MAE'])
def rel(base,new,key):
    b=float(base[key]); return (b-float(new[key]))/b if b else 0.0
def masks(ds,ids,dev): return t(ds.valid[ids],dev,torch.bool),t(ds.direct[ids],dev,torch.bool),t(ds.support[ids],dev,torch.bool)


def _feature_mmaps(art): return np.load(art/'train_node_features.npy',mmap_mode='r'),np.load(art/'val_node_features.npy',mmap_mode='r')
def _sense(arr): return np.asarray(arr[...,-NODE_SENSE_DIM:],np.float32)


def build_node_embeddings(enc,features,out_path,dev,bs=32):
    shape=(len(features),features.shape[1],enc.demb); mm=np.lib.format.open_memmap(out_path,mode='w+',dtype=np.float16,shape=shape); enc.eval()
    with torch.no_grad():
        for ids in batches(len(features),bs):
            mm[ids]=enc(t(np.asarray(features[ids],np.float32),dev)).cpu().numpy().astype(np.float16)
            if ids[0] % max(bs*50,1)==0: print(f'NODE EMBEDDINGS {Path(out_path).name} {ids[0]}/{len(features)}',flush=True)
    del mm


def ensure_embeddings(cfg,stage,art,m1,dev,force=False):
    trf,vaf=_feature_mmaps(art); p1=art/'train_node_embeddings_v05.npy'; p2=art/'val_node_embeddings_v05.npy'
    if force or not p1.exists() or not p2.exists():
        build_node_embeddings(m1.encoder,trf,p1,dev,int(cfg['node_features']['embedding_batch_size'])); build_node_embeddings(m1.encoder,vaf,p2,dev,int(cfg['node_features']['embedding_batch_size']))
    return np.load(p1,mmap_mode='r'),np.load(p2,mmap_mode='r')


def global_pool(z): return np.asarray(z,np.float32).mean(1)


def embedding_retrieval_correction(zmem,zq,cids,res,k,temp=0.12,batch=128):
    out=[]; zm=zmem/(np.linalg.norm(zmem,axis=1,keepdims=True)+1e-8); zqq=zq/(np.linalg.norm(zq,axis=1,keepdims=True)+1e-8)
    for a in range(0,len(zq),batch):
        ids=cids[a:a+batch]; sim=np.einsum('bd,bkd->bk',zqq[a:a+batch],zm[ids]); kk=min(int(k),sim.shape[1]); top=np.argpartition(-sim,kth=kk-1,axis=1)[:,:kk]
        ss=np.take_along_axis(sim,top,1); ii=np.take_along_axis(ids,top,1); o=np.argsort(-ss,axis=1); ss=np.take_along_axis(ss,o,1); ii=np.take_along_axis(ii,o,1)
        zz=ss/max(float(temp),1e-4); zz-=zz.max(1,keepdims=True); w=np.exp(zz); w/=w.sum(1,keepdims=True); rr=np.asarray(res[ii],np.float32); out.append((rr*w[:,:,None,None]).sum(1))
    return np.concatenate(out).astype(np.float32)


def select_global_alpha(cfg,base,y,valid,direct,support,corr):
    grid=[0.0]+[float(x) for x in cfg['oracle']['alpha_grid']]; best=(1e99,0.0)
    for a in grid:
        p=base+a*corr; m=metric_bundle(cfg,p,y,valid,direct,support); sc=objective(cfg,m)
        if sc<best[0]: best=(sc,a)
    return float(best[1])


def node_utility_targets(base,y,valid,rr,alpha_grid):
    """Train-only candidate utility, relative utility, and best positive scale.

    base/y/valid [B,P,N], rr [B,K,P,N]. Evaluation-only direct/support masks
    are intentionally excluded from supervision.
    """
    vv=np.asarray(valid,bool); den=vv.sum(1).clip(min=1); eb=(np.abs(base-y)*vv).sum(1)/den
    best=np.full((rr.shape[0],rr.shape[1],rr.shape[3]),np.inf,np.float32)
    best_alpha=np.zeros(best.shape,np.float32)
    for a in alpha_grid:
        ec=(np.abs(base[:,None]+float(a)*rr-y[:,None])*vv[:,None]).sum(2)/den[:,None]
        take=ec<best; best=np.where(take,ec,best).astype(np.float32); best_alpha=np.where(take,float(a),best_alpha).astype(np.float32)
    u=eb[:,None]-best; relative=u/(eb[:,None]+0.10)
    return u.astype(np.float32),relative.astype(np.float32),best_alpha.astype(np.float32)


def utility_labels(relative,pos_thr,neg_thr):
    lab=np.ones(relative.shape,np.int64); lab[relative>float(pos_thr)]=2; lab[relative<-float(neg_thr)]=0; return lab


def balanced_hard_mask(relative,labels,npos,nhard,nneutral):
    B,K,N=relative.shape; mask=np.zeros((B,K,N),bool)
    for b in range(B):
        for n in range(N):
            u=relative[b,:,n]; l=labels[b,:,n]
            p=np.flatnonzero(l==2); h=np.flatnonzero(l==0); z=np.flatnonzero(l==1)
            if len(p): mask[b,p[np.argsort(-u[p])[:npos]],n]=True
            if len(h): mask[b,h[np.argsort(u[h])[:nhard]],n]=True
            if len(z): mask[b,z[np.argsort(np.abs(u[z]))[:nneutral]],n]=True
    return mask


def pairwise_rank_loss(score,target,pos_thr,margin):
    pos_idx=target.argmax(1,keepdim=True); neg_idx=target.argmin(1,keepdim=True)
    sp=torch.gather(score,1,pos_idx).squeeze(1); sn=torch.gather(score,1,neg_idx).squeeze(1)
    good=(target.max(1).values>float(pos_thr)) & (target.max(1).values>target.min(1).values+1e-6)
    if not torch.any(good): return score.sum()*0
    return torch.relu(float(margin)-sp[good]+sn[good]).mean()


def m3_forward(net,zmem,zq,node_mem,node_q,cand,ids,dev,k=None):
    ci=np.asarray(cand['indices'][ids],np.int64); sim=np.asarray(cand['scores'][ids],np.float32)
    if k is not None: ci=ci[:,:int(k)]; sim=sim[:,:int(k)]
    q=t(np.asarray(zq[ids],np.float32),dev); c=t(np.asarray(zmem[ci],np.float32),dev); qs=t(_sense(node_q[ids]),dev); cs=t(_sense(node_mem[ci]),dev)
    logits,score,scale=net(q,c,t(sim,dev),qs,cs); return ci,logits,score,scale


def m3_raw_correction(net,zmem,zq,node_mem,node_q,cand,res,ids,cfg,dev,hard_threshold=True):
    ci,logits,score,scale=m3_forward(net,zmem,zq,node_mem,node_q,cand,ids,dev,int(cfg['node_routing']['utility_candidate_k']))
    p=torch.softmax(logits,-1)[...,2]; combined=score+torch.log(p+1e-6); best=combined.argmax(1)
    pbest=torch.gather(p,1,best[:,None,:]).squeeze(1); sbest=torch.gather(scale,1,best[:,None,:]).squeeze(1)
    rr=t(np.asarray(res[ci],np.float32),dev).permute(0,3,1,2)
    sel=torch.gather(rr,2,best[:,:,None,None].expand(-1,-1,1,rr.shape[-1])).squeeze(2)
    if hard_threshold:
        keep=pbest>=float(cfg['model']['m3']['use_threshold']); sel=sel*keep[:,:,None]; sbest=sbest*keep
    corr=(sbest[:,:,None]*sel).permute(0,2,1).contiguous()
    return corr,pbest,sbest


def m3_select_for_router(net,zmem,zq,node_mem,node_q,cand,res,ids,cfg,dev):
    ci,logits,score,scale=m3_forward(net,zmem,zq,node_mem,node_q,cand,ids,dev,int(cfg['node_routing']['utility_candidate_k']))
    p=torch.softmax(logits,-1)[...,2]; combined=score+torch.log(p+1e-6); L=min(int(cfg['node_routing']['m4_top_k']),combined.shape[1])
    vals,loc=torch.topk(combined,L,dim=1)
    loc_bn=loc.permute(0,2,1).contiguous(); p_bn=p.permute(0,2,1); psel=torch.gather(p_bn,2,loc_bn)
    # M3 ranking was substantially better calibrated than the raw beneficial
    # probability in v0.5. Give M4 relative Top-K rank confidence, while the
    # eligibility mask remains based on the deployable beneficial probability.
    conf=torch.softmax(vals,dim=1).permute(0,2,1).contiguous()
    s_bn=scale.permute(0,2,1); m3_scale=torch.gather(s_bn,2,loc_bn)
    candidate_mask=psel>=float(cfg['model']['m3'].get('router_use_threshold',cfg['model']['m3']['use_threshold']))
    cids=t(ci,dev,torch.long); global_bn=torch.gather(cids[:,None,:].expand(-1,loc_bn.shape[1],-1),2,loc_bn)
    c_all=t(np.asarray(zmem[ci],np.float32),dev).permute(0,2,1,3); D=c_all.shape[-1]; csel=torch.gather(c_all,2,loc_bn[:,:,:,None].expand(-1,-1,-1,D))
    cs_all=t(_sense(node_mem[ci]),dev).permute(0,2,1,3); S=cs_all.shape[-1]; cssel=torch.gather(cs_all,2,loc_bn[:,:,:,None].expand(-1,-1,-1,S))
    rr_all=t(np.asarray(res[ci],np.float32),dev).permute(0,3,1,2); P=rr_all.shape[-1]; rrsel=torch.gather(rr_all,2,loc_bn[:,:,:,None].expand(-1,-1,-1,P))
    return global_bn,csel,cssel,conf,m3_scale,candidate_mask,rrsel


def m3_select_for_pairwise_router(net,zmem,zq,node_mem,node_q,cand,res,ids,cfg,dev):
    """M3 Top-K plus full deployable query/support node contexts for v0.7 M4.

    Historical support contexts are indexed from Train memory only. The complete
    node feature contains incomplete-history state, sensing mask/context, frozen
    PatchSTG forecast trajectory and time covariates; no current-query future is used.
    """
    gid,csel,cssel,conf,m3_scale,candidate_mask,rr=m3_select_for_router(net,zmem,zq,node_mem,node_q,cand,res,ids,cfg,dev)
    qctx=t(np.asarray(node_q[ids],np.float32),dev)
    gi=gid.detach().cpu().numpy().astype(np.int64,copy=False)
    N=gi.shape[1]; ni=np.arange(N,dtype=np.int64)[None,:,None]
    cctx_np=np.asarray(node_mem)[gi,ni,:].astype(np.float32,copy=False)
    cctx=t(cctx_np,dev)
    return gid,csel,cssel,conf,m3_scale,candidate_mask,rr,qctx,cctx


def _binary_auroc_from_hist(pos_hist,neg_hist):
    p=float(np.sum(pos_hist)); n=float(np.sum(neg_hist))
    if p<=0 or n<=0: return None
    neg_before=np.cumsum(neg_hist)-neg_hist
    return float(np.sum(pos_hist*(neg_before+0.5*neg_hist))/(p*n))


def m3_support_diagnostics(net,zmem,zq,node_mem,node_q,cand,res,base,ds,cfg,dev,seed,n_queries=None):
    """Outcome-grounded support diagnostics. Futures are labels only, never model inputs."""
    n=min(len(ds),int(n_queries if n_queries is not None else cfg['model']['m3'].get('diagnostic_queries',1000))); rng=np.random.default_rng(seed+503)
    sub=np.sort(rng.choice(len(ds),size=n,replace=False)); K=min(int(cfg['node_routing']['utility_candidate_k']),cand['indices'].shape[1]); c=cfg['model']['m3']; grid=[float(x) for x in cfg['oracle']['alpha_grid']]
    tp=fp=fn=0; top1_hit=top4_hit=has_pos_count=0; pair_ok=pair_count=0; bins=int(cfg.get('diagnostics',{}).get('auroc_bins',200)); ph=np.zeros(bins,np.int64); nh=np.zeros(bins,np.int64); scale_sum=0.0;scale_count=0
    net.eval()
    with torch.no_grad():
        for subids in batches(len(sub),16):
            ids=sub[subids]; ci=np.asarray(cand['indices'][ids,:K],np.int64); rr=np.asarray(res[ci],np.float32)
            _,urel,astar=node_utility_targets(np.asarray(base[ids],np.float32),np.asarray(ds.y[ids],np.float32),np.asarray(ds.valid[ids],bool),rr,grid); labels=utility_labels(urel,float(c['positive_relative_threshold']),float(c['negative_relative_threshold'])); beneficial=labels==2
            _,logits,score,scale=m3_forward(net,zmem,zq,node_mem,node_q,cand,ids,dev,K); p=torch.softmax(logits,-1)[...,2].cpu().numpy(); sc=score.cpu().numpy(); sp=scale.cpu().numpy(); combined=sc+np.log(p+1e-6)
            pred=p>=float(c['use_threshold']); tp+=int((pred&beneficial).sum()); fp+=int((pred&~beneficial).sum()); fn+=int((~pred&beneficial).sum())
            bi=np.clip((p*(bins-1)).astype(np.int64),0,bins-1); ph+=np.bincount(bi[beneficial],minlength=bins); nh+=np.bincount(bi[~beneficial],minlength=bins)
            has=beneficial.any(1); has_pos_count+=int(has.sum()); top1=combined.argmax(1); top1lab=np.take_along_axis(beneficial,top1[:,None,:],axis=1).squeeze(1); top1_hit+=int((top1lab&has).sum())
            L=min(4,K); loc=np.argpartition(-combined,kth=L-1,axis=1)[:,:L,:]; top4=np.take_along_axis(beneficial,loc,axis=1).any(1); top4_hit+=int((top4&has).sum())
            best=urel.argmax(1); worst=urel.argmin(1); sb=np.take_along_axis(sc,best[:,None,:],axis=1).squeeze(1); sw=np.take_along_axis(sc,worst[:,None,:],axis=1).squeeze(1); sep=urel.max(1)>urel.min(1)+1e-6; pair_ok+=int(((sb>sw)&sep).sum()); pair_count+=int(sep.sum())
            if beneficial.any(): scale_sum+=float(np.abs(sp[beneficial]-astar[beneficial]).sum());scale_count+=int(beneficial.sum())
    return {'queries':int(n),'beneficial_precision':float(tp/max(tp+fp,1)),'beneficial_recall':float(tp/max(tp+fn,1)),'beneficial_auroc':_binary_auroc_from_hist(ph,nh),'top1_beneficial_hit_rate':float(top1_hit/max(has_pos_count,1)),'top4_beneficial_hit_rate':float(top4_hit/max(has_pos_count,1)),'pairwise_ranking_accuracy':float(pair_ok/max(pair_count,1)),'scale_mae_on_beneficial':float(scale_sum/max(scale_count,1)) if scale_count else None,'beneficial_cases_with_candidate':int(has_pos_count)}


def m3_train_diagnostics(net,zmem,node_mem,cand,res,base,ds,cfg,dev,seed):
    return m3_support_diagnostics(net,zmem,zmem,node_mem,node_mem,cand,res,base,ds,cfg,dev,seed)

def m4_soft_teacher_targets(base,y,valid,rr,candidate_mask,alpha_grid,temperature=0.12,mass_temperature=0.05,min_abs_gain=0.01,gain_floor=0.10,utility_clip=1.5):
    """Train-only soft utility teacher over M3-selected supports.

    Returns a probability distribution over L supports + NO-SUPPORT for every
    node x horizon cell, plus candidate-wise optimal residual contributions and
    the resulting oracle-weighted mixture correction. Validation futures are
    never used here during training.
    """
    bb=base.permute(0,2,1); yy=y.permute(0,2,1); vv=valid.bool().permute(0,2,1)
    B,N,P=bb.shape; L=rr.shape[2]; base_err=torch.abs(bb-yy)
    best_err=torch.full_like(rr,float('inf')); best_alpha=torch.zeros_like(rr)
    cm=candidate_mask.bool()
    for a in alpha_grid:
        err=torch.abs(bb[:,:,None,:]+float(a)*rr-yy[:,:,None,:])
        take=err<best_err
        best_err=torch.where(take,err,best_err)
        best_alpha=torch.where(take,torch.full_like(best_alpha,float(a)),best_alpha)
    gain=base_err[:,:,None,:]-best_err
    denom=base_err[:,:,None,:]+float(gain_floor)
    rel=(gain-float(min_abs_gain))/denom
    rel=torch.clamp(rel,-float(utility_clip),float(utility_clip))
    rel=rel.masked_fill(~cm[:,:,:,None],-1e4)
    # Separate intervention strength from support identity so multiple near-zero
    # candidates cannot collectively overwhelm the NO-SUPPORT option.
    cond=torch.softmax(rel/max(float(temperature),1e-4),2)
    best_rel=rel.max(2).values
    target_mass=1.0-torch.exp(-torch.relu(best_rel)/max(float(mass_temperature),1e-4))
    target_mass=torch.clamp(target_mass,0.0,1.0)
    support_probs=cond*target_mass[:,:,None,:]
    noop=(1.0-target_mass)[:,:,None,:]
    probs=torch.cat([support_probs,noop],2)
    target_delta=best_alpha*rr
    target_corr=(support_probs*target_delta).sum(2)
    hard=probs.argmax(2)
    entropy=-(probs*torch.log(probs.clamp_min(1e-8))).sum(2)
    return probs,target_delta,target_corr,target_mass,hard,rel,entropy


def m4_soft_distill_loss(slog,scale,noop,rr,corr,mass,teacher_probs,target_delta,target_corr,target_mass,valid):
    """Soft route + transported-residual distillation losses."""
    L=slog.shape[2]
    pred_logits=torch.cat([slog,noop[:,:,None,:]],2)
    logp=torch.log_softmax(pred_logits,2)
    vm=valid.bool().permute(0,2,1)
    soft_ce=-(teacher_probs*logp).sum(2)
    soft_ce=soft_ce[vm].mean() if torch.any(vm) else pred_logits.sum()*0
    pred_delta=scale*rr
    delta_err=torch.nn.functional.smooth_l1_loss(pred_delta,target_delta,reduction='none')
    w=teacher_probs[:,:,:L,:]*vm[:,:,None,:].to(delta_err.dtype)
    delta_loss=(delta_err*w).sum()/w.sum().clamp_min(1.0)
    corr_bn=corr.permute(0,2,1)
    corr_err=torch.nn.functional.smooth_l1_loss(corr_bn,target_corr,reduction='none')
    corr_loss=corr_err[vm].mean() if torch.any(vm) else corr_bn.sum()*0
    mass_bn=mass.permute(0,2,1)
    mass_mae=torch.abs(mass_bn-target_mass)[vm].mean() if torch.any(vm) else mass_bn.sum()*0
    hard=teacher_probs.argmax(2); pred=pred_logits.argmax(2)
    acc=(pred[vm]==hard[vm]).float().mean() if torch.any(vm) else torch.tensor(0.0,device=pred_logits.device)
    ent=(-(teacher_probs*torch.log(teacher_probs.clamp_min(1e-8))).sum(2))[vm].mean() if torch.any(vm) else torch.tensor(0.0,device=pred_logits.device)
    return soft_ce,delta_loss,corr_loss,mass_mae,acc,ent


def m4_pairwise_utility_rank_loss(slog,teacher_rel,candidate_mask,valid,margin=0.10,min_gap=0.02):
    """Train-only horizon-local pairwise utility ranking supervision.

    SoftMix gives the desired distribution, while this auxiliary loss forces the
    student representation to preserve which candidate is better than another.
    """
    cm=candidate_mask.bool()[:,:,:,None].expand_as(slog)
    vm=valid.bool().permute(0,2,1)[:,:,None,:].expand_as(slog)
    use=cm & vm
    hi=teacher_rel.masked_fill(~use,-1e6).argmax(2)
    lo=teacher_rel.masked_fill(~use,1e6).argmin(2)
    sb=torch.gather(slog,2,hi[:,:,None,:]).squeeze(2); sw=torch.gather(slog,2,lo[:,:,None,:]).squeeze(2)
    tb=torch.gather(teacher_rel,2,hi[:,:,None,:]).squeeze(2); tw=torch.gather(teacher_rel,2,lo[:,:,None,:]).squeeze(2)
    enough=(use.sum(2)>=2) & (tb-tw>float(min_gap)) & valid.bool().permute(0,2,1)
    if not torch.any(enough):
        z=slog.sum()*0; return z,torch.tensor(0.0,device=slog.device)
    loss=torch.relu(float(margin)-sb[enough]+sw[enough]).mean()
    acc=(sb[enough]>sw[enough]).float().mean()
    return loss,acc


def _m4_forward_selected(router,q,qs,qctx,csel,cssel,cctx,conf,candidate_mask,rr,b,speed_mean,speed_std,cfg):
    bnorm=(b-float(speed_mean))/max(float(speed_std),1e-6); rrnorm=rr/max(float(speed_std),1e-6)
    H=int(cfg['experiment']['history']); P=int(cfg['experiment']['horizon']); start=4*H
    support_base_norm=cctx[...,start:start+P]
    if support_base_norm.shape[-1]!=P: raise RuntimeError(f'Cannot recover support PatchSTG forecast slice from node context {tuple(cctx.shape)}')
    slog,scale,noop=router(q,csel,qctx,cctx,qs,cssel,rrnorm,bnorm.permute(0,2,1),support_base_norm,conf)
    slog=slog.masked_fill(~candidate_mask[:,:,:,None],-30.0)
    logits=torch.cat([slog,noop[:,:,None,:]],2); w=torch.softmax(logits,2); ws=w[:,:,:rr.shape[2],:]; no=w[:,:,-1,:]
    transported=scale*rr; corr_bn=(ws*transported).sum(2); mass=1.0-no; den=mass.clamp_min(1e-6); var=(ws*(transported-corr_bn[:,:,None,:])**2).sum(2)/den; dis=torch.sqrt(var.clamp_min(0))
    return corr_bn.permute(0,2,1).contiguous(),mass.permute(0,2,1).contiguous(),dis.permute(0,2,1).contiguous(),slog,scale,noop


def m4_local_correction(router,m3,zmem,zq,node_mem,node_q,cand,res,ids,base,speed_mean,speed_std,cfg,dev):
    _,csel,cssel,conf,m3scale,cmask,rr,qctx,cctx=m3_select_for_pairwise_router(m3,zmem,zq,node_mem,node_q,cand,res,ids,cfg,dev)
    q=t(np.asarray(zq[ids],np.float32),dev); qs=t(_sense(node_q[ids]),dev); b=t(np.asarray(base[ids],np.float32),dev)
    corr,mass,dis,_,_,_=_m4_forward_selected(router,q,qs,qctx,csel,cssel,cctx,conf,cmask,rr,b,speed_mean,speed_std,cfg)
    return corr,mass,dis


def local_harm_loss(pred,base,y,valid):
    e=torch.relu(torch.abs(pred-y)-torch.abs(base-y)); m=valid.bool(); return e[m].mean() if torch.any(m) else pred.sum()*0


def masked_local_harm_loss(pred,base,y,valid,direct):
    m=valid.bool() & direct.bool()[:,None,:]; e=torch.relu(torch.abs(pred-y)-torch.abs(base-y)); return e[m].mean() if torch.any(m) else pred.sum()*0


def continuous_gate_teacher(base,corr,y,valid,min_abs_gain=0.01,gain_floor=0.10):
    """Continuous Train-only local intervention target in [0,1].

    The target is the optimal scalar along the frozen M4 correction direction,
    clipped to [0,1]. Cells with negligible attainable gain are assigned zero.
    """
    eps=1e-6
    safe_corr=torch.where(torch.abs(corr)>eps,corr,torch.ones_like(corr))
    ratio=(y-base)/safe_corr
    ratio=torch.where(torch.abs(corr)>eps,ratio,torch.zeros_like(ratio))
    gstar=torch.clamp(ratio,0.0,1.0)
    best=base+gstar*corr
    base_err=torch.abs(base-y); best_err=torch.abs(best-y)
    gain=base_err-best_err
    rel_gain=torch.clamp(gain/(base_err+float(gain_floor)),0.0,1.0)
    keep=(gain>float(min_abs_gain)) & valid.bool() & (torch.abs(corr)>eps)
    target=torch.where(keep,gstar,torch.zeros_like(gstar))
    return target,gain,rel_gain


def continuous_gate_loss(g,target,rel_gain,valid,gain_weight=2.0):
    m=valid.bool()
    if not torch.any(m): return g.sum()*0
    err=torch.nn.functional.smooth_l1_loss(g,target,reduction='none')
    w=1.0+float(gain_weight)*rel_gain
    return (err[m]*w[m]).sum()/w[m].sum().clamp_min(1.0)


def _hist_auroc(scores,labels,mask,bins=200):
    s=np.asarray(scores,np.float32)[mask]; y=np.asarray(labels,bool)[mask]
    if s.size==0 or y.sum()==0 or (~y).sum()==0: return None
    bi=np.clip((s*(bins-1)).astype(np.int64),0,bins-1)
    ph=np.bincount(bi[y],minlength=bins); nh=np.bincount(bi[~y],minlength=bins)
    return _binary_auroc_from_hist(ph,nh)


def run_m0(cfg,stage,seed,art,tr,va,rd,dev):
    p=np.asarray(np.load(art/'val_base_pred.npy',mmap_mode='r'),np.float32); m=metrics(cfg,p,va); np.save(rd/'val_predictions.npy',p); write_json(rd/'summary.json',{'variant':'M0','stage':stage,'seed':seed,'validation':m,'description':'Frozen PatchSTG','test_evaluated':False}); print(f'M0 overall={m["overall_MAE"]:.6f} masked={m["masked_MAE"]:.6f} support={m["support_MAE"]:.6f}',flush=True)


def train_m1(cfg,stage,seed,art,tr,va,rd,dev):
    ft,fv=_feature_mmaps(art); bt=np.load(art/'train_base_pred.npy',mmap_mode='r'); bv=np.load(art/'val_base_pred.npy',mmap_mode='r'); c=cfg['model']['m1']; args=dict(din=ft.shape[-1],N=ft.shape[1],P=bt.shape[1],hidden=int(c['hidden']),demb=int(c['embedding_dim']),node_dim=int(c['node_embedding_dim']),dropout=float(c['dropout']))
    model=NodeM1Model(**args).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=float(c['lr']),weight_decay=float(c['weight_decay'])); best=1e99; wait=0; bestp=rd/'best.pt'
    for ep in range(1,int(c['epochs'])+1):
        model.train(); ls=[]
        for ids in batches(len(tr),int(c['batch_size']),True,seed+ep):
            corr,_=model(t(np.asarray(ft[ids],np.float32),dev)); b=t(bt[ids],dev); y=t(tr.y[ids],dev); valid,direct,support=masks(tr,ids,dev); pred=b+corr; loss=weighted_mae_loss(pred,y,valid,direct,support,cfg['loss'])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); ls.append(float(loss.detach()))
        model.eval(); pv=[]
        with torch.no_grad():
            for ids in batches(len(va),64): corr,_=model(t(np.asarray(fv[ids],np.float32),dev)); pv.append((t(bv[ids],dev)+corr).cpu().numpy())
        pv=np.concatenate(pv); vm=metrics(cfg,pv,va); sc=objective(cfg,vm); print(f'M1-NODE epoch={ep:03d} loss={np.mean(ls):.5f} val_obj={sc:.5f} overall={vm["overall_MAE"]:.5f} masked={vm["masked_MAE"]:.5f} support={vm["support_MAE"]:.5f}',flush=True)
        if sc<best: best=sc;wait=0;torch.save({'state_dict':model.state_dict(),'model_args':args,'model_type':'node_sensing_m1_v04','epoch':ep,'val_objective':sc},bestp)
        else: wait+=1
        if wait>=int(c['patience']): break
    ck=torch.load(bestp,map_location=dev,weights_only=False); model.load_state_dict(ck['state_dict']); model.eval(); pv=[]
    with torch.no_grad():
        for ids in batches(len(va),64): corr,_=model(t(np.asarray(fv[ids],np.float32),dev)); pv.append((t(bv[ids],dev)+corr).cpu().numpy())
    pv=np.concatenate(pv); bm=metrics(cfg,np.asarray(bv,np.float32),va); vm=metrics(cfg,pv,va); np.save(rd/'val_predictions.npy',pv.astype(np.float32)); write_json(rd/'summary.json',{'variant':'M1','stage':stage,'seed':seed,'validation':vm,'base_validation':bm,'relative_improvement_over_M0':{'overall':rel(bm,vm,'overall_MAE'),'masked':rel(bm,vm,'masked_MAE'),'support':rel(bm,vm,'support_MAE')},'best_epoch':ck['epoch'],'description':'Sensing-conditioned node-local encoder; deployable inputs exclude direct/support evaluation masks','test_evaluated':False}); save_latest(cfg,'M1',stage,rd,bestp)


def run_m2(cfg,stage,seed,art,tr,va,rd,dev):
    m1,_,_=load_m1(cfg,stage,dev); zt,zv=ensure_embeddings(cfg,stage,art,m1,dev,force=True); gt,gv=global_pool(zt),global_pool(zv); res=np.load(art/'train_residual.npy',mmap_mode='r'); tc=np.load(art/'train_candidates.npz'); vc=np.load(art/'val_candidates.npz'); bt=np.load(art/'train_base_pred.npy',mmap_mode='r'); bv=np.load(art/'val_base_pred.npy',mmap_mode='r'); rng=np.random.default_rng(seed+21); nq=min(len(tr),int(cfg['node_routing']['train_calibration_queries'])); sub=np.sort(rng.choice(len(tr),size=nq,replace=False)); k=int(cfg['retrieval']['final_k']); temp=float(cfg['retrieval']['similarity_temperature'])
    ct=embedding_retrieval_correction(gt,gt[sub],tc['indices'][sub],res,k,temp); alpha=select_global_alpha(cfg,np.asarray(bt[sub],np.float32),np.asarray(tr.y[sub],np.float32),np.asarray(tr.valid[sub],bool),np.asarray(tr.direct[sub],bool),np.asarray(tr.support[sub],bool),ct); cv=embedding_retrieval_correction(gt,gv,vc['indices'],res,k,temp); pred=np.asarray(bv,np.float32)+alpha*cv; bm=metrics(cfg,np.asarray(bv,np.float32),va); vm=metrics(cfg,pred,va); np.save(rd/'val_predictions.npy',pred.astype(np.float32)); s={'variant':'M2','stage':stage,'seed':seed,'validation':vm,'base_validation':bm,'selected_alpha':alpha,'relative_improvement_over_M0':{'overall':rel(bm,vm,'overall_MAE'),'masked':rel(bm,vm,'masked_MAE'),'support':rel(bm,vm,'support_MAE')},'description':'Global similarity retrieval negative control using pooled node embeddings','test_evaluated':False}; write_json(rd/'summary.json',s); print(f'M2 global alpha={alpha:.2f} overall={vm["overall_MAE"]:.6f} masked={vm["masked_MAE"]:.6f} support={vm["support_MAE"]:.6f}',flush=True)


def train_m3(cfg,stage,seed,art,tr,va,rd,dev):
    m1,_,_=load_m1(cfg,stage,dev); zt,zv=ensure_embeddings(cfg,stage,art,m1,dev,force=False); nft,nfv=_feature_mmaps(art); res=np.load(art/'train_residual.npy',mmap_mode='r'); tc=np.load(art/'train_candidates.npz'); vc=np.load(art/'val_candidates.npz'); bt=np.load(art/'train_base_pred.npy',mmap_mode='r'); bv=np.load(art/'val_base_pred.npy',mmap_mode='r'); c=cfg['model']['m3']; args=dict(demb=zt.shape[-1],sense_dim=NODE_SENSE_DIM,hidden=int(c['hidden']),dropout=float(c['dropout']),max_scale=float(c['max_scale'])); net=NodeUsefulnessNet(**args).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=float(c['lr']),weight_decay=float(c['weight_decay'])); K=min(int(cfg['node_routing']['utility_candidate_k']),tc['indices'].shape[1]); best=1e99;wait=0;bestp=rd/'best.pt'; grid=[float(x) for x in cfg['oracle']['alpha_grid']]
    for ep in range(1,int(c['epochs'])+1):
        net.train(); ls=[]; clsstat=np.zeros(3,np.int64); scl=[]
        for ids in batches(len(tr),int(c['query_batch_size']),True,seed+ep):
            ci=np.asarray(tc['indices'][ids,:K],np.int64); sim=np.asarray(tc['scores'][ids,:K],np.float32); rr=np.asarray(res[ci],np.float32); _,urel,astar=node_utility_targets(np.asarray(bt[ids],np.float32),np.asarray(tr.y[ids],np.float32),np.asarray(tr.valid[ids],bool),rr,grid); labels=utility_labels(urel,float(c['positive_relative_threshold']),float(c['negative_relative_threshold'])); smask=balanced_hard_mask(urel,labels,int(c['sample_positive']),int(c['sample_harmful']),int(c['sample_neutral']))
            logits,score,scale=net(t(np.asarray(zt[ids],np.float32),dev),t(np.asarray(zt[ci],np.float32),dev),t(sim,dev),t(_sense(nft[ids]),dev),t(_sense(nft[ci]),dev)); lab=t(labels,dev,torch.long); mk=t(smask,dev,torch.bool); ce=torch.nn.functional.cross_entropy(logits[mk],lab[mk]) if torch.any(mk) else logits.sum()*0; rank=pairwise_rank_loss(score,t(urel,dev),float(c['positive_relative_threshold']),float(c['rank_margin'])); posmk=mk&(lab==2); sl=torch.abs(scale[posmk]-t(astar,dev)[posmk]).mean() if torch.any(posmk) else scale.sum()*0; loss=ce+float(c['rank_weight'])*rank+float(c['scale_weight'])*sl
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step(); ls.append(float(loss.detach())); scl.append(float(sl.detach())); clsstat+=np.bincount(labels[smask],minlength=3)
        net.eval(); outs=[]; acc=[]
        with torch.no_grad():
            for ids in batches(len(va),32): raw,pb,sb=m3_raw_correction(net,zt,zv,nft,nfv,vc,res,ids,cfg,dev,True); outs.append(raw.cpu().numpy()); acc.append((pb>=float(c['use_threshold'])).float().mean().item())
        raw=np.concatenate(outs); pred=np.asarray(bv,np.float32)+raw; vm=metrics(cfg,pred,va); sc=objective(cfg,vm); print(f'M3-NODE-v05 epoch={ep:03d} loss={np.mean(ls):.5f} scale={np.mean(scl):.4f} val_obj={sc:.5f} masked={vm["masked_MAE"]:.5f} support={vm["support_MAE"]:.5f} accept={np.mean(acc):.3f} sampled_cls={clsstat.tolist()}',flush=True)
        if sc<best: best=sc;wait=0;torch.save({'state_dict':net.state_dict(),'model_args':args,'model_type':'node_usefulness_v05','epoch':ep,'val_objective':sc},bestp)
        else: wait+=1
        if wait>=int(c['patience']): break
    ck=torch.load(bestp,map_location=dev,weights_only=False); net.load_state_dict(ck['state_dict']); net.eval(); diag_train=m3_support_diagnostics(net,zt,zt,nft,nft,tc,res,bt,tr,cfg,dev,seed); diag_val=m3_support_diagnostics(net,zt,zv,nft,nfv,vc,res,bv,va,cfg,dev,seed+1000); out=[]; acc=[]; scales=[]
    with torch.no_grad():
        for ids in batches(len(va),32): raw,pb,sb=m3_raw_correction(net,zt,zv,nft,nfv,vc,res,ids,cfg,dev,True); out.append(raw.cpu().numpy()); acc.append((pb>=float(c['use_threshold'])).float().mean().item()); scales.append(sb.cpu().numpy())
    raw=np.concatenate(out); ss=np.concatenate(scales); pred=np.asarray(bv,np.float32)+raw; bm=metrics(cfg,np.asarray(bv,np.float32),va); vm=metrics(cfg,pred,va); np.save(rd/'val_predictions.npy',pred.astype(np.float32)); summary={'variant':'M3','stage':stage,'seed':seed,'validation':vm,'base_validation':bm,'best_epoch':ck['epoch'],'node_acceptance_fraction':float(np.mean(acc)),'mean_selected_query_node_scale':float(ss.mean()),'nonzero_query_node_scale_fraction':float((ss>1e-6).mean()),'train_only_diagnostics':diag_train,'validation_diagnostics_future_labels_only':diag_val,'relative_improvement_over_M0':{'overall':rel(bm,vm,'overall_MAE'),'masked':rel(bm,vm,'masked_MAE'),'support':rel(bm,vm,'support_MAE')},'description':'Outcome-grounded node screening/ranking with learned query-node support scale alpha_qn; no global node calibration','test_evaluated':False}; write_json(rd/'summary.json',summary); save_latest(cfg,'M3',stage,rd,bestp); print(f'M3-NODE-v05 final overall={vm["overall_MAE"]:.6f} masked={vm["masked_MAE"]:.6f} support={vm["support_MAE"]:.6f} accept={summary["node_acceptance_fraction"]:.3f} top4_hit={diag_val["top4_beneficial_hit_rate"]:.3f}',flush=True)


def train_m4(cfg,stage,seed,art,tr,va,rd,dev):
    m1,_,_=load_m1(cfg,stage,dev); m3,_,_=load_m3(cfg,stage,dev)
    zt,zv=ensure_embeddings(cfg,stage,art,m1,dev,False); nft,nfv=_feature_mmaps(art)
    res=np.load(art/'train_residual.npy',mmap_mode='r'); tc=np.load(art/'train_candidates.npz'); vc=np.load(art/'val_candidates.npz')
    bt=np.load(art/'train_base_pred.npy',mmap_mode='r'); bv=np.load(art/'val_base_pred.npy',mmap_mode='r')
    man=json.loads((art/'node_feature_manifest.json').read_text()); mean,std=float(man['speed_mean']),float(man['speed_std']); c=cfg['model']['m4']
    args=dict(demb=zt.shape[-1],node_feature_dim=nft.shape[-1],P=bt.shape[1],sense_dim=NODE_SENSE_DIM,
              context_dim=int(c['context_dim']),residual_dim=int(c['residual_dim']),pair_dim=int(c['pair_dim']),
              hidden=int(c['hidden']),horizon_dim=int(c['horizon_embedding_dim']),attention_heads=int(c['attention_heads']),
              max_scale=float(c['max_scale']),dropout=float(c['dropout']))
    router=PairwiseSupportRouter(**args).to(dev); opt=torch.optim.AdamW(router.parameters(),lr=float(c['lr']),weight_decay=float(c['weight_decay']))
    best=1e99; wait=0; bestp=rd/'best.pt'; m3.eval(); alphas=[float(x) for x in c['teacher_alpha_grid']]

    def teacher_for(b,y,valid,rr,cmask):
        return m4_soft_teacher_targets(
            b,y,valid,rr,cmask,alphas,
            temperature=float(c['teacher_temperature']),
            mass_temperature=float(c['teacher_mass_temperature']),
            min_abs_gain=float(c['teacher_min_abs_gain']),
            gain_floor=float(c['teacher_gain_floor']),
            utility_clip=float(c['teacher_utility_clip']))

    # Phase T: pairwise compatibility learning against the already validated SoftMix teacher.
    for ep in range(1,int(c['teacher_pretrain_epochs'])+1):
        router.train(); losses=[]; ces=[]; dls=[]; cls=[]; mas=[]; accs=[]; ents=[]; rls=[]; racc=[]
        for ids in batches(len(tr),int(c['query_batch_size']),True,seed+100+ep):
            with torch.no_grad(): _,csel,cssel,conf,m3s,cmask,rr,qctx,cctx=m3_select_for_pairwise_router(m3,zt,zt,nft,nft,tc,res,ids,cfg,dev)
            q=t(np.asarray(zt[ids],np.float32),dev); qs=t(_sense(nft[ids]),dev); b=t(bt[ids],dev); y=t(tr.y[ids],dev); valid,direct,support=masks(tr,ids,dev)
            corr,mass,dis,slog,scale,noop=_m4_forward_selected(router,q,qs,qctx,csel,cssel,cctx,conf,cmask,rr,b,mean,std,cfg)
            with torch.no_grad(): probs,tdelta,tcorr,tmass,hard,urel,ent=teacher_for(b,y,valid,rr,cmask)
            ce,dl,cl,mm,acc,te=m4_soft_distill_loss(slog,scale,noop,rr,corr,mass,probs,tdelta,tcorr,tmass,valid)
            rl,ra=m4_pairwise_utility_rank_loss(slog,urel,cmask,valid,float(c['teacher_rank_margin']),float(c['teacher_rank_min_gap']))
            loss=(float(c['teacher_soft_ce_weight'])*ce+float(c['teacher_delta_weight'])*dl+float(c['teacher_corr_weight'])*cl+float(c['teacher_mass_weight'])*mm+float(c['teacher_rank_weight'])*rl)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(router.parameters(),5.0); opt.step()
            losses.append(float(loss.detach())); ces.append(float(ce.detach())); dls.append(float(dl.detach())); cls.append(float(cl.detach())); mas.append(float(mm.detach())); accs.append(float(acc.detach())); ents.append(float(te.detach())); rls.append(float(rl.detach())); racc.append(float(ra.detach()))
        print(f'M4-PAIR pretrain={ep:03d} loss={np.mean(losses):.5f} soft_ce={np.mean(ces):.4f} rank={np.mean(rls):.4f} rank_acc={np.mean(racc):.3f} delta={np.mean(dls):.4f} corr={np.mean(cls):.4f} mass_mae={np.mean(mas):.4f} top1={np.mean(accs):.3f} teacher_H={np.mean(ents):.3f}',flush=True)

    # Phase F: forecasting/no-harm fine-tuning while keeping compatibility distillation active.
    for ep in range(1,int(c['finetune_epochs'])+1):
        router.train(); losses=[]; masses=[]; accs=[]; corrteach=[]; racc=[]
        for ids in batches(len(tr),int(c['query_batch_size']),True,seed+300+ep):
            with torch.no_grad(): _,csel,cssel,conf,m3s,cmask,rr,qctx,cctx=m3_select_for_pairwise_router(m3,zt,zt,nft,nft,tc,res,ids,cfg,dev)
            q=t(np.asarray(zt[ids],np.float32),dev); qs=t(_sense(nft[ids]),dev); b=t(bt[ids],dev); y=t(tr.y[ids],dev); valid,direct,support=masks(tr,ids,dev)
            corr,mass,dis,slog,scale,noop=_m4_forward_selected(router,q,qs,qctx,csel,cssel,cctx,conf,cmask,rr,b,mean,std,cfg)
            pred=b+corr; forecast=weighted_mae_loss(pred,y,valid,direct,support,cfg['loss']); harm=local_harm_loss(pred,b,y,valid)
            with torch.no_grad(): probs,tdelta,tcorr,tmass,hard,urel,ent=teacher_for(b,y,valid,rr,cmask)
            ce,dl,cl,mm,acc,te=m4_soft_distill_loss(slog,scale,noop,rr,corr,mass,probs,tdelta,tcorr,tmass,valid)
            rl,ra=m4_pairwise_utility_rank_loss(slog,urel,cmask,valid,float(c['teacher_rank_margin']),float(c['teacher_rank_min_gap']))
            teach=(float(c['teacher_soft_ce_weight'])*ce+float(c['teacher_delta_weight'])*dl+float(c['teacher_corr_weight'])*cl+float(c['teacher_mass_weight'])*mm+float(c['teacher_rank_weight'])*rl)
            loss=forecast+float(c['local_harm_lambda'])*harm+float(c['support_mass_lambda'])*mass.mean()+float(c['teacher_keep_weight'])*teach
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(router.parameters(),5.0); opt.step()
            losses.append(float(loss.detach())); masses.append(float(mass.mean().detach())); accs.append(float(acc.detach())); corrteach.append(float(cl.detach())); racc.append(float(ra.detach()))
        router.eval(); pv=[]; ms=[]
        with torch.no_grad():
            for ids in batches(len(va),int(c.get('eval_batch_size',12))):
                corr,mass,dis=m4_local_correction(router,m3,zt,zv,nft,nfv,vc,res,ids,bv,mean,std,cfg,dev)
                pv.append((t(bv[ids],dev)+corr).cpu().numpy()); ms.append(float(mass.mean().cpu()))
        pv=np.concatenate(pv); vm=metrics(cfg,pv,va); sc=objective(cfg,vm)
        print(f'M4-PAIR-v07 finetune={ep:03d} loss={np.mean(losses):.5f} teacher_top1={np.mean(accs):.3f} rank_acc={np.mean(racc):.3f} corr_distill={np.mean(corrteach):.4f} val_obj={sc:.5f} overall={vm["overall_MAE"]:.5f} masked={vm["masked_MAE"]:.5f} support={vm["support_MAE"]:.5f} support_mass={np.mean(ms):.3f}',flush=True)
        if sc<best:
            best=sc; wait=0; torch.save({'state_dict':router.state_dict(),'model_args':args,'model_type':'pairwise_support_router_v07','epoch':ep,'val_objective':sc,'teacher_pretrain_epochs':int(c['teacher_pretrain_epochs']),'teacher_mode':'soft_local_utility_pairwise_compatibility'},bestp)
        else: wait+=1
        if wait>=int(c['patience']): break

    ck=torch.load(bestp,map_location=dev,weights_only=False); router.load_state_dict(ck['state_dict']); router.eval(); pv=[]; ms=[]
    # Future-labelled Validation diagnostics below are reporting only and are never model inputs.
    diag_n=min(len(va),int(c.get('diagnostic_queries',1000))); diag_ids=np.arange(diag_n); dtop=[]; drank=[]; dcorr=[]; dmass=[]
    with torch.no_grad():
        for ids in batches(len(va),int(c.get('eval_batch_size',12))):
            corr,mass,dis=m4_local_correction(router,m3,zt,zv,nft,nfv,vc,res,ids,bv,mean,std,cfg,dev)
            pv.append((t(bv[ids],dev)+corr).cpu().numpy()); ms.append(float(mass.mean().cpu()))
        for ids in batches(diag_n,int(c.get('eval_batch_size',12))):
            qids=diag_ids[ids]
            _,csel,cssel,conf,m3s,cmask,rr,qctx,cctx=m3_select_for_pairwise_router(m3,zt,zv,nft,nfv,vc,res,qids,cfg,dev)
            q=t(np.asarray(zv[qids],np.float32),dev); qs=t(_sense(nfv[qids]),dev); b=t(bv[qids],dev); y=t(va.y[qids],dev); valid,direct,support=masks(va,qids,dev)
            corr,mass,dis,slog,scale,noop=_m4_forward_selected(router,q,qs,qctx,csel,cssel,cctx,conf,cmask,rr,b,mean,std,cfg)
            probs,tdelta,tcorr,tmass,hard,urel,ent=teacher_for(b,y,valid,rr,cmask)
            ce,dl,cl,mm,acc,te=m4_soft_distill_loss(slog,scale,noop,rr,corr,mass,probs,tdelta,tcorr,tmass,valid); rl,ra=m4_pairwise_utility_rank_loss(slog,urel,cmask,valid,float(c['teacher_rank_margin']),float(c['teacher_rank_min_gap']))
            dtop.append(float(acc.cpu())); drank.append(float(ra.cpu())); dcorr.append(float(cl.cpu())); dmass.append(float(mm.cpu()))
    pv=np.concatenate(pv); bm=metrics(cfg,np.asarray(bv,np.float32),va); vm=metrics(cfg,pv,va); np.save(rd/'val_predictions.npy',pv.astype(np.float32))
    diag={'queries':int(diag_n),'teacher_top1_agreement':float(np.mean(dtop)),'pairwise_utility_rank_accuracy':float(np.mean(drank)),'mixture_correction_distill_loss':float(np.mean(dcorr)),'support_mass_mae':float(np.mean(dmass))}
    s={'variant':'M4','stage':stage,'seed':seed,'validation':vm,'base_validation':bm,'best_epoch':ck['epoch'],'mean_support_mass':float(np.mean(ms)),'teacher_pretrain_epochs':int(c['teacher_pretrain_epochs']),'teacher_mode':'soft_local_utility_pairwise_compatibility','validation_diagnostics_future_labels_only':diag,'relative_improvement_over_M0':{'overall':rel(bm,vm,'overall_MAE'),'masked':rel(bm,vm,'masked_MAE'),'support':rel(bm,vm,'support_MAE')},'description':'Pairwise query-support compatibility learning over the M3 Top-K shortlist using complete deployable query/support node context, residual-trajectory encoding, candidate self-attention and SoftMix outcome-grounded distillation','test_evaluated':False}
    write_json(rd/'summary.json',s); save_latest(cfg,'M4',stage,rd,bestp)
    print(f'M4-PAIR-v07 final overall={vm["overall_MAE"]:.6f} masked={vm["masked_MAE"]:.6f} support={vm["support_MAE"]:.6f} support_mass={s["mean_support_mass"]:.3f} val_rank_acc={diag["pairwise_utility_rank_accuracy"]:.3f}',flush=True)


def train_m5(cfg,stage,seed,art,tr,va,rd,dev):
    m1,_,_=load_m1(cfg,stage,dev); m3,_,_=load_m3(cfg,stage,dev); router,_,_=load_m4(cfg,stage,dev)
    zt,zv=ensure_embeddings(cfg,stage,art,m1,dev,False); nft,nfv=_feature_mmaps(art)
    res=np.load(art/'train_residual.npy',mmap_mode='r'); tc=np.load(art/'train_candidates.npz'); vc=np.load(art/'val_candidates.npz')
    bt=np.load(art/'train_base_pred.npy',mmap_mode='r'); bv=np.load(art/'val_base_pred.npy',mmap_mode='r')
    man=json.loads((art/'node_feature_manifest.json').read_text()); mean,std=float(man['speed_mean']),float(man['speed_std']); c=cfg['model']['m5']
    args=dict(demb=zt.shape[-1],P=bt.shape[1],N=bt.shape[2],sense_dim=NODE_SENSE_DIM,node_dim=int(c['node_embedding_dim']),horizon_dim=int(c['horizon_embedding_dim']),hidden=int(c['hidden']))
    gate=SensingAwareSafeGate(**args).to(dev); opt=torch.optim.AdamW(gate.parameters(),lr=float(c['lr']),weight_decay=float(c['weight_decay']))
    best=1e99; wait=0; bestp=rd/'best.pt'; router.eval(); m3.eval()

    # Phase T: continuous Train-only intervention-strength supervision.
    for ep in range(1,int(c['teacher_pretrain_epochs'])+1):
        gate.train(); losses=[]; gms=[]; tms=[]; maes=[]; pos=[]
        for ids in batches(len(tr),int(c['query_batch_size']),True,seed+700+ep):
            with torch.no_grad(): corr,mass,dis=m4_local_correction(router,m3,zt,zt,nft,nft,tc,res,ids,bt,mean,std,cfg,dev)
            b=t(bt[ids],dev); q=t(np.asarray(zt[ids],np.float32),dev); qs=t(_sense(nft[ids]),dev); y=t(tr.y[ids],dev); valid,direct,support=masks(tr,ids,dev)
            g=gate(q,qs,(b-mean)/max(std,1e-6),corr/max(std,1e-6),mass,dis/max(std,1e-6))
            with torch.no_grad(): tgt,gain,relg=continuous_gate_teacher(b,corr,y,valid,float(c['teacher_min_abs_gain']),float(c['teacher_gain_floor']))
            gl=continuous_gate_loss(g,tgt,relg,valid,float(c['teacher_gain_weight'])); loss=float(c['teacher_gate_weight'])*gl
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(gate.parameters(),5.0); opt.step()
            losses.append(float(loss.detach())); gms.append(float(g.mean().detach())); tms.append(float(tgt[valid].mean().detach()) if torch.any(valid) else 0.0); maes.append(float(torch.abs(g[valid]-tgt[valid]).mean().detach()) if torch.any(valid) else 0.0); pos.append(float((tgt[valid]>1e-4).float().mean().detach()) if torch.any(valid) else 0.0)
        print(f'M5-CONT pretrain={ep:03d} loss={np.mean(losses):.5f} gate={np.mean(gms):.3f} target={np.mean(tms):.3f} teacher_mae={np.mean(maes):.3f} positive={np.mean(pos):.3f}',flush=True)

    # Phase F: forecast/no-harm fine-tuning while retaining continuous intervention supervision.
    for ep in range(1,int(c['finetune_epochs'])+1):
        gate.train(); losses=[]; gms=[]; tmae=[]
        for ids in batches(len(tr),int(c['query_batch_size']),True,seed+900+ep):
            with torch.no_grad(): corr,mass,dis=m4_local_correction(router,m3,zt,zt,nft,nft,tc,res,ids,bt,mean,std,cfg,dev)
            b=t(bt[ids],dev); q=t(np.asarray(zt[ids],np.float32),dev); qs=t(_sense(nft[ids]),dev); y=t(tr.y[ids],dev); valid,direct,support=masks(tr,ids,dev)
            g=gate(q,qs,(b-mean)/max(std,1e-6),corr/max(std,1e-6),mass,dis/max(std,1e-6)); pred=b+g*corr
            forecast=weighted_mae_loss(pred,y,valid,direct,support,cfg['loss']); harm=local_harm_loss(pred,b,y,valid); mharm=masked_local_harm_loss(pred,b,y,valid,direct)
            with torch.no_grad(): tgt,gain,relg=continuous_gate_teacher(b,corr,y,valid,float(c['teacher_min_abs_gain']),float(c['teacher_gain_floor']))
            gl=continuous_gate_loss(g,tgt,relg,valid,float(c['teacher_gain_weight']))
            loss=forecast+float(c['local_harm_lambda'])*harm+float(c['masked_harm_lambda'])*mharm+float(c['teacher_keep_weight'])*float(c['teacher_gate_weight'])*gl+float(c['gate_l1_lambda'])*g.mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(gate.parameters(),5.0); opt.step()
            losses.append(float(loss.detach())); gms.append(float(g.mean().detach())); tmae.append(float(torch.abs(g[valid]-tgt[valid]).mean().detach()) if torch.any(valid) else 0.0)
        gate.eval(); pv=[]; gs=[]
        with torch.no_grad():
            for ids in batches(len(va),24):
                corr,mass,dis=m4_local_correction(router,m3,zt,zv,nft,nfv,vc,res,ids,bv,mean,std,cfg,dev); b=t(bv[ids],dev); q=t(np.asarray(zv[ids],np.float32),dev); qs=t(_sense(nfv[ids]),dev)
                g=gate(q,qs,(b-mean)/max(std,1e-6),corr/max(std,1e-6),mass,dis/max(std,1e-6)); pv.append((b+g*corr).cpu().numpy()); gs.append(g.cpu().numpy())
        pv=np.concatenate(pv); gg=np.concatenate(gs); vm=metrics(cfg,pv,va); sc=objective(cfg,vm)
        print(f'M5-SAFE-v07 finetune={ep:03d} loss={np.mean(losses):.5f} teacher_mae={np.mean(tmae):.3f} val_obj={sc:.5f} overall={vm["overall_MAE"]:.5f} masked={vm["masked_MAE"]:.5f} support={vm["support_MAE"]:.5f} gate={gg.mean():.3f}',flush=True)
        if sc<best:
            best=sc; wait=0; torch.save({'state_dict':gate.state_dict(),'model_args':args,'model_type':'sensing_safe_gate_v07','epoch':ep,'val_objective':sc,'teacher_pretrain_epochs':int(c['teacher_pretrain_epochs']),'teacher_mode':'continuous_local_intervention'},bestp)
        else: wait+=1
        if wait>=int(c['patience']): break

    ck=torch.load(bestp,map_location=dev,weights_only=False); gate.load_state_dict(ck['state_dict']); gate.eval(); pv=[]; gs=[]
    # Streaming validation diagnostics use future labels only for reporting, never as model inputs.
    bins=int(cfg.get('diagnostics',{}).get('auroc_bins',200)); ph=np.zeros(bins,np.int64); nh=np.zeros(bins,np.int64)
    pos_sum=neg_sum=0.0; pos_n=neg_n=0; teacher_abs=teacher_n=0.0; act_harm=act_n=0; mact_harm=mact_n=0
    with torch.no_grad():
        for ids in batches(len(va),24):
            corr,mass,dis=m4_local_correction(router,m3,zt,zv,nft,nfv,vc,res,ids,bv,mean,std,cfg,dev); b=t(bv[ids],dev); q=t(np.asarray(zv[ids],np.float32),dev); qs=t(_sense(nfv[ids]),dev); y=t(va.y[ids],dev); valid,direct,support=masks(va,ids,dev)
            g=gate(q,qs,(b-mean)/max(std,1e-6),corr/max(std,1e-6),mass,dis/max(std,1e-6)); pred=b+g*corr; pv.append(pred.cpu().numpy()); gs.append(g.cpu().numpy())
            tgt,gain,relg=continuous_gate_teacher(b,corr,y,valid,float(c['teacher_min_abs_gain']),float(c['teacher_gain_floor']))
            vm=valid.bool(); label=(tgt>1e-4)&vm; gn=g.detach().cpu().numpy(); ln=label.cpu().numpy(); vn=vm.cpu().numpy(); bi=np.clip((gn*(bins-1)).astype(np.int64),0,bins-1); ph+=np.bincount(bi[ln],minlength=bins); nh+=np.bincount(bi[vn&~ln],minlength=bins)
            if torch.any(label): pos_sum+=float(g[label].sum().cpu()); pos_n+=int(label.sum().cpu())
            neg=vm&~label
            if torch.any(neg): neg_sum+=float(g[neg].sum().cpu()); neg_n+=int(neg.sum().cpu())
            teacher_abs+=float(torch.abs(g[vm]-tgt[vm]).sum().cpu()); teacher_n+=int(vm.sum().cpu())
            harmful=(torch.abs(pred-y)>torch.abs(b-y)+1e-8)&vm; active=(g>0.5)&vm; act_harm+=int((harmful&active).sum().cpu()); act_n+=int(active.sum().cpu()); md=direct.bool()[:,None,:]; mactive=active&md; mact_harm+=int((harmful&mactive).sum().cpu()); mact_n+=int(mactive.sum().cpu())
    pv=np.concatenate(pv); gg=np.concatenate(gs); base=np.asarray(bv,np.float32); bm=metrics(cfg,base,va); vm=metrics(cfg,pv,va)
    vv=np.asarray(va.valid,bool); dm=np.asarray(va.direct,bool)[:,None,:]; sm=np.asarray(va.support,bool)[:,None,:]; err0=np.abs(base-np.asarray(va.y)); err1=np.abs(pv-np.asarray(va.y))
    harm=float(((err1>err0+1e-8)&vv).sum()/max(vv.sum(),1)); mharm=float(((err1>err0+1e-8)&vv&dm).sum()/max((vv&dm).sum(),1)); sharm=float(((err1>err0+1e-8)&vv&sm).sum()/max((vv&sm).sum(),1))
    np.save(rd/'val_predictions.npy',pv.astype(np.float32)); gate_auc=_binary_auroc_from_hist(ph,nh)
    s={'variant':'M5','stage':stage,'seed':seed,'validation':vm,'base_validation':bm,'best_epoch':ck['epoch'],'teacher_pretrain_epochs':int(c['teacher_pretrain_epochs']),'teacher_mode':'continuous_local_intervention','gate_mean':float(gg.mean()),'gate_active_fraction_gt_0_5':float((gg>0.5).mean()),'gate_teacher_mae_validation_future_labels_only':float(teacher_abs/max(teacher_n,1)),'gate_utility_auroc_validation_future_labels_only':gate_auc,'gate_positive_mean_validation_future_labels_only':float(pos_sum/max(pos_n,1)),'gate_nonpositive_mean_validation_future_labels_only':float(neg_sum/max(neg_n,1)),'activated_cell_harm_fraction':float(act_harm/max(act_n,1)),'masked_activated_cell_harm_fraction':float(mact_harm/max(mact_n,1)),'harmful_cell_fraction_vs_M0':harm,'masked_harmful_cell_fraction_vs_M0':mharm,'support_harmful_cell_fraction_vs_M0':sharm,'relative_improvement_over_M0':{'overall':rel(bm,vm,'overall_MAE'),'masked':rel(bm,vm,'masked_MAE'),'support':rel(bm,vm,'support_MAE')},'description':'Continuous local intervention-strength distillation followed by sensing-aware safe-gate forecast/no-harm fine-tuning','test_evaluated':False}
    write_json(rd/'summary.json',s); save_latest(cfg,'M5',stage,rd,bestp)
    print(f'M5-SAFE-v07 final overall={vm["overall_MAE"]:.6f} masked={vm["masked_MAE"]:.6f} support={vm["support_MAE"]:.6f} gate={s["gate_mean"]:.3f} gate_auc={gate_auc} masked_harm={mharm:.3f}',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',required=True,choices=['M0','M1','M2','M3','M4','M5']);ap.add_argument('--stage',default='development',choices=['development','formal']);ap.add_argument('--seed',type=int,default=10);a=ap.parse_args();cfg=load_yaml(ROOT/'configs/presure.yaml');seed_all(a.seed);art=load_artifacts(cfg,a.stage);cache=cache_dir(cfg,a.stage);tr=CacheSplit(cache,'train');va=CacheSplit(cache,'val');dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');rd=run_dir(cfg,a.variant,a.stage,stamp(),a.seed);rd.mkdir(parents=True,exist_ok=True);bm=json.loads((art/'backbone_manifest.json').read_text());rm=json.loads((art/'retrieval_manifest.json').read_text());lineage={'variant':a.variant,'stage':a.stage,'seed':a.seed,'implementation_revision':cfg.get('implementation_revision'),'cache_fingerprint':bm.get('cache_identity',{}).get('cache_fingerprint'),'backbone_checkpoint':bm.get('checkpoint'),'backbone_checkpoint_sha256':bm.get('checkpoint_sha256'),'retrieval_memory_split':rm.get('memory_split'),'deployable_inputs_exclude_direct_support_masks':True,'test_instantiated':False};write_json(rd/'RUN_MANIFEST.json',{**lineage,'status':'STARTED'})
    try: {'M0':run_m0,'M1':train_m1,'M2':run_m2,'M3':train_m3,'M4':train_m4,'M5':train_m5}[a.variant](cfg,a.stage,a.seed,art,tr,va,rd,dev)
    except Exception as e: write_json(rd/'RUN_MANIFEST.json',{**lineage,'status':'FAILED','error':repr(e)});raise
    write_json(rd/'RUN_MANIFEST.json',{**lineage,'status':'FINISHED'});print(f'{a.variant} {a.stage} PASS run_dir={rd}')
if __name__=='__main__': main()
