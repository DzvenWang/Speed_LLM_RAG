#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, json, numpy as np, torch
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe'); ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml, write_json, stamp, seed_all
from presure.contract import cache_dir
from presure.data import CacheSplit
from presure.metrics import metric_bundle, weighted_mae_loss, per_query_mae_torch
from presure.models import M1Model, HorizonUtilityNet, RegionHorizonUtilityNet, LocalSafeGate
from presure.pipeline import load_artifacts, run_dir, normalize_features, load_m1, load_m3, load_m4


def t(a, dev, dtype=torch.float32): return torch.tensor(np.asarray(a), device=dev, dtype=dtype)
def save_latest(cfg,var,stage,rd,ck):
    p=Path(cfg['project']['exp_root'])/var/cfg['experiment']['subgraph']/stage; p.mkdir(parents=True,exist_ok=True)
    write_json(p/'LATEST.json',{'run_dir':str(rd),'checkpoint':str(ck),'variant':var,'stage':stage,'implementation_revision':cfg.get('implementation_revision')})
def masks(ds,ids,dev): return t(ds.valid[ids],dev,torch.bool),t(ds.direct[ids],dev,torch.bool),t(ds.support[ids],dev,torch.bool)
def metrics(cfg,pred,ds): return metric_bundle(cfg,pred,np.asarray(ds.y),np.asarray(ds.valid,bool),np.asarray(ds.direct,bool),np.asarray(ds.support,bool))
def objective_from_metrics(cfg,m):
    w=cfg['oracle']['objective_weights']; return float(w['overall'])*float(m['overall_MAE'])+float(w['masked'])*float(m['masked_MAE'])+float(w['support'])*float(m['support_MAE'])
def rel_improvement(base,new,key):
    b=float(base[key]); return (b-float(new[key]))/b if b else 0.0
def batches(n,bs,shuffle,seed):
    idx=np.arange(n)
    if shuffle: np.random.default_rng(seed).shuffle(idx)
    for s in range(0,n,bs): yield idx[s:s+bs]
def embeddings(enc,feat,dev,bs=512):
    out=[]; enc.eval()
    with torch.no_grad():
        for ids in batches(len(feat),bs,False,0): out.append(enc(t(feat[ids],dev)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

def embedding_retrieval_correction(zmem,zq,cids,res,k,temp=0.12,batch=128):
    out=[]; zm=zmem/(np.linalg.norm(zmem,axis=1,keepdims=True)+1e-8); zqq=zq/(np.linalg.norm(zq,axis=1,keepdims=True)+1e-8)
    for a in range(0,len(zq),batch):
        ids=cids[a:a+batch]; sim=np.einsum('bd,bkd->bk',zqq[a:a+batch],zm[ids])
        top=np.argpartition(-sim,kth=min(k-1,sim.shape[1]-1),axis=1)[:,:k]
        ss=np.take_along_axis(sim,top,1); ii=np.take_along_axis(ids,top,1); order=np.argsort(-ss,axis=1)
        ss=np.take_along_axis(ss,order,1); ii=np.take_along_axis(ii,order,1)
        zz=ss/max(float(temp),1e-4); zz-=zz.max(1,keepdims=True); w=np.exp(zz); w/=w.sum(1,keepdims=True)
        rr=np.asarray(res[ii],np.float32); out.append((rr*w[:,:,None,None]).sum(1))
    return np.concatenate(out).astype(np.float32)


def node_weights_np(valid,direct,support,cfg):
    w=cfg['oracle']['objective_weights']
    return np.asarray(valid,np.float32)*(float(w['overall'])+float(w['masked'])*np.asarray(direct,np.float32)[:,None,:]+float(w['support'])*np.asarray(support,np.float32)[:,None,:])


def horizon_utility_targets(base,y,valid,direct,support,rr,cfg):
    """Utility [B,K,P] for applying one candidate residual to an entire horizon."""
    ww=node_weights_np(valid,direct,support,cfg); den=ww.sum(2).clip(min=1e-6)
    eb=np.abs(base-y); base_obj=(eb*ww).sum(2)/den
    ec=np.abs(base[:,None]+rr-y[:,None]); cand_obj=(ec*ww[:,None]).sum(3)/den[:,None]
    return (base_obj[:,None]-cand_obj).astype(np.float32)


def region_horizon_utility_targets(base,y,valid,direct,support,rr,regions,cfg):
    """Utility [B,K,P,R] for graph-region × horizon support roles."""
    ww=node_weights_np(valid,direct,support,cfg); B,K,P,N=rr.shape; R=int(regions.max())+1
    out=np.zeros((B,K,P,R),np.float32); eb=np.abs(base-y); ec=np.abs(base[:,None]+rr-y[:,None])
    for r in range(R):
        nodes=np.flatnonzero(regions==r); wr=ww[:,:,nodes]; den=wr.sum(2).clip(min=1e-6)
        bo=(eb[:,:,nodes]*wr).sum(2)/den
        co=(ec[:,:,:,nodes]*wr[:,None]).sum(3)/den[:,None]
        out[:,:,:,r]=bo[:,None]-co
    return out


def horizon_local_features(base,obs,direct,support,rr):
    """Deployable local descriptors [B,K,P,8]; candidate residual is Train-memory evidence."""
    rr=np.asarray(rr,np.float32); base=np.asarray(base,np.float32); B,K,P,N=rr.shape
    rmean=rr.mean(3); rstd=rr.std(3); rabs=np.abs(rr).mean(3)
    bm=base.mean(2)[:,None,:].repeat(K,1); bs=base.std(2)[:,None,:].repeat(K,1)
    of=np.asarray(obs,np.float32).mean((1,2))[:,None,None]*np.ones((B,K,P),np.float32)
    df=np.asarray(direct,np.float32).mean(1)[:,None,None]*np.ones((B,K,P),np.float32)
    sf=np.asarray(support,np.float32).mean(1)[:,None,None]*np.ones((B,K,P),np.float32)
    return np.stack([rmean,rstd,rabs,bm,bs,of,df,sf],-1).astype(np.float32)


def region_horizon_local_features(base,obs,direct,support,rr,regions):
    """Deployable graph-region × horizon descriptors [B,K,P,R,8]."""
    rr=np.asarray(rr,np.float32); base=np.asarray(base,np.float32); B,K,P,N=rr.shape; R=int(regions.max())+1
    out=np.empty((B,K,P,R,8),np.float32); obs=np.asarray(obs,np.float32); direct=np.asarray(direct,np.float32); support=np.asarray(support,np.float32)
    for r in range(R):
        nodes=np.flatnonzero(regions==r); x=rr[:,:,:,nodes]
        out[:,:,:,r,0]=x.mean(3); out[:,:,:,r,1]=x.std(3); out[:,:,:,r,2]=np.abs(x).mean(3)
        bm=base[:,:,nodes].mean(2); bs=base[:,:,nodes].std(2)
        out[:,:,:,r,3]=bm[:,None,:]; out[:,:,:,r,4]=bs[:,None,:]
        of=obs[:,:,nodes].mean((1,2)); df=direct[:,nodes].mean(1); sf=support[:,nodes].mean(1)
        out[:,:,:,r,5]=of[:,None,None]; out[:,:,:,r,6]=df[:,None,None]; out[:,:,:,r,7]=sf[:,None,None]
    return out


def ranking_loss(pred,target,margin):
    """Candidate axis is dim=1; remaining axes are independent local roles."""
    top=target.argmax(1,keepdim=True); pos=torch.gather(pred,1,top)
    raw=torch.relu(float(margin)-pos+pred)
    mask=torch.ones_like(raw,dtype=torch.bool); mask.scatter_(1,top,False)
    return raw[mask].mean() if torch.any(mask) else pred.sum()*0


def _positive_top_weights(u,k,eps=0.0):
    k=min(int(k),u.shape[1]); vals,idx=torch.topk(u,k,dim=1)
    pos=torch.relu(vals-float(eps)); den=pos.sum(1,keepdim=True)
    w=torch.where(den>1e-12,pos/(den+1e-12),torch.zeros_like(pos))
    return idx,w


def horizon_corr_from_utility(u,rr,k,eps=0.0):
    """u [B,K,P], rr [B,K,P,N] -> [B,P,N]."""
    top,w=_positive_top_weights(u,k,eps); B,kk,P=top.shape; N=rr.shape[-1]
    corr=torch.zeros((B,P,N),device=rr.device,dtype=rr.dtype)
    for p in range(P):
        ix=top[:,:,p,None].expand(-1,-1,N)
        rp=torch.gather(rr[:,:,p,:],1,ix)
        corr[:,p,:]=(rp*w[:,:,p,None]).sum(1)
    return corr, (w.sum(1)>0)


def region_corr_from_utility(u,rr,regions,k,eps=0.0):
    """u [B,K,P,R], rr [B,K,P,N] -> local [B,P,N]."""
    top,w=_positive_top_weights(u,k,eps); B,kk,P,R=top.shape; N=rr.shape[-1]
    corr=torch.zeros((B,P,N),device=rr.device,dtype=rr.dtype)
    reg=torch.as_tensor(regions,device=rr.device,dtype=torch.long)
    for r in range(R):
        nodes=torch.nonzero(reg==r,as_tuple=False).squeeze(1)
        if nodes.numel()==0: continue
        rr_reg=rr[:,:,:,nodes]  # B,K,P,Nr
        ix=top[:,:,:,r,None].expand(-1,-1,-1,nodes.numel())
        chosen=torch.gather(rr_reg,1,ix)
        c=(chosen*w[:,:,:,r,None]).sum(1)
        corr[:,:,nodes]=c
    return corr,(w.sum(1)>0)


def calibrate_horizon_alpha(base,y,valid,direct,support,corr,cfg):
    grid=[0.0]+[float(x) for x in cfg['oracle']['alpha_grid']]; ww=node_weights_np(valid,direct,support,cfg); P=base.shape[1]; out=np.zeros(P,np.float32)
    for p in range(P):
        den=ww[:,p].sum().clip(min=1e-6); scores=[]
        for a in grid: scores.append((float((np.abs(base[:,p]+a*corr[:,p]-y[:,p])*ww[:,p]).sum()/den),a))
        out[p]=min(scores)[1]
    return out


def calibrate_region_horizon_alpha(base,y,valid,direct,support,corr,regions,cfg):
    grid=[0.0]+[float(x) for x in cfg['oracle']['alpha_grid']]; ww=node_weights_np(valid,direct,support,cfg); P=base.shape[1]; R=int(regions.max())+1; out=np.zeros((P,R),np.float32)
    for r in range(R):
        nodes=np.flatnonzero(regions==r)
        for p in range(P):
            wr=ww[:,p,nodes]; den=wr.sum().clip(min=1e-6); scores=[]
            for a in grid: scores.append((float((np.abs(base[:,p,nodes]+a*corr[:,p,nodes]-y[:,p,nodes])*wr).sum()/den),a))
            out[p,r]=min(scores)[1]
    return out


def apply_horizon_alpha(corr,alpha_h): return corr*np.asarray(alpha_h,np.float32)[None,:,None]
def apply_region_alpha(corr,alpha_hr,regions):
    out=np.asarray(corr,np.float32).copy()
    for r in range(int(regions.max())+1): out[:,:,regions==r]*=np.asarray(alpha_hr,np.float32)[:,r][None,:,None]
    return out


def run_m0(cfg,stage,seed,art,tr,va,rd,dev):
    p=np.asarray(np.load(art/'val_base_pred.npy',mmap_mode='r'),np.float32); m=metrics(cfg,p,va); np.save(rd/'val_predictions.npy',p)
    write_json(rd/'summary.json',{'variant':'M0','stage':stage,'seed':seed,'validation':m,'description':'Frozen PatchSTG','test_evaluated':False})
    print(f'M0 overall={m["overall_MAE"]:.6f} masked={m["masked_MAE"]:.6f} support={m["support_MAE"]:.6f}',flush=True)


def train_m1(cfg,stage,seed,art,tr,va,rd,dev):
    feat_tr=normalize_features(art,'train'); feat_va=normalize_features(art,'val'); base_tr=np.load(art/'train_base_pred.npy',mmap_mode='r'); base_va=np.load(art/'val_base_pred.npy',mmap_mode='r'); mc=cfg['model']; c=mc['m1']
    args=dict(din=feat_tr.shape[1],hidden=int(mc['feature_hidden']),demb=int(mc['embedding_dim']),dropout=float(mc['dropout']),P=int(cfg['experiment']['horizon']),N=base_tr.shape[2],rank=int(mc['residual_rank']))
    model=M1Model(**args).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=float(c['lr']),weight_decay=float(c['weight_decay'])); best=1e9; wait=0; bestp=rd/'best.pt'
    for ep in range(1,int(c['epochs'])+1):
        model.train(); ls=[]
        for ids in batches(len(tr),int(c['batch_size']),True,seed+ep):
            x=t(feat_tr[ids],dev); b=t(base_tr[ids],dev); y=t(tr.y[ids],dev); valid,direct,support=masks(tr,ids,dev); corr,_=model(x); corr=torch.clamp(corr,-float(mc['correction_clip']),float(mc['correction_clip'])); pred=b+corr
            loss=weighted_mae_loss(pred,y,valid,direct,support,cfg['loss']); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); ls.append(float(loss.detach()))
        model.eval(); preds=[]
        with torch.no_grad():
            for ids in batches(len(va),256,False,0):
                corr,_=model(t(feat_va[ids],dev)); preds.append((t(base_va[ids],dev)+torch.clamp(corr,-float(mc['correction_clip']),float(mc['correction_clip']))).cpu().numpy())
        pred=np.concatenate(preds); vm=metrics(cfg,pred,va); score=objective_from_metrics(cfg,vm); print(f'M1 epoch={ep:03d} train={np.mean(ls):.5f} val_obj={score:.5f} overall={vm["overall_MAE"]:.5f} masked={vm["masked_MAE"]:.5f} support={vm["support_MAE"]:.5f}',flush=True)
        if score<best: best=score;wait=0;torch.save({'state_dict':model.state_dict(),'model_args':args,'epoch':ep,'val_objective':best},bestp)
        else: wait+=1
        if wait>=int(c['patience']): break
    ck=torch.load(bestp,map_location=dev,weights_only=False); model.load_state_dict(ck['state_dict']); model.eval(); preds=[]
    with torch.no_grad():
        for ids in batches(len(va),256,False,0):
            corr,_=model(t(feat_va[ids],dev)); preds.append((t(base_va[ids],dev)+torch.clamp(corr,-float(mc['correction_clip']),float(mc['correction_clip']))).cpu().numpy())
    pred=np.concatenate(preds); vm=metrics(cfg,pred,va); np.save(rd/'val_predictions.npy',pred.astype(np.float32))
    write_json(rd/'summary.json',{'variant':'M1','stage':stage,'seed':seed,'validation':vm,'best_epoch':ck['epoch'],'description':'Sensing-aware query representation + query-only diagnostic head','test_evaluated':False}); save_latest(cfg,'M1',stage,rd,bestp)


def run_m2(cfg,stage,seed,art,tr,va,rd,dev):
    m1,_=load_m1(cfg,stage,dev); res=np.load(art/'train_residual.npy',mmap_mode='r'); tc=np.load(art/'train_candidates.npz'); vc=np.load(art/'val_candidates.npz'); k=int(cfg['retrieval']['final_k']); temp=float(cfg['retrieval']['similarity_temperature']); ftr=normalize_features(art,'train'); fva=normalize_features(art,'val'); ztr=embeddings(m1.encoder,ftr,dev); zva=embeddings(m1.encoder,fva,dev)
    ctr=embedding_retrieval_correction(ztr,ztr,tc['indices'],res,k,temp); cva=embedding_retrieval_correction(ztr,zva,vc['indices'],res,k,temp); base_tr=np.asarray(np.load(art/'train_base_pred.npy',mmap_mode='r'),np.float32); base_va=np.asarray(np.load(art/'val_base_pred.npy',mmap_mode='r'),np.float32)
    grid=[0.0]+[float(x) for x in cfg['oracle']['alpha_grid']]; scored=[]; rng=np.random.default_rng(seed); sub=np.sort(rng.choice(len(tr),size=min(len(tr),4000),replace=False)); train_view=type('D',(),{'y':tr.y[sub],'valid':tr.valid[sub],'direct':tr.direct[sub],'support':tr.support[sub]})()
    for a in grid:
        mm=metrics(cfg,base_tr[sub]+a*ctr[sub],train_view); scored.append((objective_from_metrics(cfg,mm),a,mm))
    _,alpha,_=min(scored,key=lambda x:x[0]); pred=base_va+alpha*cva; vm=metrics(cfg,pred,va); bm=metrics(cfg,base_va,va); np.save(rd/'val_predictions.npy',pred.astype(np.float32))
    summary={'variant':'M2','stage':stage,'seed':seed,'validation':vm,'base_validation':bm,'alpha_train_selected':alpha,'train_alpha_grid_scores':[{'alpha':a,'objective':o} for o,a,_ in scored],'relative_improvement_over_M0':{'overall':rel_improvement(bm,vm,'overall_MAE'),'masked':rel_improvement(bm,vm,'masked_MAE'),'support':rel_improvement(bm,vm,'support_MAE')},'description':'Global sensing-aware similarity retrieval; negative-control retrieval stage','test_evaluated':False}
    write_json(rd/'summary.json',summary); p=Path(cfg['project']['exp_root'])/'M2'/cfg['experiment']['subgraph']/stage;p.mkdir(parents=True,exist_ok=True);write_json(p/'LATEST.json',{'run_dir':str(rd),'alpha':alpha,'variant':'M2','stage':stage})
    print(f'M2 alpha={alpha:.2f} overall={vm["overall_MAE"]:.6f} masked={vm["masked_MAE"]:.6f} support={vm["support_MAE"]:.6f} rel_masked={summary["relative_improvement_over_M0"]["masked"]:.4%}',flush=True)


def _m3_corr(net,zmem,zq,res,cand,ids,ds,base,cfg,dev):
    cids=cand['indices'][ids,:min(int(cfg['local_routing']['utility_candidate_k']),cand['indices'].shape[1])]; sim=cand['scores'][ids,:cids.shape[1]]
    rr_np=np.asarray(res[cids],np.float32); local=horizon_local_features(np.asarray(base[ids],np.float32),np.asarray(ds.obs[ids]),np.asarray(ds.direct[ids]),np.asarray(ds.support[ids]),rr_np)
    q=t(zq[ids],dev)[:,None,:].expand(-1,cids.shape[1],-1); cz=t(zmem[cids],dev); u=net(q,cz,t(sim,dev),t(local,dev)); rr=t(rr_np,dev)
    corr,accept=horizon_corr_from_utility(u,rr,int(cfg['local_routing']['m3_top_k']),float(cfg['model']['m3'].get('positive_utility_epsilon',0.0)))
    return corr,accept,u


def _m3_predict(net,zmem,zq,res,cand,base,ds,alpha_h,cfg,dev,return_stats=False):
    out=[]; accepts=[]; net.eval(); ah=t(np.asarray(alpha_h,np.float32),dev)
    with torch.no_grad():
        for ids in batches(len(zq),96,False,0):
            corr,acc,_=_m3_corr(net,zmem,zq,res,cand,ids,ds,base,cfg,dev); out.append((t(base[ids],dev)+corr*ah[None,:,None]).cpu().numpy()); accepts.append(acc.cpu().numpy())
    pred=np.concatenate(out); ac=np.concatenate(accepts)
    stats={'query_horizon_acceptance_fraction':float(ac.mean()),'queries_accepting_any_support_fraction':float(ac.any(1).mean())}
    return (pred,stats) if return_stats else pred


def train_m3(cfg,stage,seed,art,tr,va,rd,dev):
    m1,_=load_m1(cfg,stage,dev); enc=m1.encoder; enc.eval(); ftr=normalize_features(art,'train'); fva=normalize_features(art,'val'); ztr=embeddings(enc,ftr,dev); zva=embeddings(enc,fva,dev); res=np.load(art/'train_residual.npy',mmap_mode='r'); tc=np.load(art/'train_candidates.npz'); vc=np.load(art/'val_candidates.npz'); base_tr=np.load(art/'train_base_pred.npy',mmap_mode='r'); base_va=np.load(art/'val_base_pred.npy',mmap_mode='r'); c=cfg['model']['m3']; uk=min(int(cfg['local_routing']['utility_candidate_k']),tc['indices'].shape[1]); P=base_tr.shape[1]
    net=HorizonUtilityNet(ztr.shape[1],P,local_dim=8,horizon_dim=12).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=float(c['lr']),weight_decay=float(c['weight_decay'])); best=1e9; wait=0; bestp=rd/'best.pt'; eval_alpha=np.full(P,float(c.get('eval_scale',0.5)),np.float32)
    for ep in range(1,int(c['epochs'])+1):
        net.train(); ls=[]
        for ids in batches(len(tr),int(c['query_batch_size']),True,seed+ep):
            ci=tc['indices'][ids,:uk]; sim=tc['scores'][ids,:uk]; rr=np.asarray(res[ci],np.float32)*float(c.get('candidate_scale',1.0)); target_np=horizon_utility_targets(np.asarray(base_tr[ids],np.float32),np.asarray(tr.y[ids],np.float32),np.asarray(tr.valid[ids],bool),np.asarray(tr.direct[ids],bool),np.asarray(tr.support[ids],bool),rr,cfg)
            q=t(ztr[ids],dev)[:,None,:].expand(-1,uk,-1); local=horizon_local_features(np.asarray(base_tr[ids],np.float32),np.asarray(tr.obs[ids]),np.asarray(tr.direct[ids]),np.asarray(tr.support[ids]),rr); pred=net(q,t(ztr[ci],dev),t(sim,dev),t(local,dev)); target=t(target_np,dev); reg=torch.nn.functional.smooth_l1_loss(pred,target); rank=ranking_loss(pred,target,float(c['rank_margin'])); loss=reg+0.15*rank
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step(); ls.append(float(loss.detach()))
        predv,st=_m3_predict(net,ztr,zva,res,vc,base_va,va,eval_alpha,cfg,dev,True); vm=metrics(cfg,predv,va); score=objective_from_metrics(cfg,vm)
        print(f'M3-H epoch={ep:03d} loss={np.mean(ls):.5f} val_obj={score:.5f} overall={vm["overall_MAE"]:.5f} masked={vm["masked_MAE"]:.5f} support={vm["support_MAE"]:.5f} role_accept={st["query_horizon_acceptance_fraction"]:.3f}',flush=True)
        if score<best: best=score;wait=0;torch.save({'state_dict':net.state_dict(),'model_type':'horizon_utility_v03','demb':ztr.shape[1],'P':P,'local_dim':8,'horizon_dim':12,'epoch':ep,'val_objective':best},bestp)
        else: wait+=1
        if wait>=int(c['patience']): break
    ck=torch.load(bestp,map_location=dev,weights_only=False); net.load_state_dict(ck['state_dict']); rng=np.random.default_rng(seed+31); nq=min(len(tr),int(cfg['local_routing']['train_calibration_queries'])); sub=np.sort(rng.choice(len(tr),size=nq,replace=False)); raw=[]
    with torch.no_grad():
        for ids0 in batches(len(sub),96,False,0):
            qids=sub[ids0]; corr,_,_=_m3_corr(net,ztr,ztr,res,tc,qids,tr,base_tr,cfg,dev); raw.append(corr.cpu().numpy())
    raw=np.concatenate(raw); alpha_h=calibrate_horizon_alpha(np.asarray(base_tr[sub],np.float32),np.asarray(tr.y[sub],np.float32),np.asarray(tr.valid[sub],bool),np.asarray(tr.direct[sub],bool),np.asarray(tr.support[sub],bool),raw,cfg); ck['alpha_h']=alpha_h; torch.save(ck,bestp)
    predv,st=_m3_predict(net,ztr,zva,res,vc,base_va,va,alpha_h,cfg,dev,True); vm=metrics(cfg,predv,va); bm=metrics(cfg,np.asarray(base_va,np.float32),va); np.save(rd/'val_predictions.npy',predv.astype(np.float32)); summary={'variant':'M3','stage':stage,'seed':seed,'validation':vm,'base_validation':bm,'best_epoch':ck['epoch'],'alpha_h_train_selected':alpha_h.tolist(),**st,'relative_improvement_over_M0':{'overall':rel_improvement(bm,vm,'overall_MAE'),'masked':rel_improvement(bm,vm,'masked_MAE'),'support':rel_improvement(bm,vm,'support_MAE')},'description':'Horizon-aware predictive utility routing with explicit no-support path','test_evaluated':False}; write_json(rd/'summary.json',summary); save_latest(cfg,'M3',stage,rd,bestp)
    print(f'M3-H final overall={vm["overall_MAE"]:.6f} masked={vm["masked_MAE"]:.6f} support={vm["support_MAE"]:.6f} rel_masked={summary["relative_improvement_over_M0"]["masked"]:.4%} alpha_h={alpha_h.tolist()}',flush=True)


def _m4_corr(net,zmem,zq,res,cand,ids,ds,base,regions,alpha_hr,cfg,dev):
    k0=min(int(cfg['local_routing']['utility_candidate_k']),cand['indices'].shape[1]); cids=cand['indices'][ids,:k0]; sim=cand['scores'][ids,:k0]
    rr_np=np.asarray(res[cids],np.float32); local=region_horizon_local_features(np.asarray(base[ids],np.float32),np.asarray(ds.obs[ids]),np.asarray(ds.direct[ids]),np.asarray(ds.support[ids]),rr_np,regions)
    q=t(zq[ids],dev)[:,None,:].expand(-1,k0,-1); u=net(q,t(zmem[cids],dev),t(sim,dev),t(local,dev)); rr=t(rr_np,dev); corr,accept=region_corr_from_utility(u,rr,regions,int(cfg['local_routing']['m4_top_k']),float(cfg['model']['m4'].get('positive_utility_epsilon',0.0)))
    # Calibrated Train-only scale by graph-region × horizon.
    ah=torch.as_tensor(alpha_hr,device=dev,dtype=corr.dtype); reg=torch.as_tensor(regions,device=dev,dtype=torch.long)
    for r in range(int(reg.max().item())+1): corr[:,:,reg==r] *= ah[:,r][None,:,None]
    return corr,accept,u


def _m4_predict(net,zmem,zq,res,cand,base,ds,regions,alpha_hr,cfg,dev,return_stats=False):
    out=[]; accepts=[]; net.eval()
    with torch.no_grad():
        for ids in batches(len(zq),64,False,0):
            corr,acc,_=_m4_corr(net,zmem,zq,res,cand,ids,ds,base,regions,alpha_hr,cfg,dev); out.append((t(base[ids],dev)+corr).cpu().numpy()); accepts.append(acc.cpu().numpy())
    pred=np.concatenate(out); ac=np.concatenate(accepts); stats={'region_horizon_acceptance_fraction':float(ac.mean()),'queries_accepting_any_region_support_fraction':float(ac.reshape(len(ac),-1).any(1).mean())}
    return (pred,stats) if return_stats else pred


def train_m4(cfg,stage,seed,art,tr,va,rd,dev):
    m1,_=load_m1(cfg,stage,dev); enc=m1.encoder; enc.eval(); ftr=normalize_features(art,'train'); fva=normalize_features(art,'val'); ztr=embeddings(enc,ftr,dev); zva=embeddings(enc,fva,dev); res=np.load(art/'train_residual.npy',mmap_mode='r'); tc=np.load(art/'train_candidates.npz'); vc=np.load(art/'val_candidates.npz'); base_tr=np.load(art/'train_base_pred.npy',mmap_mode='r'); base_va=np.load(art/'val_base_pred.npy',mmap_mode='r'); regions=np.load(art/'node_regions.npy').astype(np.int32); P=base_tr.shape[1]; R=int(regions.max())+1; c=cfg['model']['m4']; uk=min(int(cfg['local_routing']['utility_candidate_k']),tc['indices'].shape[1])
    net=RegionHorizonUtilityNet(ztr.shape[1],P,R,local_dim=8,horizon_dim=8,region_dim=16).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=float(c['lr']),weight_decay=float(c['weight_decay'])); best=1e9; wait=0; bestp=rd/'best.pt'; eval_alpha=np.full((P,R),0.5,np.float32)
    for ep in range(1,int(c['epochs'])+1):
        net.train(); ls=[]
        for ids in batches(len(tr),int(c['query_batch_size']),True,seed+ep):
            ci=tc['indices'][ids,:uk]; sim=tc['scores'][ids,:uk]; rr=np.asarray(res[ci],np.float32); target_np=region_horizon_utility_targets(np.asarray(base_tr[ids],np.float32),np.asarray(tr.y[ids],np.float32),np.asarray(tr.valid[ids],bool),np.asarray(tr.direct[ids],bool),np.asarray(tr.support[ids],bool),rr,regions,cfg)
            q=t(ztr[ids],dev)[:,None,:].expand(-1,uk,-1); local=region_horizon_local_features(np.asarray(base_tr[ids],np.float32),np.asarray(tr.obs[ids]),np.asarray(tr.direct[ids]),np.asarray(tr.support[ids]),rr,regions); pred=net(q,t(ztr[ci],dev),t(sim,dev),t(local,dev)); target=t(target_np,dev); regloss=torch.nn.functional.smooth_l1_loss(pred,target); rank=ranking_loss(pred,target,float(c['rank_margin'])); loss=regloss+0.10*rank
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.0); opt.step(); ls.append(float(loss.detach()))
        predv,st=_m4_predict(net,ztr,zva,res,vc,base_va,va,regions,eval_alpha,cfg,dev,True); vm=metrics(cfg,predv,va); score=objective_from_metrics(cfg,vm)
        print(f'M4-RH epoch={ep:03d} loss={np.mean(ls):.5f} val_obj={score:.5f} overall={vm["overall_MAE"]:.5f} masked={vm["masked_MAE"]:.5f} support={vm["support_MAE"]:.5f} role_accept={st["region_horizon_acceptance_fraction"]:.3f}',flush=True)
        if score<best: best=score;wait=0;torch.save({'state_dict':net.state_dict(),'model_type':'region_horizon_utility_v03','demb':ztr.shape[1],'P':P,'R':R,'local_dim':8,'horizon_dim':8,'region_dim':16,'epoch':ep,'val_objective':best,'region_manifest':json.loads((art/'region_manifest.json').read_text())},bestp)
        else: wait+=1
        if wait>=int(c['patience']): break
    ck=torch.load(bestp,map_location=dev,weights_only=False); net.load_state_dict(ck['state_dict']); rng=np.random.default_rng(seed+43); nq=min(len(tr),int(cfg['local_routing']['train_calibration_queries'])); sub=np.sort(rng.choice(len(tr),size=nq,replace=False)); raw=[]
    # Get unscaled local correction by passing alpha=1 for all local roles.
    ones=np.ones((P,R),np.float32)
    with torch.no_grad():
        for ids0 in batches(len(sub),64,False,0):
            qids=sub[ids0]; corr,_,_=_m4_corr(net,ztr,ztr,res,tc,qids,tr,base_tr,regions,ones,cfg,dev); raw.append(corr.cpu().numpy())
    raw=np.concatenate(raw); alpha_hr=calibrate_region_horizon_alpha(np.asarray(base_tr[sub],np.float32),np.asarray(tr.y[sub],np.float32),np.asarray(tr.valid[sub],bool),np.asarray(tr.direct[sub],bool),np.asarray(tr.support[sub],bool),raw,regions,cfg); ck['alpha_hr']=alpha_hr; torch.save(ck,bestp)
    predv,st=_m4_predict(net,ztr,zva,res,vc,base_va,va,regions,alpha_hr,cfg,dev,True); vm=metrics(cfg,predv,va); bm=metrics(cfg,np.asarray(base_va,np.float32),va); np.save(rd/'val_predictions.npy',predv.astype(np.float32)); summary={'variant':'M4','stage':stage,'seed':seed,'validation':vm,'base_validation':bm,'best_epoch':ck['epoch'],'n_graph_regions':R,'alpha_hr_nonzero_fraction':float((alpha_hr>0).mean()),**st,'relative_improvement_over_M0':{'overall':rel_improvement(bm,vm,'overall_MAE'),'masked':rel_improvement(bm,vm,'masked_MAE'),'support':rel_improvement(bm,vm,'support_MAE')},'description':'Graph-region × horizon predictive utility routing and local residual transport','test_evaluated':False}; write_json(rd/'summary.json',summary); save_latest(cfg,'M4',stage,rd,bestp)
    print(f'M4-RH final overall={vm["overall_MAE"]:.6f} masked={vm["masked_MAE"]:.6f} support={vm["support_MAE"]:.6f} rel_masked={summary["relative_improvement_over_M0"]["masked"]:.4%} nonzero_alpha={summary["alpha_hr_nonzero_fraction"]:.3f}',flush=True)


def _masked_query_mae(pred,y,valid,direct):
    m=valid.bool() & direct.bool()[:,None,:]; e=torch.abs(pred-y)*m; return e.flatten(1).sum(1)/m.flatten(1).sum(1).clamp_min(1)


def _m5_predict(gate,router,zmem,zq,res,cand,base,ds,regions,alpha_hr,cfg,dev,return_stats=False):
    out=[]; gs=[]; gate.eval()
    with torch.no_grad():
        for ids in batches(len(zq),48,False,0):
            corr,_,_=_m4_corr(router,zmem,zq,res,cand,ids,ds,base,regions,alpha_hr,cfg,dev); b=t(base[ids],dev); obs=t(np.asarray(ds.obs[ids],np.float32).mean(1),dev); dm=t(ds.direct[ids],dev,torch.bool); sm=t(ds.support[ids],dev,torch.bool); g=gate(t(zq[ids],dev),b,corr,obs,dm,sm); out.append((b+g*corr).cpu().numpy()); gs.append(g.cpu().numpy())
    pred=np.concatenate(out); gg=np.concatenate(gs); stats={'gate_mean':float(gg.mean()),'gate_active_fraction_gt_0_5':float((gg>0.5).mean())}
    return (pred,stats) if return_stats else pred


def train_m5(cfg,stage,seed,art,tr,va,rd,dev):
    m1,_=load_m1(cfg,stage,dev); router,m4ck=load_m4(cfg,stage,dev); enc=m1.encoder; enc.eval(); router.eval(); ftr=normalize_features(art,'train'); fva=normalize_features(art,'val'); ztr=embeddings(enc,ftr,dev); zva=embeddings(enc,fva,dev); res=np.load(art/'train_residual.npy',mmap_mode='r'); tc=np.load(art/'train_candidates.npz'); vc=np.load(art/'val_candidates.npz'); base_tr=np.load(art/'train_base_pred.npy',mmap_mode='r'); base_va=np.load(art/'val_base_pred.npy',mmap_mode='r'); regions=np.load(art/'node_regions.npy').astype(np.int32); alpha_hr=np.asarray(m4ck['alpha_hr'],np.float32); P,N=base_tr.shape[1:]; c=cfg['model']['m5']
    gate=LocalSafeGate(ztr.shape[1],P,N,int(c['node_embedding_dim']),int(c['horizon_embedding_dim']),int(c['hidden'])).to(dev); opt=torch.optim.AdamW(gate.parameters(),lr=float(c['lr']),weight_decay=float(c['weight_decay'])); best=1e9; wait=0; bestp=rd/'best.pt'
    for ep in range(1,int(c['epochs'])+1):
        gate.train(); ls=[]; harms=[]
        for ids in batches(len(tr),int(c['query_batch_size']),True,seed+ep):
            with torch.no_grad(): corr,_,_=_m4_corr(router,ztr,ztr,res,tc,ids,tr,base_tr,regions,alpha_hr,cfg,dev); b=t(base_tr[ids],dev); q=t(ztr[ids],dev)
            obs=t(np.asarray(tr.obs[ids],np.float32).mean(1),dev); valid,direct,support=masks(tr,ids,dev); g=gate(q,b,corr,obs,direct,support); pred=b+g*corr; y=t(tr.y[ids],dev); forecast=weighted_mae_loss(pred,y,valid,direct,support,cfg['loss']); harm=torch.relu(per_query_mae_torch(pred,y,valid)-per_query_mae_torch(b,y,valid)).mean(); mharm=torch.relu(_masked_query_mae(pred,y,valid,direct)-_masked_query_mae(b,y,valid,direct)).mean(); loss=forecast+float(c['harm_lambda'])*harm+float(c['masked_harm_lambda'])*mharm+float(c['gate_l1_lambda'])*g.mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(gate.parameters(),5.0); opt.step(); ls.append(float(loss.detach())); harms.append(float(harm.detach()))
        predv,st=_m5_predict(gate,router,ztr,zva,res,vc,base_va,va,regions,alpha_hr,cfg,dev,True); vm=metrics(cfg,predv,va); score=objective_from_metrics(cfg,vm); print(f'M5-LOCAL epoch={ep:03d} loss={np.mean(ls):.5f} harm={np.mean(harms):.5f} val_obj={score:.5f} overall={vm["overall_MAE"]:.5f} masked={vm["masked_MAE"]:.5f} support={vm["support_MAE"]:.5f} gate={st["gate_mean"]:.3f}',flush=True)
        if score<best: best=score;wait=0;torch.save({'state_dict':gate.state_dict(),'model_type':'local_safe_gate','demb':ztr.shape[1],'P':P,'N':N,'epoch':ep,'val_objective':best,'node_dim':int(c['node_embedding_dim']),'horizon_dim':int(c['horizon_embedding_dim']),'hidden':int(c['hidden'])},bestp)
        else: wait+=1
        if wait>=int(c['patience']): break
    ck=torch.load(bestp,map_location=dev,weights_only=False); gate.load_state_dict(ck['state_dict']); predv,st=_m5_predict(gate,router,ztr,zva,res,vc,base_va,va,regions,alpha_hr,cfg,dev,True); np.save(rd/'val_predictions.npy',predv.astype(np.float32)); base_arr=np.asarray(base_va,np.float32); bm=metrics(cfg,base_arr,va); fm=metrics(cfg,predv,va); vv=np.asarray(va.valid,bool); den=vv.sum((1,2)).clip(min=1); be=(np.abs(base_arr-np.asarray(va.y))*vv).sum((1,2))/den; fe=(np.abs(predv-np.asarray(va.y))*vv).sum((1,2))/den; harm_frac=float(np.mean(fe>be+1e-8)); summary={'variant':'M5','stage':stage,'seed':seed,'validation':fm,'base_validation':bm,'relative_improvement_over_M0':{'overall':rel_improvement(bm,fm,'overall_MAE'),'masked':rel_improvement(bm,fm,'masked_MAE'),'support':rel_improvement(bm,fm,'support_MAE')},'harmful_augmentation_fraction_vs_M0':harm_frac,**st,'best_epoch':ck['epoch'],'description':'Node×horizon sensing-aware safe gate over graph-region×horizon local support correction','test_evaluated':False}; write_json(rd/'summary.json',summary); save_latest(cfg,'M5',stage,rd,bestp)
    print(f'M5-LOCAL final overall={fm["overall_MAE"]:.6f} masked={fm["masked_MAE"]:.6f} support={fm["support_MAE"]:.6f} rel_masked={summary["relative_improvement_over_M0"]["masked"]:.4%} harm_frac={harm_frac:.3f}',flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--variant',required=True,choices=['M0','M1','M2','M3','M4','M5']); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--seed',type=int,default=10); a=ap.parse_args(); cfg=load_yaml(ROOT/'configs/presure.yaml'); seed_all(a.seed); art=load_artifacts(cfg,a.stage); cache=cache_dir(cfg,a.stage); tr=CacheSplit(cache,'train'); va=CacheSplit(cache,'val'); dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); rd=run_dir(cfg,a.variant,a.stage,stamp(),a.seed); rd.mkdir(parents=True,exist_ok=True); bm=json.loads((art/'backbone_manifest.json').read_text()); rm=json.loads((art/'retrieval_manifest.json').read_text()); lineage={'variant':a.variant,'stage':a.stage,'seed':a.seed,'implementation_revision':cfg.get('implementation_revision'),'cache_fingerprint':bm.get('cache_identity',{}).get('cache_fingerprint'),'backbone_checkpoint':bm.get('checkpoint'),'backbone_checkpoint_sha256':bm.get('checkpoint_sha256'),'retrieval_memory_split':rm.get('memory_split'),'test_instantiated':False}; write_json(rd/'RUN_MANIFEST.json',{**lineage,'status':'STARTED'})
    try:
        {'M0':run_m0,'M1':train_m1,'M2':run_m2,'M3':train_m3,'M4':train_m4,'M5':train_m5}[a.variant](cfg,a.stage,a.seed,art,tr,va,rd,dev)
    except Exception as e:
        write_json(rd/'RUN_MANIFEST.json',{**lineage,'status':'FAILED','error':repr(e)}); raise
    write_json(rd/'RUN_MANIFEST.json',{**lineage,'status':'FINISHED'}); print(f'{a.variant} {a.stage} PASS run_dir={rd}')


if __name__=='__main__': main()
