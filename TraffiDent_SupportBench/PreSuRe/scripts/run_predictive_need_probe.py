#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, copy, csv, json, math, sys
import numpy as np
import torch

ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe')
ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from presure.utils import load_yaml, write_json, stamp, seed_all
from presure.contract import cache_dir
from presure.data import CacheSplit, cache_stats
from presure.features import NODE_SENSE_DIM
from presure.models import NodeUsefulnessNet
from presure.probes import PredictiveNeedNet, SupportUtilityProbeNet, exact_spearman_sample, pearson, binary_auroc
from presure.pipeline import load_artifacts, load_m1, load_m3, run_dir


def t(a,dev,dtype=torch.float32):
    return torch.as_tensor(np.array(a,copy=True),device=dev,dtype=dtype)


def batches(ids,bs,shuffle=False,seed=0):
    ids=np.asarray(ids,np.int64).copy()
    if shuffle: np.random.default_rng(seed).shuffle(ids)
    for a in range(0,len(ids),int(bs)): yield ids[a:a+int(bs)]


def _sense(x): return np.asarray(x[...,-NODE_SENSE_DIM:],np.float32)


def _split_ids(n,cfg,seed):
    c=cfg['predictive_need_probe']['split']
    counts=[int(c['need_train_queries']),int(c['need_dev_queries']),int(c['utility_train_queries']),int(c['utility_dev_queries'])]
    if sum(counts)>n: raise RuntimeError(f'Probe split needs {sum(counts)} Train queries but only {n} exist')
    ids=np.arange(n,dtype=np.int64); np.random.default_rng(seed+1701).shuffle(ids)
    a=0; out={}
    for name,k in zip(['need_train','need_dev','utility_train','utility_dev'],counts): out[name]=np.sort(ids[a:a+k]);a+=k
    return out


def need_loss(pred,target,valid,cfg):
    c=cfg['predictive_need_probe']['need_model']; vm=valid.bool()
    if not torch.any(vm): return pred.sum()*0, pred.sum()*0, pred.sum()*0
    hub=torch.nn.functional.smooth_l1_loss(pred,target,reduction='none',beta=float(c['huber_beta']))
    mag=torch.clamp(torch.abs(target),0,float(c['target_weight_clip']))
    w=1.0+float(c['magnitude_weight'])*mag
    reg=(hub[vm]*w[vm]).sum()/w[vm].sum().clamp_min(1.0)
    sig=(torch.abs(target)>=float(c['sign_min_abs_norm'])) & vm
    if torch.any(sig):
        sgn=torch.where(target[sig]>=0,torch.ones_like(target[sig]),-torch.ones_like(target[sig]))
        sl=torch.nn.functional.softplus(-sgn*pred[sig]).mean()
    else: sl=pred.sum()*0
    return reg+float(c['sign_weight'])*sl,reg,sl


def _need_eval(model,node_feat,base,ds,ids,mean,std,cfg,dev):
    c=cfg['predictive_need_probe']['need_model']; preds=[]; tgts=[]; valids=[]; directs=[]; supports=[]
    model.eval()
    with torch.no_grad():
        for bi in batches(ids,int(c['eval_batch_size'])):
            x=t(node_feat[bi],dev); pn=model(x).cpu().numpy()*float(std)
            target=np.asarray(ds.y[bi],np.float32)-np.asarray(base[bi],np.float32)
            preds.append(pn.astype(np.float32));tgts.append(target);valids.append(np.asarray(ds.valid[bi],bool));directs.append(np.asarray(ds.direct[bi],bool));supports.append(np.asarray(ds.support[bi],bool))
    p=np.concatenate(preds); y=np.concatenate(tgts); v=np.concatenate(valids); d=np.concatenate(directs)[:,None,:]; s=np.concatenate(supports)[:,None,:]
    def view(mask):
        z=mask & v
        if not np.any(z): return {}
        e=p[z]-y[z]; az=np.abs(y[z]); sig=az>=float(c['sign_min_abs_raw'])
        zero_mae=float(np.mean(az)); mae=float(np.mean(np.abs(e))); rmse=float(np.sqrt(np.mean(e*e)))
        out={'count':int(z.sum()),'MAE_raw':mae,'RMSE_raw':rmse,'zero_predictor_MAE_raw':zero_mae,'MAE_improvement_over_zero':float((zero_mae-mae)/zero_mae) if zero_mae>0 else 0.0}
        if np.any(sig): out['sign_accuracy_meaningful']=float(np.mean(np.sign(p[z][sig])==np.sign(y[z][sig])));out['meaningful_count']=int(sig.sum())
        else: out['sign_accuracy_meaningful']=None;out['meaningful_count']=0
        out['pearson']=pearson(p[z],y[z]); out['spearman_sample']=exact_spearman_sample(p[z],y[z],int(c['correlation_sample_points']),int(cfg['experiment']['seed'])+91)
        # High-error group threshold is view-local q75 of |true base error|.
        q75=float(np.quantile(az,0.75)); high=az>=q75
        out['high_error_q75_raw']=q75; out['high_error_MAE_raw']=float(np.mean(np.abs(e[high])))
        out['high_error_sign_accuracy']=float(np.mean(np.sign(p[z][high])==np.sign(y[z][high]))) if np.any(high) else None
        return out
    return p, {'overall':view(np.ones_like(v,bool)),'masked':view(d),'support':view(s)}


def train_need(cfg,stage,seed,art,tr,va,idsplit,rd,dev):
    c=cfg['predictive_need_probe']['need_model']; nft=np.load(art/'train_node_features.npy',mmap_mode='r'); nfv=np.load(art/'val_node_features.npy',mmap_mode='r'); bt=np.load(art/'train_base_pred.npy',mmap_mode='r'); bv=np.load(art/'val_base_pred.npy',mmap_mode='r'); mean,std=cache_stats(cache_dir(cfg,stage)); H=int(cfg['experiment']['history']);P=int(cfg['experiment']['horizon'])
    args=dict(din=nft.shape[-1],N=nft.shape[1],P=P,base_start=4*H,hidden=int(c['hidden']),demb=int(c['embedding_dim']),node_dim=int(c['node_embedding_dim']),horizon_dim=int(c['horizon_embedding_dim']),dropout=float(c['dropout']))
    model=PredictiveNeedNet(**args).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=float(c['lr']),weight_decay=float(c['weight_decay'])); best=1e99;wait=0;bestp=rd/'need_best.pt'
    for ep in range(1,int(c['epochs'])+1):
        model.train(); ls=[];regs=[];signs=[]
        for bi in batches(idsplit['need_train'],int(c['batch_size']),True,seed+2000+ep):
            x=t(nft[bi],dev); target=(t(tr.y[bi],dev)-t(bt[bi],dev))/float(std); valid=t(tr.valid[bi],dev,torch.bool); pred=model(x); loss,reg,sl=need_loss(pred,target,valid,cfg)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(c['grad_clip']));opt.step();ls.append(float(loss.detach()));regs.append(float(reg.detach()));signs.append(float(sl.detach()))
        # internal Need-dev objective: normalized valid-cell MAE only; disjoint from Utility train/dev and final validation.
        model.eval(); ae=cnt=0.0
        with torch.no_grad():
            for bi in batches(idsplit['need_dev'],int(c['eval_batch_size'])):
                pred=model(t(nft[bi],dev)); target=(t(tr.y[bi],dev)-t(bt[bi],dev))/float(std); valid=t(tr.valid[bi],dev,torch.bool); ae+=float(torch.abs(pred-target)[valid].sum().cpu());cnt+=int(valid.sum().cpu())
        score=ae/max(cnt,1);print(f'NEED epoch={ep:03d} loss={np.mean(ls):.5f} reg={np.mean(regs):.5f} sign={np.mean(signs):.5f} dev_mae_norm={score:.5f}',flush=True)
        if score<best:
            best=score;wait=0;torch.save({'state_dict':model.state_dict(),'model_args':args,'model_type':'predictive_need_probe_v071','epoch':ep,'dev_mae_norm':score},bestp)
        else: wait+=1
        if wait>=int(c['patience']): break
    ck=torch.load(bestp,map_location=dev,weights_only=False);model.load_state_dict(ck['state_dict']);model.eval()
    val_ids=np.arange(min(len(va),int(cfg['predictive_need_probe']['split']['final_val_queries'])),dtype=np.int64)
    # deterministic shuffled validation subset avoids selecting only earliest times.
    rng=np.random.default_rng(seed+1702);allv=np.arange(len(va),dtype=np.int64);rng.shuffle(allv);val_ids=np.sort(allv[:len(val_ids)])
    _,metrics=_need_eval(model,nfv,bv,va,val_ids,mean,std,cfg,dev)
    # Cache deployment-safe need predictions for the Utility B probe on its disjoint train/dev subsets and final val subset.
    pred_files={}
    for name,ids,nodef in [('utility_train',idsplit['utility_train'],nft),('utility_dev',idsplit['utility_dev'],nft),('final_val',val_ids,nfv)]:
        path=rd/f'need_pred_{name}.npy'; mm=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(len(ids),P,nft.shape[1]));
        with torch.no_grad():
            for j,bi in enumerate(batches(np.arange(len(ids)),int(c['eval_batch_size']))):
                src=ids[bi]; mm[bi]=model(t(nodef[src],dev)).cpu().numpy().astype(np.float16)
        del mm;pred_files[name]=str(path)
    np.save(rd/'utility_train_ids.npy',idsplit['utility_train']);np.save(rd/'utility_dev_ids.npy',idsplit['utility_dev']);np.save(rd/'final_val_ids.npy',val_ids)
    write_json(rd/'need_summary.json',{'model_type':'predictive_need_probe_v071','best_epoch':int(ck['epoch']),'dev_mae_norm':float(ck['dev_mae_norm']),'validation_future_used_for_diagnostic_only':True,'deployable_inputs_only':True,'validation':metrics,'prediction_files':pred_files})
    print('NEED FINAL',json.dumps(metrics['overall'],sort_keys=True),flush=True)
    return model,val_ids,metrics


def _load_embeddings(art):
    a=art/'train_node_embeddings_v05.npy';b=art/'val_node_embeddings_v05.npy'
    if not a.exists() or not b.exists(): raise RuntimeError('Need/M3 probe requires cached M1 node embeddings from v0.7 M3 run; rerun M3 once if missing')
    return np.load(a,mmap_mode='r'),np.load(b,mmap_mode='r')


def _m3_top4(m3,zmem,zq,node_mem,node_q,cand,res,ids,cfg,dev):
    K=min(int(cfg['node_routing']['utility_candidate_k']),cand['indices'].shape[1]);L=int(cfg['node_routing']['m4_top_k'])
    ci=np.asarray(cand['indices'][ids,:K],np.int64); sim=np.asarray(cand['scores'][ids,:K],np.float32)
    q=t(zq[ids],dev); c=t(zmem[ci],dev); qs=t(_sense(node_q[ids]),dev); cs=t(_sense(node_mem[ci]),dev)
    with torch.no_grad(): logits,score,_=m3(q,c,t(sim,dev),qs,cs)
    p=torch.softmax(logits,-1)[...,2]; combined=score+torch.log(p+1e-6); vals,loc=torch.topk(combined,min(L,K),dim=1)
    loc_bn=loc.permute(0,2,1).contiguous(); p_bn=p.permute(0,2,1); psel=torch.gather(p_bn,2,loc_bn)
    conf=torch.softmax(vals,dim=1).permute(0,2,1).contiguous(); mask=psel>=float(cfg['model']['m3'].get('router_use_threshold',cfg['model']['m3']['use_threshold']))
    cids=t(ci,dev,torch.long); gid=torch.gather(cids[:,None,:].expand(-1,loc_bn.shape[1],-1),2,loc_bn).cpu().numpy().astype(np.int32)
    return gid,conf.cpu().numpy().astype(np.float16),mask.cpu().numpy().astype(bool)


def prepare_top4_cache(cfg,stage,seed,art,idsplit,val_ids,rd,dev):
    m3,_,_=load_m3(cfg,stage,dev);m3.eval();zt,zv=_load_embeddings(art);nft=np.load(art/'train_node_features.npy',mmap_mode='r');nfv=np.load(art/'val_node_features.npy',mmap_mode='r');tc=np.load(art/'train_candidates.npz');vc=np.load(art/'val_candidates.npz');res=np.load(art/'train_residual.npy',mmap_mode='r');bs=int(cfg['predictive_need_probe']['utility_model']['selection_batch_size']);out={}
    for name,ids,zq,nodeq,cand in [('utility_train',idsplit['utility_train'],zt,nft,tc),('utility_dev',idsplit['utility_dev'],zt,nft,tc),('final_val',val_ids,zv,nfv,vc)]:
        N=nft.shape[1];L=int(cfg['node_routing']['m4_top_k']); gp=rd/f'top4_ids_{name}.npy';cp=rd/f'top4_conf_{name}.npy';mp=rd/f'top4_mask_{name}.npy';gmm=np.lib.format.open_memmap(gp,mode='w+',dtype=np.int32,shape=(len(ids),N,L));cmm=np.lib.format.open_memmap(cp,mode='w+',dtype=np.float16,shape=(len(ids),N,L));mmm=np.lib.format.open_memmap(mp,mode='w+',dtype=np.bool_,shape=(len(ids),N,L))
        for bi in batches(np.arange(len(ids)),bs):
            src=ids[bi];g,cf,m=_m3_top4(m3,zt,zq,nft,nodeq,cand,res,src,cfg,dev);gmm[bi]=g;cmm[bi]=cf;mmm[bi]=m
        del gmm,cmm,mmm;out[name]={'ids':str(gp),'conf':str(cp),'mask':str(mp)}
    write_json(rd/'top4_cache_manifest.json',out);return out


def _gather_support_context(node_mem,res,gids):
    # gids [B,N,L]
    N=gids.shape[1];ni=np.arange(N,dtype=np.int64)[None,:,None]
    ctx=np.asarray(node_mem)[gids,ni,:].astype(np.float32,copy=False)
    rr=np.asarray(res)[gids,:,ni].astype(np.float32,copy=False)
    return ctx,rr


def utility_targets(need_norm,rr_norm,candidate_mask,cfg):
    # need [B,P,N] -> [B,N,L,P] target relative utility, max over alpha grid.
    c=cfg['predictive_need_probe']['utility_model'];n=need_norm.permute(0,2,1)[:,:,None,:];base=torch.abs(n);best=torch.full_like(rr_norm,-1e9)
    for a in c['alpha_grid']:
        gain=base-torch.abs(n-float(a)*rr_norm);best=torch.maximum(best,gain)
    rel=best/(base+float(c['gain_floor']));rel=torch.clamp(rel,-float(c['utility_clip']),float(c['utility_clip']))
    return rel.masked_fill(~candidate_mask[:,:,:,None],-float(c['utility_clip']))


def utility_loss(pred,target,cmask,valid,cfg):
    c=cfg['predictive_need_probe']['utility_model'];vm=valid.bool().permute(0,2,1)[:,:,None,:].expand_as(pred);use=vm & cmask[:,:,:,None].expand_as(pred)
    if not torch.any(use): z=pred.sum()*0;return z,z,z,z,torch.tensor(0.0,device=pred.device)
    reg=torch.nn.functional.smooth_l1_loss(pred,target,reduction='none',beta=float(c['huber_beta']))[use].mean()
    label=(target>float(c['beneficial_threshold'])).float();cls=torch.nn.functional.binary_cross_entropy_with_logits(pred/float(c['classification_temperature']),label,reduction='none')[use].mean()
    hi=target.masked_fill(~use,-1e6).argmax(2);lo=target.masked_fill(~use,1e6).argmin(2);sp=torch.gather(pred,2,hi[:,:,None,:]).squeeze(2);sn=torch.gather(pred,2,lo[:,:,None,:]).squeeze(2);tp=torch.gather(target,2,hi[:,:,None,:]).squeeze(2);tn=torch.gather(target,2,lo[:,:,None,:]).squeeze(2);enough=(use.sum(2)>=2)&(tp-tn>float(c['rank_min_gap']))&valid.bool().permute(0,2,1)
    if torch.any(enough): rank=torch.relu(float(c['rank_margin'])-sp[enough]+sn[enough]).mean();racc=(sp[enough]>sn[enough]).float().mean()
    else: rank=pred.sum()*0;racc=torch.tensor(0.0,device=pred.device)
    loss=reg+float(c['rank_weight'])*rank+float(c['classification_weight'])*cls
    return loss,reg,rank,cls,racc


def _utility_batch(mode,model,node_q,node_mem,res,base,ds,src_ids,top4_ids,top4_conf,top4_mask,need_pred_norm,mean,std,cfg,dev):
    H=int(cfg['experiment']['history']);P=int(cfg['experiment']['horizon']);qctx=t(node_q[src_ids],dev);gids=np.asarray(top4_ids,np.int64);cctx_np,rr_np=_gather_support_context(node_mem,res,gids);cctx=t(cctx_np,dev);rr=t(rr_np,dev)/float(std);qs=t(_sense(node_q[src_ids]),dev);ss=t(_sense(cctx_np),dev);conf=t(top4_conf,dev);cm=t(top4_mask,dev,torch.bool);b=t(base[src_ids],dev);qbn=(b-float(mean))/float(std);sbase=cctx[...,4*H:4*H+P]
    true_need=(t(ds.y[src_ids],dev)-b)/float(std)
    if mode=='A': need=torch.zeros_like(true_need)
    elif mode=='B': need=t(need_pred_norm,dev)
    elif mode=='C': need=true_need
    else: raise ValueError(mode)
    pred=model(qctx,cctx,qs,ss,rr,qbn,sbase,conf,need);target=utility_targets(true_need,rr,cm,cfg);valid=t(ds.valid[src_ids],dev,torch.bool)
    return pred,target,cm,valid


def _utility_eval(mode,model,node_q,node_mem,res,base,ds,ids,topcache,need_pred,mean,std,cfg,dev,seed):
    c=cfg['predictive_need_probe']['utility_model'];bs=int(c['eval_batch_size']);top_ids=np.load(topcache['ids'],mmap_mode='r');top_conf=np.load(topcache['conf'],mmap_mode='r');top_mask=np.load(topcache['mask'],mmap_mode='r');sample_s=[];sample_y=[];rng=np.random.default_rng(seed+3100+ord(mode));top1_ok=top1_n=rank_ok=rank_n=benef_top1=benef_n=0
    model.eval()
    with torch.no_grad():
        for bi in batches(np.arange(len(ids)),bs):
            src=ids[bi];npred=None if mode!='B' else np.asarray(need_pred[bi],np.float32);pred,target,cm,valid=_utility_batch(mode,model,node_q,node_mem,res,base,ds,src,np.asarray(top_ids[bi]),np.asarray(top_conf[bi]),np.asarray(top_mask[bi]),npred,mean,std,cfg,dev)
            vm=valid.bool().permute(0,2,1);use=cm[:,:,:,None]&vm[:,:,None,:]
            masked_pred=pred.masked_fill(~use,-1e6);masked_tar=target.masked_fill(~use,-1e6);pi=masked_pred.argmax(2);ti=masked_tar.argmax(2);ok=(use.sum(2)>0)&vm;top1_ok+=int(((pi==ti)&ok).sum().cpu());top1_n+=int(ok.sum().cpu())
            chosen=torch.gather(target,2,pi[:,:,None,:]).squeeze(2);benef_top1+=int(((chosen>float(c['beneficial_threshold']))&ok).sum().cpu());benef_n+=int(ok.sum().cpu())
            hi=masked_tar.argmax(2);lo=target.masked_fill(~use,1e6).argmin(2);sp=torch.gather(pred,2,hi[:,:,None,:]).squeeze(2);sn=torch.gather(pred,2,lo[:,:,None,:]).squeeze(2);tp=torch.gather(target,2,hi[:,:,None,:]).squeeze(2);tn=torch.gather(target,2,lo[:,:,None,:]).squeeze(2);en=(use.sum(2)>=2)&(tp-tn>float(c['rank_min_gap']))&vm;rank_ok+=int(((sp>sn)&en).sum().cpu());rank_n+=int(en.sum().cpu())
            # sample valid candidate-horizon cells for correlation/AUROC diagnostics
            pp=pred[use].cpu().numpy();yy=target[use].cpu().numpy();take=min(len(pp),int(c['sample_points_per_batch']))
            if take>0:
                ii=rng.choice(len(pp),size=take,replace=False) if len(pp)>take else np.arange(len(pp));sample_s.append(pp[ii]);sample_y.append(yy[ii])
    ss=np.concatenate(sample_s) if sample_s else np.array([],np.float32);yy=np.concatenate(sample_y) if sample_y else np.array([],np.float32);label=yy>float(c['beneficial_threshold'])
    return {'mode':mode,'deployable':mode!='C','top1_agreement':float(top1_ok/max(top1_n,1)),'pairwise_rank_accuracy':float(rank_ok/max(rank_n,1)),'beneficial_top1_hit_rate':float(benef_top1/max(benef_n,1)),'pearson':pearson(ss,yy),'spearman_sample':exact_spearman_sample(ss,yy,int(c['correlation_sample_points']),seed+333),'beneficial_auroc':binary_auroc(ss,label,int(c['auroc_sample_points']),seed+444),'sampled_cells':int(len(ss))}


def analytic_predicted_need_utility(cfg,stage,seed,art,va,val_ids,rd,topcache,dev):
    """No learned utility student: infer utility analytically from predicted Need.

    This is the cleanest test of whether the predicted future error state contains the
    information required to match Top-4 historical residuals. It removes utility-network
    optimization as a confound.
    """
    c=cfg['predictive_need_probe']['utility_model']; nft=np.load(art/'train_node_features.npy',mmap_mode='r'); nfv=np.load(art/'val_node_features.npy',mmap_mode='r');res=np.load(art/'train_residual.npy',mmap_mode='r');bv=np.load(art/'val_base_pred.npy',mmap_mode='r');_,std=cache_stats(cache_dir(cfg,stage));top_ids=np.load(topcache['final_val']['ids'],mmap_mode='r');top_mask=np.load(topcache['final_val']['mask'],mmap_mode='r');need_pred=np.load(rd/'need_pred_final_val.npy',mmap_mode='r');rng=np.random.default_rng(seed+7201);sample_s=[];sample_y=[];top1_ok=top1_n=rank_ok=rank_n=benef_top1=benef_n=0
    for bi in batches(np.arange(len(val_ids)),int(c['eval_batch_size'])):
        src=val_ids[bi];gids=np.asarray(top_ids[bi],np.int64);cm=t(top_mask[bi],dev,torch.bool);_,rr_np=_gather_support_context(nft,res,gids);rr=t(rr_np,dev)/float(std);b=t(bv[src],dev);true_need=(t(va.y[src],dev)-b)/float(std);pred_need=t(need_pred[bi],dev);valid=t(va.valid[src],dev,torch.bool)
        target=utility_targets(true_need,rr,cm,cfg);score=utility_targets(pred_need,rr,cm,cfg);vm=valid.bool().permute(0,2,1);use=cm[:,:,:,None]&vm[:,:,None,:]
        ms=score.masked_fill(~use,-1e6);mt=target.masked_fill(~use,-1e6);pi=ms.argmax(2);ti=mt.argmax(2);ok=(use.sum(2)>0)&vm;top1_ok+=int(((pi==ti)&ok).sum().cpu());top1_n+=int(ok.sum().cpu());chosen=torch.gather(target,2,pi[:,:,None,:]).squeeze(2);benef_top1+=int(((chosen>float(c['beneficial_threshold']))&ok).sum().cpu());benef_n+=int(ok.sum().cpu())
        hi=mt.argmax(2);lo=target.masked_fill(~use,1e6).argmin(2);sp=torch.gather(score,2,hi[:,:,None,:]).squeeze(2);sn=torch.gather(score,2,lo[:,:,None,:]).squeeze(2);tp=torch.gather(target,2,hi[:,:,None,:]).squeeze(2);tn=torch.gather(target,2,lo[:,:,None,:]).squeeze(2);en=(use.sum(2)>=2)&(tp-tn>float(c['rank_min_gap']))&vm;rank_ok+=int(((sp>sn)&en).sum().cpu());rank_n+=int(en.sum().cpu())
        pp=score[use].cpu().numpy();yy=target[use].cpu().numpy();take=min(len(pp),int(c['sample_points_per_batch']));ii=rng.choice(len(pp),size=take,replace=False) if len(pp)>take else np.arange(len(pp));sample_s.append(pp[ii]);sample_y.append(yy[ii])
    ss=np.concatenate(sample_s) if sample_s else np.array([],np.float32);yy=np.concatenate(sample_y) if sample_y else np.array([],np.float32);lab=yy>float(c['beneficial_threshold'])
    out={'mode':'B_ANALYTIC','deployable':True,'top1_agreement':float(top1_ok/max(top1_n,1)),'pairwise_rank_accuracy':float(rank_ok/max(rank_n,1)),'beneficial_top1_hit_rate':float(benef_top1/max(benef_n,1)),'pearson':pearson(ss,yy),'spearman_sample':exact_spearman_sample(ss,yy,int(c['correlation_sample_points']),seed+7222),'beneficial_auroc':binary_auroc(ss,lab,int(c['auroc_sample_points']),seed+7233),'sampled_cells':int(len(ss)),'description':'Analytic support utility computed from predicted Need and historical residuals; no learned utility network.'}
    write_json(rd/'utility_B_analytic_summary.json',out);print('UTILITY FINAL B_ANALYTIC',json.dumps(out,sort_keys=True),flush=True);return out


def train_utility_modes(cfg,stage,seed,art,tr,va,idsplit,val_ids,rd,topcache,dev):
    c=cfg['predictive_need_probe']['utility_model'];nft=np.load(art/'train_node_features.npy',mmap_mode='r');nfv=np.load(art/'val_node_features.npy',mmap_mode='r');res=np.load(art/'train_residual.npy',mmap_mode='r');bt=np.load(art/'train_base_pred.npy',mmap_mode='r');bv=np.load(art/'val_base_pred.npy',mmap_mode='r');mean,std=cache_stats(cache_dir(cfg,stage));P=int(cfg['experiment']['horizon']);H=int(cfg['experiment']['history']);args=dict(node_feature_dim=nft.shape[-1],P=P,sense_dim=NODE_SENSE_DIM,context_dim=int(c['context_dim']),residual_dim=int(c['residual_dim']),pair_dim=int(c['pair_dim']),hidden=int(c['hidden']),horizon_dim=int(c['horizon_embedding_dim']),attention_heads=int(c['attention_heads']),dropout=float(c['dropout']))
    seed_all(seed+3400);template=SupportUtilityProbeNet(**args);init=copy.deepcopy(template.state_dict());results={}
    need_files={'utility_train':np.load(rd/'need_pred_utility_train.npy',mmap_mode='r'),'utility_dev':np.load(rd/'need_pred_utility_dev.npy',mmap_mode='r'),'final_val':np.load(rd/'need_pred_final_val.npy',mmap_mode='r')}
    for mode in ['A','B','C']:
        seed_all(seed+3400);model=SupportUtilityProbeNet(**args).to(dev);model.load_state_dict(init);opt=torch.optim.AdamW(model.parameters(),lr=float(c['lr']),weight_decay=float(c['weight_decay']));best=1e99;wait=0;bestp=rd/f'utility_{mode}_best.pt';tc={k:np.load(vv['ids'],mmap_mode='r') for k,vv in topcache.items()};cf={k:np.load(vv['conf'],mmap_mode='r') for k,vv in topcache.items()};cm={k:np.load(vv['mask'],mmap_mode='r') for k,vv in topcache.items()}
        for ep in range(1,int(c['epochs'])+1):
            model.train();ls=[];rs=[]
            for bi in batches(np.arange(len(idsplit['utility_train'])),int(c['batch_size']),True,seed+3500+ep):
                src=idsplit['utility_train'][bi];npred=None if mode!='B' else np.asarray(need_files['utility_train'][bi],np.float32);pred,target,mask,valid=_utility_batch(mode,model,nft,nft,res,bt,tr,src,np.asarray(tc['utility_train'][bi]),np.asarray(cf['utility_train'][bi]),np.asarray(cm['utility_train'][bi]),npred,mean,std,cfg,dev);loss,reg,rank,cls,racc=utility_loss(pred,target,mask,valid,cfg);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(c['grad_clip']));opt.step();ls.append(float(loss.detach()));rs.append(float(racc.detach()))
            # internal utility-dev score = 1-rank_acc + 0.25*(1-top1), fully disjoint from final validation.
            dmet=_utility_eval(mode,model,nft,nft,res,bt,tr,idsplit['utility_dev'],topcache['utility_dev'],need_files['utility_dev'],mean,std,cfg,dev,seed+ep)
            score=(1.0-float(dmet['pairwise_rank_accuracy']))+0.25*(1.0-float(dmet['top1_agreement']))
            print(f'UTILITY-{mode} epoch={ep:03d} loss={np.mean(ls):.5f} train_rank={np.mean(rs):.3f} dev_rank={dmet["pairwise_rank_accuracy"]:.3f} dev_top1={dmet["top1_agreement"]:.3f} dev_spearman={dmet["spearman_sample"]:.3f}',flush=True)
            if score<best:
                best=score;wait=0;torch.save({'state_dict':model.state_dict(),'model_args':args,'model_type':'support_utility_probe_v071','mode':mode,'epoch':ep,'dev_score':score},bestp)
            else: wait+=1
            if wait>=int(c['patience']): break
        ck=torch.load(bestp,map_location=dev,weights_only=False);model.load_state_dict(ck['state_dict']);met=_utility_eval(mode,model,nfv,nft,res,bv,va,val_ids,topcache['final_val'],need_files['final_val'],mean,std,cfg,dev,seed+5000+ord(mode));met['best_epoch']=int(ck['epoch']);met['privileged_true_need_input']=bool(mode=='C');results[mode]=met;write_json(rd/f'utility_{mode}_summary.json',met);print('UTILITY FINAL',mode,json.dumps(met,sort_keys=True),flush=True)
    return results


def decision(need_metrics,utility,analytic_b,cfg):
    th=cfg['predictive_need_probe']['decision'];n=need_metrics['overall'];A=utility['A'];B=utility['B'];C=utility['C'];need_ok=(float(n.get('sign_accuracy_meaningful') or 0)>=float(th['need_sign_accuracy_min']) and float(n.get('spearman_sample') or 0)>=float(th['need_spearman_min']));b_gain=float(B['pairwise_rank_accuracy'])-float(A['pairwise_rank_accuracy']);b_ok=float(B['pairwise_rank_accuracy'])>=float(th['utility_b_rank_min']) and b_gain>=float(th['utility_b_gain_over_a_min']);analytic_ok=float(analytic_b['pairwise_rank_accuracy'])>=float(th['analytic_b_rank_min']);c_ok=float(C['pairwise_rank_accuracy'])>=float(th['privileged_c_rank_min'])
    if need_ok and analytic_ok: status='GO_V08'
    elif c_ok and (need_ok or analytic_ok or float(analytic_b['pairwise_rank_accuracy'])>=float(th['analytic_b_rank_conditional']) or b_gain>=float(th['utility_b_gain_over_a_conditional'])): status='CONDITIONAL_V08'
    else: status='NO_GO_V08'
    reasons={'need_signal_pass':need_ok,'learned_utility_B_pass':b_ok,'analytic_predicted_need_utility_pass':analytic_ok,'privileged_C_pass':c_ok,'utility_B_rank_gain_over_A':b_gain,'analytic_B_pairwise_rank_accuracy':float(analytic_b['pairwise_rank_accuracy']),'thresholds':th}
    return status,reasons


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development',choices=['development']);ap.add_argument('--seed',type=int,default=10);a=ap.parse_args();cfg=load_yaml(ROOT/'configs/presure.yaml');seed_all(a.seed);art=load_artifacts(cfg,a.stage);cache=cache_dir(cfg,a.stage);tr=CacheSplit(cache,'train');va=CacheSplit(cache,'val');dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');rd=run_dir(cfg,'PredictiveNeedProbe',a.stage,stamp(),a.seed);rd.mkdir(parents=True,exist_ok=True);idsplit=_split_ids(len(tr),cfg,a.seed);write_json(rd/'split_manifest.json',{k:v.tolist() for k,v in idsplit.items()})
    lineage={'variant':'PredictiveNeedProbe','stage':a.stage,'seed':a.seed,'implementation_revision':cfg.get('implementation_revision'),'deployable_probe_inputs_exclude_direct_support_masks':True,'validation_future_used_for_diagnostic_only':True,'utility_C_is_privileged_non_deployable':True};write_json(rd/'RUN_MANIFEST.json',{**lineage,'status':'STARTED'})
    try:
        # Check v0.7 M1/M3 prerequisites before spending probe compute.
        load_m1(cfg,a.stage,dev);load_m3(cfg,a.stage,dev)
        need,val_ids,need_metrics=train_need(cfg,a.stage,a.seed,art,tr,va,idsplit,rd,dev)
        topcache=prepare_top4_cache(cfg,a.stage,a.seed,art,idsplit,val_ids,rd,dev)
        analytic_b=analytic_predicted_need_utility(cfg,a.stage,a.seed,art,va,val_ids,rd,topcache,dev)
        utility=train_utility_modes(cfg,a.stage,a.seed,art,tr,va,idsplit,val_ids,rd,topcache,dev)
        status,reasons=decision(need_metrics,utility,analytic_b,cfg)
        summary={'variant':'PredictiveNeedProbe','stage':a.stage,'seed':a.seed,'predictive_need_validation':need_metrics,'analytic_predicted_need_utility_validation':analytic_b,'support_utility_validation':utility,'v08_recommendation':status,'decision_diagnostics':reasons,'interpretation':{'A':'query/support only; deployable','B':'query/support + predicted need; deployable','B_ANALYTIC':'utility computed directly from predicted need + support residual; deployable','C':'query/support + true need; privileged diagnostic only'},'test_evaluated':False}
        write_json(rd/'summary.json',summary);(rd/'DECISION.txt').write_text(status+'\n'+json.dumps(reasons,indent=2,sort_keys=True)+'\n')
        rows=[{'probe':'Need','view':'overall',**need_metrics['overall']},{'probe':'Utility','view':'B_ANALYTIC',**analytic_b}] + [{'probe':'Utility','view':m,**utility[m]} for m in ['A','B','C']]
        keys=sorted(set().union(*(r.keys() for r in rows)))
        with open(rd/'probe_summary.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
        write_json(Path(cfg['project']['exp_root'])/'PredictiveNeedProbe'/cfg['experiment']['subgraph']/a.stage/'LATEST.json',{'run_dir':str(rd),'summary':str(rd/'summary.json'),'recommendation':status,'implementation_revision':cfg.get('implementation_revision')});write_json(rd/'RUN_MANIFEST.json',{**lineage,'status':'FINISHED','recommendation':status})
        print('PREDICTIVE NEED PROBE FINAL',status,flush=True);print(json.dumps(summary,indent=2,sort_keys=True),flush=True);print(f'PredictiveNeedProbe PASS run_dir={rd}',flush=True)
    except Exception as e:
        write_json(rd/'RUN_MANIFEST.json',{**lineage,'status':'FAILED','error':repr(e)});raise

if __name__=='__main__': main()
