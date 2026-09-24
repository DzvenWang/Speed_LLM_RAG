#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from presure.p1_config import load_config
from presure.contract import cache_dir
from presure.p1_paths import artifact_dir, variant_dir
from presure.p1_data import P1Dataset, VARIANTS
from presure.p1_model import P1ResidualAdapter
from presure.p1_loss import weighted_mae
from presure.metrics import metric_dict_np, improvement_pct
from presure.utils import seed_all, log, write_json


def ids_for(cfg,n_train,n_val):
    seed=int(cfg['experiment']['seed']); rng=np.random.default_rng(seed)
    tr=rng.permutation(n_train)[:min(int(cfg['training']['train_queries']),n_train)]
    vp=rng.permutation(n_val); dev=vp[:min(int(cfg['training']['dev_queries']),n_val)]
    return tr.astype(np.int64),dev.astype(np.int64)


def loader(ds,batch,shuffle,cfg):
    nw=int(cfg['runtime'].get('num_workers',0)); pin=bool(cfg['runtime'].get('pin_memory',True)) and torch.cuda.is_available(); kw={'num_workers':nw,'pin_memory':pin,'drop_last':False,'shuffle':shuffle}
    if nw>0: kw.update(prefetch_factor=int(cfg['runtime'].get('prefetch_factor',2)),persistent_workers=bool(cfg['runtime'].get('persistent_workers',True)))
    return DataLoader(ds,batch_size=int(batch),**kw)


def move(b,d): return {k:(v.to(d,non_blocking=True) if torch.is_tensor(v) else v) for k,v in b.items()}


def run(model,dl,d,weights,opt=None):
    train=opt is not None; model.train(train); total=den=0.; preds=[]; delta=[]
    ctx=torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in dl:
            b=move(b,d)
            if train: opt.zero_grad(set_to_none=True)
            p,dd=model(b['base'],b['latent'],b['qhist'],b['shist'],b['sfuture'],b['scov'])
            loss,_=weighted_mae(p,b['target_norm'],b['valid'],b['direct'],b['support'],weights)
            if train:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            n=int(b['valid'].sum().item()); total+=float(loss.detach())*max(n,1); den+=max(n,1)
            if not train: preds.append(p.detach().cpu().numpy()); delta.append(dd.detach().abs().cpu().numpy())
    if train: return total/max(den,1),None,None
    return total/max(den,1),np.concatenate(preds).astype(np.float32),float(np.concatenate(delta).mean())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--variant',required=True,choices=VARIANTS); ap.add_argument('--run-tag',required=True); args=ap.parse_args()
    cfg=load_config(ROOT); seed=int(cfg['experiment']['seed']); seed_all(seed); art=artifact_dir(cfg,args.stage); ready=json.loads((art/'READY.json').read_text()); cache=cache_dir(cfg,args.stage)
    mean=float(ready['speed_mean']); std=float(ready['speed_std'])
    ntr=len(np.load(cache/'train_y_raw.npy',mmap_mode='r')); nva=len(np.load(cache/'val_y_raw.npy',mmap_mode='r')); tr_ids,dev_ids=ids_for(cfg,ntr,nva)
    trds=P1Dataset(cache,art,'train',args.variant,tr_ids,mean,std); dvds=P1Dataset(cache,art,'val',args.variant,dev_ids,mean,std)
    tc=cfg['training']; trl=loader(trds,tc['batch_size'],True,cfg); dvl=loader(dvds,tc['eval_batch_size'],False,cfg)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=P1ResidualAdapter(int(ready['latent_pca_dim']),6,int(cfg['experiment']['horizon']),int(tc['adapter_hidden']),float(tc['dropout'])).to(device)
    params=sum(p.numel() for p in model.parameters() if p.requires_grad); opt=torch.optim.AdamW(model.parameters(),lr=float(tc['learning_rate']),weight_decay=float(tc['weight_decay']))
    # Exact zero-start check: P1 must replay Frozen PatchSTG before learning.
    b0=move(next(iter(dvl)),device); model.eval()
    with torch.no_grad(): p0,d0=model(b0['base'],b0['latent'],b0['qhist'],b0['shist'],b0['sfuture'],b0['scov'])
    if not torch.equal(p0,b0['base']):
        md=float((p0-b0['base']).abs().max()); raise RuntimeError(f'zero-init no-op failed max_abs={md}')
    yva=np.asarray(np.load(cache/'val_y_raw.npy',mmap_mode='r')[dev_ids],np.float32); vv=np.asarray(np.load(cache/'val_future_valid.npy',mmap_mode='r')[dev_ids],bool); dd=np.asarray(np.load(cache/'val_direct_mask.npy',mmap_mode='r')[dev_ids],bool); ss=np.asarray(np.load(cache/'val_support_mask.npy',mmap_mode='r')[dev_ids],bool)
    base_norm=np.asarray(np.load(art/'val_base_pred_norm.npy',mmap_mode='r')[dev_ids],np.float32); base_raw=base_norm*std+mean; base_m=metric_dict_np(base_raw,yva,vv,dd,ss)
    rdir=variant_dir(cfg,args.stage,args.run_tag,args.variant); rdir.mkdir(parents=True,exist_ok=True); np.savez(rdir/'split_ids.npz',train=tr_ids,dev=dev_ids)
    log(f'P1 {args.variant} START params={params} device={device} train={len(trds)} dev={len(dvds)} base_masked={base_m["masked_mae"]:.6f}')
    best=None; bad=0; t0=time.time()
    for ep in range(1,int(tc['epochs'])+1):
        et=time.time(); tl,_,_=run(model,trl,device,cfg['loss'],opt); vl,pn,cabs=run(model,dvl,device,cfg['loss'],None); pr=pn*std+mean; m=metric_dict_np(pr,yva,vv,dd,ss); gain=improvement_pct(base_m['masked_mae'],m['masked_mae'])
        log(f'P1 {args.variant} EPOCH {ep:02d} train_loss={tl:.6f} dev_loss={vl:.6f} dev_masked={m["masked_mae"]:.6f} gain={gain:+.3f}% correction_abs_norm={cabs:.6f} sec={time.time()-et:.1f}')
        if best is None or m['masked_mae']<best['score']-1e-7:
            best={'score':m['masked_mae'],'epoch':ep,'metrics':m,'gain':gain,'corr':cabs,'state':copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})}; bad=0; torch.save({'variant':args.variant,'epoch':ep,'state_dict':best['state'],'config':cfg,'artifact_ready':ready},rdir/'best.pt')
        else:
            bad+=1
            if bad>=int(tc['patience']): log(f'P1 {args.variant} EARLY STOP epoch={ep} best={best["epoch"]}'); break
    summary={'version':cfg['version'],'variant':args.variant,'trainable_params':params,'best_epoch':best['epoch'],'baseline_dev':base_m,'model_dev':best['metrics'],'dev_gain_pct':best['gain'],'correction_abs_mean_norm':best['corr'],'wall_seconds':time.time()-t0,'retrieval_contract':{'memory':'train only','candidate_episode_end_before_query':True,'historical_future_masked':True},'final_evaluated':False}
    write_json(rdir/'summary.json',summary); log(f'P1 {args.variant} COMPLETE gain={best["gain"]:+.3f}% summary={rdir/"summary.json"}')

if __name__=='__main__': main()
