#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,json
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe'); ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml,write_json,seed_all,log
from presure.contract import cache_dir
from presure.data import CacheSplit,deterministic_split
from presure.paths import artifact_dir
from presure.future_teacher import compute_distance_for_ids
from presure.ranker import DualQueryRanker,soft_teacher,ranker_loss
from presure.ranking_metrics import ranking_metrics

class RankDS(Dataset):
    def __init__(self,qdesc,mdesc,ci,ids,dist): self.q=qdesc; self.m=mdesc; self.ci=ci; self.ids=np.asarray(ids,np.int64); self.d=np.asarray(dist,np.float32)
    def __len__(self): return len(self.ids)
    def __getitem__(self,j):
        qi=int(self.ids[j]); ci=np.asarray(self.ci[qi],np.int64); return {'q':torch.from_numpy(np.asarray(self.q[qi],np.float32).copy()),'c':torch.from_numpy(np.asarray(self.m[ci],np.float32).copy()),'d':torch.from_numpy(np.asarray(self.d[j],np.float32).copy())}

def eval_ranker(model,dl,device,temp,topk):
    model.eval(); pp=[]; dd=[]; losses=[]
    with torch.no_grad():
        for b in dl:
            q=b['q'].to(device,non_blocking=True); c=b['c'].to(device,non_blocking=True); d=b['d'].to(device,non_blocking=True); p=model(q,c); losses.append(float(ranker_loss(p,soft_teacher(d,temp)).item())); pp.append(p.cpu().numpy()); dd.append(d.cpu().numpy())
    m=ranking_metrics(np.concatenate(pp),np.concatenate(dd),topk); m['loss']=float(np.mean(losses)); return m

def infer_topk(model,qdesc,mdesc,cand,topk,batch,device):
    model.eval(); oi=np.empty((len(qdesc),topk),np.int32); os=np.empty((len(qdesc),topk),np.float32)
    with torch.no_grad():
        for a in range(0,len(qdesc),int(batch)):
            z=slice(a,min(len(qdesc),a+int(batch))); ci=np.asarray(cand[z],np.int64); q=torch.from_numpy(np.asarray(qdesc[z],np.float32).copy()).to(device,non_blocking=True); c=torch.from_numpy(np.asarray(mdesc[ci],np.float32).copy()).to(device,non_blocking=True); s=model(q,c); v,ix=torch.topk(s,k=topk,dim=1); sel=torch.gather(torch.from_numpy(ci).to(device),1,ix); oi[z]=sel.cpu().numpy().astype(np.int32); os[z]=v.cpu().numpy().astype(np.float32)
    return oi,os

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--force',action='store_true'); a=ap.parse_args(); cfg=load_yaml(ROOT/'configs/presure_v12.yaml'); seed_all(cfg['experiment']['seed']); out=artifact_dir(cfg,a.stage); ready=json.loads((out/'READY.json').read_text())
    reuse=[out/'dual_query_retriever_best.pt',out/'train_topk8.npz',out/'val_topk8.npz',out/'retriever_summary.json']
    if not a.force and all(x.exists() for x in reuse):
        try:
            ck=torch.load(out/'dual_query_retriever_best.pt',map_location='cpu',weights_only=False)
            if str(ck.get('version')) in {str(cfg['version']),'1.2.0'}:
                log(f'DUAL QUERY RETRIEVER already ready: {out} retriever_version={ck.get("version")} compatible_with={cfg["version"]}')
                return
        except Exception as e:
            log(f'WARN retriever reuse check failed; retraining: {e}')
    cache=cache_dir(cfg,a.stage); tr=CacheSplit(cache,'train'); va=CacheSplit(cache,'val'); qtr=np.load(out/'train_desc_dual.npy',mmap_mode='r'); qva=np.load(out/'val_desc_dual.npy',mmap_mode='r'); tc=np.load(out/'train_candidates.npz'); vc=np.load(out/'val_candidates.npz'); ti=tc['indices']; vi=vc['indices']; rc=cfg['retriever']; seed=int(cfg['experiment']['seed']); rng=np.random.default_rng(seed); perm=rng.permutation(len(tr)); nt=min(int(rc['train_queries']),len(tr)); nh=min(int(rc['train_holdout_queries']),max(len(tr)-nt,0)); train_ids=perm[:nt]; hold_ids=perm[nt:nt+nh]; dev_ids,final_ids=deterministic_split(len(va),int(rc['dev_queries']),int(rc['final_val_queries']),seed); np.savez(out/'retriever_split_ids.npz',train=train_ids,holdout=hold_ids,dev=dev_ids,final=final_ids)
    std=float(ready['speed_std']); td=compute_distance_for_ids(tr,tr,ti,train_ids,std,cfg,64); hd=compute_distance_for_ids(tr,tr,ti,hold_ids,std,cfg,64); dd=compute_distance_for_ids(va,tr,vi,dev_ids,std,cfg,64)
    rt=cfg['runtime']; nw=int(rt['num_workers']); common={'num_workers':nw,'pin_memory':bool(rt['pin_memory']) and torch.cuda.is_available()};
    if nw>0: common.update(prefetch_factor=int(rt['prefetch_factor']),persistent_workers=bool(rt['persistent_workers']))
    train_dl=DataLoader(RankDS(qtr,qtr,ti,train_ids,td),batch_size=int(rc['batch_size']),shuffle=True,**common); hold_dl=DataLoader(RankDS(qtr,qtr,ti,hold_ids,hd),batch_size=int(rc['eval_batch_size']),shuffle=False,**common); dev_dl=DataLoader(RankDS(qva,qtr,vi,dev_ids,dd),batch_size=int(rc['eval_batch_size']),shuffle=False,**common)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); model=DualQueryRanker(qtr.shape[1],int(rc['d_model']),int(rc['hidden']),float(rc['dropout'])).to(device); opt=torch.optim.AdamW(model.parameters(),lr=float(rc['lr']),weight_decay=float(rc['weight_decay'])); best=None; bad=0; topk=int(cfg['candidate_pool']['top_k']); temp=float(cfg['future_teacher']['temperature'])
    for ep in range(1,int(rc['epochs'])+1):
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        model.train(); ls=[]
        for b in train_dl:
            q=b['q'].to(device,non_blocking=True); c=b['c'].to(device,non_blocking=True); d=b['d'].to(device,non_blocking=True); p=model(q,c); loss=ranker_loss(p,soft_teacher(d,temp)); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(rc['grad_clip'])); opt.step(); ls.append(float(loss.item()))
        hm=eval_ranker(model,hold_dl,device,temp,topk); mem=''
        if device.type=='cuda': mem=f' cuda_peak_alloc={torch.cuda.max_memory_allocated(device)/1024**3:.2f}GB cuda_peak_reserved={torch.cuda.max_memory_reserved(device)/1024**3:.2f}GB'
        log(f'DUAL RETRIEVER EPOCH {ep:02d} train={np.mean(ls):.5f} hold_R@{topk}={hm[f"recall_at_{topk}"]:.4f} hold_NDCG={hm[f"ndcg_at_{topk}"]:.4f} hold_regret={hm[f"normalized_future_regret_at_{topk}"]:.4f}{mem}')
        score=hm[f'ndcg_at_{topk}']-0.25*hm[f'normalized_future_regret_at_{topk}']
        if best is None or score>best['score']+1e-6: best={'score':score,'epoch':ep,'hold':hm,'state':{k:v.detach().cpu() for k,v in model.state_dict().items()}}; bad=0
        else:
            bad+=1
            if bad>=int(rc['patience']): break
    model.load_state_dict(best['state']); dm=eval_ranker(model,dev_dl,device,temp,topk); torch.save({'version':cfg['version'],'epoch':best['epoch'],'state_dict':best['state'],'input_dim':int(qtr.shape[1]),'holdout':best['hold'],'dev':dm},out/'dual_query_retriever_best.pt')
    tri,trs=infer_topk(model,qtr,qtr,ti,topk,int(rc['eval_batch_size']),device); vai,vas=infer_topk(model,qva,qtr,vi,topk,int(rc['eval_batch_size']),device); np.savez_compressed(out/'train_topk8.npz',indices=tri,scores=trs); np.savez_compressed(out/'val_topk8.npz',indices=vai,scores=vas)
    write_json(out/'retriever_summary.json',{'best_epoch':best['epoch'],'holdout':best['hold'],'dev':dm,'top_k':topk,'teacher':'future-evolution compatibility on Train/Dev labels only; future is never deployable input','final_validation_retrieval_metrics_locked':True}); log(f'DUAL RETRIEVER FROZEN epoch={best["epoch"]} DEV R@{topk}={dm[f"recall_at_{topk}"]:.4f} NDCG={dm[f"ndcg_at_{topk}"]:.4f} regret={dm[f"normalized_future_regret_at_{topk}"]:.4f}'); log('TRAIN DUAL QUERY RETRIEVER PASS')
if __name__=='__main__': main()
