#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse, csv, json, sys
import numpy as np
import torch

ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe')
ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from presure.utils import load_yaml, write_json, stamp, seed_all
from presure.contract import cache_dir
from presure.data import CacheSplit, cache_stats
from presure.features import NODE_SENSE_DIM
from presure.pipeline import art_dir, run_dir
from presure.query_error_direction import (
    QueryFutureErrorDirectionNet, derive_abs_error_thresholds, threshold_array,
    query_direction_loss, evaluate_direction_probe,
)


def t(a,dev,dtype=torch.float32): return torch.as_tensor(np.array(a,copy=True),device=dev,dtype=dtype)
def batches(ids,bs,shuffle=False,seed=0):
    x=np.asarray(ids,np.int64).copy()
    if shuffle: np.random.default_rng(seed).shuffle(x)
    for i in range(0,len(x),int(bs)): yield x[i:i+int(bs)]


def load_v09a_splits(cfg,stage):
    base=Path(cfg['project']['exp_root'])/'PredictiveUtilityRetrieval'/cfg['experiment']['subgraph']/stage
    j=json.loads((base/'LATEST.json').read_text()); rd=Path(j['run_dir'])
    return rd,{k:np.load(rd/f'{k}.npy') for k in ['train_ids','dev_ids','final_val_ids']}


def raw_error(ds,base,ids): return np.asarray(ds.y[ids],np.float32)-np.asarray(base[ids],np.float32)


def train_class_pos_weight(tr,base,ids,thresholds):
    e=raw_error(tr,base,ids); v=np.asarray(tr.valid[ids],bool)
    use=v & (np.abs(e)>=threshold_array(thresholds,'q50'))
    pos=int(np.sum((e>0)&use)); neg=int(np.sum((e<=0)&use))
    return float(neg/max(pos,1)), {'positive':pos,'negative':neg,'eligible':int(use.sum())}


def predict(model,node_features,ids,batch_size,dev):
    pe=[]; pp=[]; model.eval()
    with torch.no_grad():
        for bi in batches(ids,batch_size):
            err,logit=model(t(node_features[bi],dev)); pe.append(err.cpu().numpy().astype(np.float32)); pp.append(torch.sigmoid(logit).cpu().numpy().astype(np.float32))
    return np.concatenate(pe,0),np.concatenate(pp,0)


def calibrate_direction_threshold(prob, error_raw, ds, ids, thresholds, grid):
    from presure.need_state import binary_precision_recall_f1
    P=error_raw.shape[1]
    valid=np.asarray(ds.valid[ids],bool)
    masked=valid & np.asarray(ds.direct[ids],bool)[:,None,:].repeat(P,axis=1)
    use=masked & (np.abs(error_raw)>=threshold_array(thresholds,'q75'))
    if not np.any(use): return 0.5, []
    rows=[]; best=None
    for th in grid:
        m=binary_precision_recall_f1(prob[use],error_raw[use]>0,float(th))
        row={'threshold':float(th),**m}; rows.append(row)
        key=(float(m['balanced_accuracy']),float(m['f1']),float(m['accuracy']),-abs(float(th)-0.5))
        if best is None or key>best[0]: best=(key,float(th))
    return best[1],rows

def dev_objective(metrics):
    m=metrics['views']['masked']
    q75=m['q75']; q90=m['q90']; reg=m['regression']
    vals=[q75.get('auroc') or 0.0,q75.get('balanced_accuracy') or 0.0,q90.get('auroc') or 0.0,q90.get('balanced_accuracy') or 0.0]
    return float(np.mean(vals)+0.10*max(float(reg.get('spearman') or 0.0),0.0))


def decision(metrics,cfg):
    d=cfg['query_future_error_direction_probe']['decision']; m=metrics['views']['masked']; q75=m['q75']; q90=m['q90']; reg=m['regression']
    go={
        'q75_auroc':float(q75.get('auroc') or 0)>=float(d['masked_q75_auroc_go']),
        'q75_balacc':float(q75.get('balanced_accuracy') or 0)>=float(d['masked_q75_balanced_accuracy_go']),
        'q90_auroc':float(q90.get('auroc') or 0)>=float(d['masked_q90_auroc_go']),
        'q90_balacc':float(q90.get('balanced_accuracy') or 0)>=float(d['masked_q90_balanced_accuracy_go']),
        'reg_spearman':float(reg.get('spearman') or 0)>=float(d['masked_regression_spearman_go']),
    }
    if all(go.values()): return 'GO_RESIDUAL_ALIGNMENT',go
    cond=(float(q75.get('auroc') or 0)>=float(d['masked_q75_auroc_conditional']) and
          float(q75.get('balanced_accuracy') or 0)>=float(d['masked_q75_balanced_accuracy_conditional']) and
          float(q90.get('auroc') or 0)>=float(d['masked_q90_auroc_conditional']) and
          float(q90.get('balanced_accuracy') or 0)>=float(d['masked_q90_balanced_accuracy_conditional']))
    return ('CONDITIONAL_ALIGNMENT' if cond else 'NO_GO_QUERY_REPRESENTATION'),go


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development']); ap.add_argument('--seed',type=int,default=None); a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/presure.yaml')
    if str(cfg.get('version')) not in {'0.9.4','0.9.5'}: raise RuntimeError(f'Query direction probe requires v0.9.4+, got {cfg.get("version")}')
    seed=int(cfg['experiment']['seed'] if a.seed is None else a.seed); seed_all(seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    art=art_dir(cfg,a.stage); cache=cache_dir(cfg,a.stage); tr=CacheSplit(cache,'train'); va=CacheSplit(cache,'val'); _,speed_std=cache_stats(cache)
    nft=np.load(art/'train_node_features.npy',mmap_mode='r'); nfv=np.load(art/'val_node_features.npy',mmap_mode='r')
    bt=np.load(art/'train_base_pred.npy',mmap_mode='r'); bv=np.load(art/'val_base_pred.npy',mmap_mode='r')
    source_run,splits=load_v09a_splits(cfg,a.stage); train_ids=splits['train_ids']; dev_ids=splits['dev_ids']; final_ids=splits['final_val_ids']
    pc=cfg['query_future_error_direction_probe']; mc=pc['model']; tc=pc['training']; H=int(cfg['experiment']['history']); P=int(cfg['experiment']['horizon'])
    train_err=raw_error(tr,bt,train_ids); train_valid=np.asarray(tr.valid[train_ids],bool)
    thresholds=derive_abs_error_thresholds(train_err,train_valid,tuple(float(x) for x in pc['thresholds']['evaluation_quantiles']))
    pos_weight,counts=train_class_pos_weight(tr,bt,train_ids,thresholds)
    rd=run_dir(cfg,'QueryFutureErrorDirectionProbe',a.stage,stamp(),seed); rd.mkdir(parents=True,exist_ok=True)
    for k,v in splits.items(): np.save(rd/f'{k}.npy',v)
    write_json(rd/'train_error_thresholds.json',{**thresholds,'source':'v0.9A train_ids only; horizon-specific |Y-PatchSTG| quantiles','class_counts_q50':counts,'positive_definition':'Y_future > PatchSTG forecast (positive correction / underprediction)'})
    args=dict(din=nft.shape[-1],num_nodes=nft.shape[1],P=P,base_start=4*H,sense_dim=NODE_SENSE_DIM,
              encoder_hidden=int(mc['encoder_hidden']),state_dim=int(mc['state_dim']),node_dim=int(mc['node_embedding_dim']),
              horizon_dim=int(mc['horizon_embedding_dim']),hidden=int(mc['hidden']),dropout=float(mc['dropout']))
    model=QueryFutureErrorDirectionNet(**args).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=float(tc['lr']),weight_decay=float(tc['weight_decay']))
    pw=torch.tensor(pos_weight,device=dev,dtype=torch.float32)
    q50=t(np.asarray(thresholds['q50'],np.float32)[None,:,None],dev)
    best=-1e18; best_ep=0; wait=0; bestp=rd/'query_future_error_direction_best.pt'
    print('='*68,flush=True); print('PreSuRe v0.9.4 Query Future-Error Direction Probe',flush=True)
    print(f'v0.9A split source={source_run}',flush=True); print(f'train={len(train_ids)} dev={len(dev_ids)} final_val={len(final_ids)}',flush=True)
    print('INPUT=query only: incomplete sensing context + frozen PatchSTG forecast; no retrieval/support candidates',flush=True)
    print('TARGET=E*=Y_future-PatchSTG; direction positive means PatchSTG underpredicts / positive correction needed',flush=True)
    print(f'Train direction supervision uses |E*| >= horizon-specific q50; pos_weight={pos_weight:.4f}',flush=True); print('='*68,flush=True)
    for ep in range(1,int(tc['epochs'])+1):
        model.train(); losses=[]; regs=[]; dirs=[]
        for bi in batches(train_ids,int(tc['batch_size']),True,seed+1009*ep):
            x=t(nft[bi],dev); er=t(raw_error(tr,bt,bi),dev); vv=t(np.asarray(tr.valid[bi],bool),dev,torch.bool)
            em,dl=model(x); loss,parts=query_direction_loss(em,dl,er,vv,speed_std,q50,pw,float(tc['huber_beta']),float(tc['regression_weight']),float(tc['direction_weight']),float(tc['magnitude_weight']),float(tc['target_weight_clip']))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(tc['grad_clip'])); opt.step()
            losses.append(float(loss.detach().cpu())); regs.append(parts['regression']); dirs.append(parts['direction'])
        pe,pp=predict(model,nft,dev_ids,int(tc['eval_batch_size']),dev); de=raw_error(tr,bt,dev_ids)
        met=evaluate_direction_probe(pe,pp,de,tr,dev_ids,thresholds,speed_std,int(tc['metric_sample_points']),seed+ep)
        sc=dev_objective(met); mq=met['views']['masked']
        print(f'EPOCH {ep:02d} loss={np.mean(losses):.5f} reg={np.mean(regs):.5f} dir={np.mean(dirs):.5f} '
              f'masked_q75_auc={mq["q75"].get("auroc")} bal={mq["q75"].get("balanced_accuracy")} '
              f'masked_q90_auc={mq["q90"].get("auroc")} bal={mq["q90"].get("balanced_accuracy")} '
              f'spearman={mq["regression"].get("spearman")}',flush=True)
        if sc>best+1e-6:
            best=sc; best_ep=ep; wait=0; torch.save({'epoch':ep,'state_dict':model.state_dict(),'model_args':args,'dev_objective':sc,'train_thresholds':thresholds,'speed_std':speed_std,'pos_weight':pos_weight,'split_source_run':str(source_run)},bestp)
        else:
            wait+=1
            if wait>=int(tc['patience']): print(f'EARLY STOP epoch={ep} best_epoch={best_ep}',flush=True); break
    ck=torch.load(bestp,map_location=dev,weights_only=False); model.load_state_dict(ck['state_dict'])
    # Calibrate only the binary decision threshold on Train-dev masked q75; AUROC is threshold-free.
    dpe,dpp=predict(model,nft,dev_ids,int(tc['eval_batch_size']),dev); derr=raw_error(tr,bt,dev_ids)
    dir_th,dir_grid=calibrate_direction_threshold(dpp,derr,tr,dev_ids,thresholds,[float(x) for x in tc['direction_threshold_grid']])
    dev_final=evaluate_direction_probe(dpe,dpp,derr,tr,dev_ids,thresholds,speed_std,int(tc['metric_sample_points']),seed+8000,direction_threshold=dir_th)
    pe,pp=predict(model,nfv,final_ids,int(tc['eval_batch_size']),dev); fe=raw_error(va,bv,final_ids)
    metrics=evaluate_direction_probe(pe,pp,fe,va,final_ids,thresholds,speed_std,int(tc['metric_sample_points']),seed+9000,direction_threshold=dir_th)
    status,checks=decision(metrics,cfg)
    summary={'version':cfg['version'],'implementation_revision':cfg.get('implementation_revision'),'model_type':'query_future_error_direction_probe_v094','stage':a.stage,'seed':seed,'run_dir':str(rd),'checkpoint':str(bestp),'best_epoch':int(ck['epoch']),'split_source_run':str(source_run),'splits_reused_from_v09a':True,'train_thresholds':thresholds,'train_q50_class_counts':counts,'direction_threshold_calibrated_on_train_dev_masked_q75':dir_th,'direction_threshold_grid':dir_grid,'query_only_inputs':True,'candidate_inputs_used':False,'qwen_used':False,'dual_memory_used':False,'validation_future_input_used':False,'train_dev_metrics_at_calibrated_threshold':dev_final,'final_val':metrics,'decision_checks':checks,'decision':status}
    write_json(rd/'summary.json',summary); (rd/'DECISION.txt').write_text(status+'\n'+json.dumps(checks,indent=2,sort_keys=True)+'\n')
    rows=[]
    for view,vv in metrics['views'].items():
        for task,mm in vv.items(): rows.append({'view':view,'slice':task,**mm})
    with (rd/'direction_metrics.csv').open('w',newline='') as f:
        keys=sorted(set().union(*(r.keys() for r in rows))); w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    with (rd/'masked_horizon_metrics.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['horizon','slice','count','auroc','balanced_accuracy','accuracy','prevalence'])
        for h,d in metrics['masked_by_horizon'].items():
            for sl,m in d.items(): w.writerow([h,sl,m['count'],m['auroc'],m['balanced_accuracy'],m['accuracy'],m['prevalence']])
    latest=Path(cfg['project']['exp_root'])/'QueryFutureErrorDirectionProbe'/cfg['experiment']['subgraph']/a.stage/'LATEST.json'; latest.parent.mkdir(parents=True,exist_ok=True); write_json(latest,{'run_dir':str(rd),'checkpoint':str(bestp),'summary':str(rd/'summary.json'),'decision':status,'version':cfg['version']})
    print('='*68,flush=True); print(f'BEST epoch={ck["epoch"]} direction_threshold={dir_th:g}',flush=True); print('FINAL masked:',json.dumps(metrics['views']['masked'],indent=2),flush=True); print(f'DECISION = {status} checks={checks}',flush=True); print(f'SUMMARY {rd/"summary.json"}',flush=True); print('='*68,flush=True)

if __name__=='__main__': main()
