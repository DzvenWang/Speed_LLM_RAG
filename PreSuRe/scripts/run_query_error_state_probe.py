#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse,csv,json,sys
import numpy as np, torch

ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe'); ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml,write_json,stamp,seed_all
from presure.contract import cache_dir
from presure.data import CacheSplit,cache_stats
from presure.features import NODE_SENSE_DIM
from presure.pipeline import art_dir,run_dir
from presure.query_error_direction import derive_abs_error_thresholds,threshold_array,query_direction_loss,evaluate_direction_probe
from presure.query_error_state import QueryErrorStateNet,RECENT_ERROR_DIM,PHYSICAL_TRANSITION_DIM


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
    e=raw_error(tr,base,ids); v=np.asarray(tr.valid[ids],bool); use=v&(np.abs(e)>=threshold_array(thresholds,'q50'))
    pos=int(np.sum((e>0)&use)); neg=int(np.sum((e<=0)&use)); return float(neg/max(pos,1)),{'positive':pos,'negative':neg,'eligible':int(use.sum())}

def calibrate(prob,error_raw,ds,ids,thresholds,grid):
    from presure.need_state import binary_precision_recall_f1
    P=error_raw.shape[1]; valid=np.asarray(ds.valid[ids],bool); masked=valid&np.asarray(ds.direct[ids],bool)[:,None,:].repeat(P,1); use=masked&(np.abs(error_raw)>=threshold_array(thresholds,'q75'))
    rows=[]; best=None
    for th in grid:
        m=binary_precision_recall_f1(prob[use],error_raw[use]>0,float(th)); row={'threshold':float(th),**m}; rows.append(row)
        key=(float(m['balanced_accuracy']),float(m['f1']),float(m['accuracy']),-abs(float(th)-.5))
        if best is None or key>best[0]: best=(key,float(th))
    return (best[1] if best else .5),rows

def dev_score(metrics):
    m=metrics['views']['masked']; q75=m['q75']; q90=m['q90']; r=m['regression']
    vals=[q75.get('auroc') or 0,q75.get('balanced_accuracy') or 0,q90.get('auroc') or 0,q90.get('balanced_accuracy') or 0]
    return float(np.mean(vals)+.10*max(float(r.get('spearman') or 0),0))

def predict(model,nf,hid,rec,phy,ids,bs,dev):
    ee=[];pp=[];model.eval()
    with torch.no_grad():
        for bi in batches(ids,bs):
            e,l=model(t(nf[bi],dev),t(hid[bi],dev),t(rec[bi],dev),t(phy[bi],dev)); ee.append(e.cpu().numpy().astype(np.float32)); pp.append(torch.sigmoid(l).cpu().numpy().astype(np.float32))
    return np.concatenate(ee),np.concatenate(pp)

def decision(m,cfg):
    d=cfg['query_error_state_probe']['decision']; z=m['views']['masked']; q75=z['q75'];q90=z['q90'];r=z['regression']
    go={'q75_auroc':float(q75.get('auroc') or 0)>=float(d['masked_q75_auroc_go']),'q75_balacc':float(q75.get('balanced_accuracy') or 0)>=float(d['masked_q75_balanced_accuracy_go']),'q90_auroc':float(q90.get('auroc') or 0)>=float(d['masked_q90_auroc_go']),'q90_balacc':float(q90.get('balanced_accuracy') or 0)>=float(d['masked_q90_balanced_accuracy_go']),'reg_spearman':float(r.get('spearman') or 0)>=float(d['masked_regression_spearman_go'])}
    if all(go.values()): return 'GO_RESIDUAL_ALIGNMENT',go
    cond=float(q75.get('auroc') or 0)>=float(d['masked_q75_auroc_conditional']) and float(q90.get('auroc') or 0)>=float(d['masked_q90_auroc_conditional'])
    return ('CONDITIONAL_ALIGNMENT' if cond else 'NO_GO_QUERY_REPRESENTATION'),go

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development',choices=['development']);ap.add_argument('--seed',type=int,default=None);a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/presure.yaml');
    if str(cfg.get('version')) not in {'0.9.5','0.9.5.1'}: raise RuntimeError(f'expected v0.9.5 got {cfg.get("version")}')
    seed=int(cfg['experiment']['seed'] if a.seed is None else a.seed);seed_all(seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    art=art_dir(cfg,a.stage);cache=cache_dir(cfg,a.stage);tr=CacheSplit(cache,'train');va=CacheSplit(cache,'val');_,speed_std=cache_stats(cache)
    nftr=np.load(art/'train_node_features.npy',mmap_mode='r');nfva=np.load(art/'val_node_features.npy',mmap_mode='r');bt=np.load(art/'train_base_pred.npy',mmap_mode='r');bv=np.load(art/'val_base_pred.npy',mmap_mode='r')
    htr=np.load(art/'train_patchstg_hidden.npy',mmap_mode='r');hva=np.load(art/'val_patchstg_hidden.npy',mmap_mode='r');rtr=np.load(art/'train_recent_error_state.npy',mmap_mode='r');rva=np.load(art/'val_recent_error_state.npy',mmap_mode='r');ptr=np.load(art/'train_physical_transition.npy',mmap_mode='r');pva=np.load(art/'val_physical_transition.npy',mmap_mode='r')
    am=json.loads((art/'query_error_state_artifact_manifest.json').read_text())
    source,splits=load_v09a_splits(cfg,a.stage); train_ids=splits['train_ids'];dev_ids=splits['dev_ids'];final_ids=splits['final_val_ids']
    pc=cfg['query_error_state_probe'];mc=pc['model'];tc=pc['training'];H=int(cfg['experiment']['history']);P=int(cfg['experiment']['horizon'])
    thresholds=derive_abs_error_thresholds(raw_error(tr,bt,train_ids),np.asarray(tr.valid[train_ids],bool),tuple(float(x) for x in pc['thresholds']['evaluation_quantiles']));pw,counts=train_class_pos_weight(tr,bt,train_ids,thresholds);q50=t(np.asarray(thresholds['q50'],np.float32)[None,:,None],dev);pwt=torch.tensor(pw,device=dev)
    rd=run_dir(cfg,'QueryErrorStateProbe',a.stage,stamp(),seed);rd.mkdir(parents=True,exist_ok=True)
    for k,v in splits.items():np.save(rd/f'{k}.npy',v)
    write_json(rd/'train_error_thresholds.json',{**thresholds,'class_counts_q50':counts,'source':'exact v0.9A train_ids only'})
    variants=pc['variants'];results={};best_choice=None
    print('='*72,flush=True);print('PreSuRe v0.9.5 Query Error-State Representation Probe',flush=True);print(f'v0.9A split source={source}',flush=True);print(f'hidden={am["hidden"]["module_name"]} projected_dim={am["hidden"]["projected_dim"]}',flush=True);print(f'recent_error_train={am["recent_error_train"]}',flush=True);print('No retrieval/support candidate is an input.',flush=True);print('='*72,flush=True)
    recent_available=bool(am.get('recent_error_train',{}).get('available')) and bool(am.get('recent_error_val',{}).get('available'))
    for vi,(name,vc) in enumerate(variants.items()):
        effective_recent=bool(vc['use_recent']) and recent_available
        args=dict(din=nftr.shape[-1],num_nodes=nftr.shape[1],P=P,base_start=4*H,hidden_dim=htr.shape[-1],recent_dim=RECENT_ERROR_DIM,physical_dim=PHYSICAL_TRANSITION_DIM,use_hidden=bool(vc['use_hidden']),use_recent=effective_recent,use_physical=bool(vc['use_physical']),sense_dim=NODE_SENSE_DIM,encoder_hidden=int(mc['encoder_hidden']),state_dim=int(mc['state_dim']),node_dim=int(mc['node_embedding_dim']),horizon_dim=int(mc['horizon_embedding_dim']),extra_dim=int(mc['extra_dim']),hidden=int(mc['hidden']),dropout=float(mc['dropout']))
        model=QueryErrorStateNet(**args).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=float(tc['lr']),weight_decay=float(tc['weight_decay']));best=-1e18;wait=0;bestp=rd/f'{name}_best.pt'
        for ep in range(1,int(tc['epochs'])+1):
            model.train();losses=[]
            for bi in batches(train_ids,int(tc['batch_size']),True,seed+1009*ep+53*vi):
                em,dl=model(t(nftr[bi],dev),t(htr[bi],dev),t(rtr[bi],dev),t(ptr[bi],dev));er=t(raw_error(tr,bt,bi),dev);vv=t(np.asarray(tr.valid[bi],bool),dev,torch.bool)
                loss,_=query_direction_loss(em,dl,er,vv,speed_std,q50,pwt,float(tc['huber_beta']),float(tc['regression_weight']),float(tc['direction_weight']),float(tc['magnitude_weight']),float(tc['target_weight_clip']))
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(tc['grad_clip']));opt.step();losses.append(float(loss.detach().cpu()))
            pe,pp=predict(model,nftr,htr,rtr,ptr,dev_ids,int(tc['eval_batch_size']),dev);de=raw_error(tr,bt,dev_ids);met=evaluate_direction_probe(pe,pp,de,tr,dev_ids,thresholds,speed_std,int(tc['metric_sample_points']),seed+vi*101+ep);sc=dev_score(met);mq=met['views']['masked']
            print(f'VARIANT {name} EPOCH {ep:02d} loss={np.mean(losses):.5f} q75_auc={mq["q75"].get("auroc")} q75_bal={mq["q75"].get("balanced_accuracy")} q90_auc={mq["q90"].get("auroc")} q90_bal={mq["q90"].get("balanced_accuracy")} spearman={mq["regression"].get("spearman")}',flush=True)
            if sc>best+1e-6: best=sc;wait=0;torch.save({'epoch':ep,'state_dict':model.state_dict(),'model_args':args,'dev_objective':sc},bestp)
            else:
                wait+=1
                if wait>=int(tc['patience']):break
        ck=torch.load(bestp,map_location=dev,weights_only=False);model.load_state_dict(ck['state_dict']);dpe,dpp=predict(model,nftr,htr,rtr,ptr,dev_ids,int(tc['eval_batch_size']),dev);derr=raw_error(tr,bt,dev_ids);th,grid=calibrate(dpp,derr,tr,dev_ids,thresholds,[float(x) for x in tc['direction_threshold_grid']]);devmet=evaluate_direction_probe(dpe,dpp,derr,tr,dev_ids,thresholds,speed_std,int(tc['metric_sample_points']),seed+7000+vi,direction_threshold=th)
        fpe,fpp=predict(model,nfva,hva,rva,pva,final_ids,int(tc['eval_batch_size']),dev);ferr=raw_error(va,bv,final_ids);fm=evaluate_direction_probe(fpe,fpp,ferr,va,final_ids,thresholds,speed_std,int(tc['metric_sample_points']),seed+9000+vi,direction_threshold=th)
        results[name]={'config':vc,'effective_use_recent':effective_recent,'checkpoint':str(bestp),'best_epoch':int(ck['epoch']),'dev_objective':float(ck['dev_objective']),'direction_threshold':th,'direction_threshold_grid':grid,'dev':devmet,'final_val':fm}
        key=(float(ck['dev_objective']),name=='full')
        if best_choice is None or key>best_choice[0]:best_choice=(key,name)
    selected=best_choice[1];status,checks=decision(results[selected]['final_val'],cfg)
    # v0.9.4 final result is read for reporting only; never used to select v0.9.5 variant.
    old=None;lp=Path(cfg['project']['exp_root'])/'QueryFutureErrorDirectionProbe'/cfg['experiment']['subgraph']/a.stage/'LATEST.json'
    if lp.exists():
        try: old=json.loads(Path(json.loads(lp.read_text())['summary']).read_text()).get('final_val')
        except Exception: old=None
    summary={'version':'0.9.5','implementation_revision':cfg.get('implementation_revision'),'run_dir':str(rd),'split_source_run':str(source),'artifact_manifest':am,'variant_selection':'Train-dev objective only; Final Validation never selects variant','selected_variant':selected,'variants':results,'v094_reference_final_val_reporting_only':old,'decision':status,'decision_checks':checks}
    write_json(rd/'summary.json',summary);(rd/'DECISION.txt').write_text(status+'\n'+json.dumps(checks,indent=2,sort_keys=True)+'\n')
    with (rd/'variant_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['variant','dev_objective','q75_auroc','q75_balacc','q90_auroc','q90_balacc','reg_spearman','threshold'])
        for n,r in results.items():
            m=r['final_val']['views']['masked'];w.writerow([n,r['dev_objective'],m['q75'].get('auroc'),m['q75'].get('balanced_accuracy'),m['q90'].get('auroc'),m['q90'].get('balanced_accuracy'),m['regression'].get('spearman'),r['direction_threshold']])
    latest=Path(cfg['project']['exp_root'])/'QueryErrorStateProbe'/cfg['experiment']['subgraph']/a.stage/'LATEST.json';latest.parent.mkdir(parents=True,exist_ok=True);write_json(latest,{'run_dir':str(rd),'summary':str(rd/'summary.json'),'selected_variant':selected,'decision':status})
    print('='*72,flush=True);print(f'SELECTED variant={selected} by Train-dev only',flush=True);print('FINAL masked:',json.dumps(results[selected]['final_val']['views']['masked'],indent=2),flush=True);print(f'DECISION={status} checks={checks}',flush=True);print(f'SUMMARY {rd/"summary.json"}',flush=True);print('='*72,flush=True)

if __name__=='__main__':main()
