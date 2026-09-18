#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, csv, json, math, re, sys, gc
import numpy as np, torch

ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe')
ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from presure.utils import load_yaml, write_json, stamp, seed_all
from presure.contract import cache_dir
from presure.data import CacheSplit, cache_stats
from presure.features import NODE_SENSE_DIM
from presure.pipeline import art_dir, run_dir
from presure.query_error_direction import derive_abs_error_thresholds, threshold_array, query_direction_loss, evaluate_direction_probe
from presure.query_error_state import QueryErrorStateNet, RECENT_ERROR_REPAIRED_DIM, PHYSICAL_TRANSITION_DIM
from presure.backbone import load_patchstg_adapter
from presure.backbone_hidden import HiddenLayerChoice, infer_hidden_ids


def t(a,dev,dtype=torch.float32): return torch.as_tensor(np.array(a,copy=True),device=dev,dtype=dtype)
def batches(n,bs,shuffle=False,seed=0):
    x=np.arange(int(n),dtype=np.int64)
    if shuffle: np.random.default_rng(seed).shuffle(x)
    for i in range(0,len(x),int(bs)): yield x[i:i+int(bs)]

def load_v09a_splits(cfg,stage):
    base=Path(cfg['project']['exp_root'])/'PredictiveUtilityRetrieval'/cfg['experiment']['subgraph']/stage
    j=json.loads((base/'LATEST.json').read_text());rd=Path(j['run_dir'])
    return rd,{k:np.load(rd/f'{k}.npy') for k in ['train_ids','dev_ids','final_val_ids']}

def raw_error(ds,base,ids): return np.asarray(ds.y[ids],np.float32)-np.asarray(base[ids],np.float32)

def train_class_pos_weight(tr,base,ids,thresholds):
    e=raw_error(tr,base,ids);v=np.asarray(tr.valid[ids],bool);use=v&(np.abs(e)>=threshold_array(thresholds,'q50'))
    pos=int(np.sum((e>0)&use));neg=int(np.sum((e<=0)&use));return float(neg/max(pos,1)),{'positive':pos,'negative':neg,'eligible':int(use.sum())}

def calibrate(prob,error_raw,ds,ids,thresholds,grid):
    from presure.need_state import binary_precision_recall_f1
    P=error_raw.shape[1];valid=np.asarray(ds.valid[ids],bool);masked=valid&np.asarray(ds.direct[ids],bool)[:,None,:].repeat(P,1);use=masked&(np.abs(error_raw)>=threshold_array(thresholds,'q75'))
    rows=[];best=None
    for th in grid:
        m=binary_precision_recall_f1(prob[use],error_raw[use]>0,float(th));row={'threshold':float(th),**m};rows.append(row)
        key=(float(m['balanced_accuracy']),float(m['f1']),float(m['accuracy']),-abs(float(th)-.5))
        if best is None or key>best[0]:best=(key,float(th))
    return (best[1] if best else .5),rows

def dev_score(metrics):
    m=metrics['views']['masked'];q75=m['q75'];q90=m['q90'];r=m['regression']
    vals=[q75.get('auroc') or 0,q75.get('balanced_accuracy') or 0,q90.get('auroc') or 0,q90.get('balanced_accuracy') or 0]
    return float(np.mean(vals)+.10*max(float(r.get('spearman') or 0),0))

def decision(m,cfg):
    d=cfg['query_error_state_diagnostics']['decision'];z=m['views']['masked'];q75=z['q75'];q90=z['q90'];r=z['regression']
    go={'q75_auroc':float(q75.get('auroc') or 0)>=float(d['masked_q75_auroc_go']),
        'q75_balacc':float(q75.get('balanced_accuracy') or 0)>=float(d['masked_q75_balanced_accuracy_go']),
        'q90_auroc':float(q90.get('auroc') or 0)>=float(d['masked_q90_auroc_go']),
        'q90_balacc':float(q90.get('balanced_accuracy') or 0)>=float(d['masked_q90_balanced_accuracy_go']),
        'reg_spearman':float(r.get('spearman') or 0)>=float(d['masked_regression_spearman_go'])}
    if all(go.values()): return 'GO_RESIDUAL_ALIGNMENT',go
    cond=float(q75.get('auroc') or 0)>=float(d['masked_q75_auroc_conditional']) and float(q90.get('auroc') or 0)>=float(d['masked_q90_auroc_conditional'])
    return ('CONDITIONAL_ALIGNMENT' if cond else 'NO_GO_QUERY_REPRESENTATION'),go

def slug(s):
    return re.sub(r'[^A-Za-z0-9_.-]+','_',str(s)).strip('_')[:120]


def make_model_args(nf,N,P,H,hidden_dim,mc,use_hidden,use_recent,use_physical):
    return dict(din=int(nf.shape[-1]),num_nodes=int(N),P=int(P),base_start=4*int(H),hidden_dim=max(int(hidden_dim),1),
                recent_dim=RECENT_ERROR_REPAIRED_DIM,physical_dim=PHYSICAL_TRANSITION_DIM,
                use_hidden=bool(use_hidden),use_recent=bool(use_recent),use_physical=bool(use_physical),sense_dim=NODE_SENSE_DIM,
                encoder_hidden=int(mc['encoder_hidden']),state_dim=int(mc['state_dim']),node_dim=int(mc['node_embedding_dim']),
                horizon_dim=int(mc['horizon_embedding_dim']),extra_dim=int(mc['extra_dim']),hidden=int(mc['hidden']),dropout=float(mc['dropout']))


def train_representation(name, seed, epochs, patience, cfg, dev, speed_std,
                         tr, bt, train_ids, dev_ids, nftr, rec_tr, phy_tr,
                         hidden_train, hidden_dev, use_hidden, use_recent, use_physical,
                         thresholds, q50, pwt, rd):
    pc=cfg['query_error_state_diagnostics'];mc=pc['model'];tc=pc['training'];H=int(cfg['experiment']['history']);P=int(cfg['experiment']['horizon']);N=int(nftr.shape[1])
    hdim=int(hidden_train.shape[-1]) if use_hidden else 1
    args=make_model_args(nftr,N,P,H,hdim,mc,use_hidden,use_recent,use_physical)
    seed_all(int(seed));model=QueryErrorStateNet(**args).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=float(tc['lr']),weight_decay=float(tc['weight_decay']))
    best=-1e18;wait=0;path=rd/f'{slug(name)}.seed{seed}.pt';history=[]
    dummy_train=np.zeros((len(train_ids),N,1),np.float16) if not use_hidden else hidden_train
    dummy_dev=np.zeros((len(dev_ids),N,1),np.float16) if not use_hidden else hidden_dev
    for ep in range(1,int(epochs)+1):
        model.train();losses=[]
        for pos in batches(len(train_ids),int(tc['batch_size']),True,int(seed)+1009*ep):
            ids=train_ids[pos]
            em,dl=model(t(nftr[ids],dev),t(dummy_train[pos],dev),t(rec_tr[ids],dev),t(phy_tr[ids],dev))
            er=t(raw_error(tr,bt,ids),dev);vv=t(np.asarray(tr.valid[ids],bool),dev,torch.bool)
            loss,_=query_direction_loss(em,dl,er,vv,speed_std,q50,pwt,float(tc['huber_beta']),float(tc['regression_weight']),float(tc['direction_weight']),float(tc['magnitude_weight']),float(tc['target_weight_clip']))
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(tc['grad_clip']));opt.step();losses.append(float(loss.detach().cpu()))
        # dev prediction
        ee=[];pp=[];model.eval()
        with torch.no_grad():
            for pos in batches(len(dev_ids),int(tc['eval_batch_size'])):
                ids=dev_ids[pos];e,l=model(t(nftr[ids],dev),t(dummy_dev[pos],dev),t(rec_tr[ids],dev),t(phy_tr[ids],dev));ee.append(e.cpu().numpy().astype(np.float32));pp.append(torch.sigmoid(l).cpu().numpy().astype(np.float32))
        pe=np.concatenate(ee);prob=np.concatenate(pp);derr=raw_error(tr,bt,dev_ids)
        met=evaluate_direction_probe(pe,prob,derr,tr,dev_ids,thresholds,speed_std,int(tc['metric_sample_points']),int(seed)+ep)
        sc=dev_score(met);mq=met['views']['masked'];history.append({'epoch':ep,'loss':float(np.mean(losses)),'dev_objective':sc,'q75_auc':mq['q75'].get('auroc'),'q90_auc':mq['q90'].get('auroc'),'spearman':mq['regression'].get('spearman')})
        print(f'REP {name} seed={seed} EPOCH {ep:02d} loss={np.mean(losses):.5f} q75_auc={mq["q75"].get("auroc")} q75_bal={mq["q75"].get("balanced_accuracy")} q90_auc={mq["q90"].get("auroc")} q90_bal={mq["q90"].get("balanced_accuracy")} spearman={mq["regression"].get("spearman")}',flush=True)
        if sc>best+1e-6:
            best=sc;wait=0;torch.save({'epoch':ep,'state_dict':model.state_dict(),'model_args':args,'dev_objective':sc,'seed':int(seed),'name':name},path)
        else:
            wait+=1
            if wait>=int(patience):break
    ck=torch.load(path,map_location=dev,weights_only=False);model.load_state_dict(ck['state_dict']);model.eval();ee=[];pp=[]
    with torch.no_grad():
        for pos in batches(len(dev_ids),int(tc['eval_batch_size'])):
            ids=dev_ids[pos];e,l=model(t(nftr[ids],dev),t(dummy_dev[pos],dev),t(rec_tr[ids],dev),t(phy_tr[ids],dev));ee.append(e.cpu().numpy().astype(np.float32));pp.append(torch.sigmoid(l).cpu().numpy().astype(np.float32))
    pe=np.concatenate(ee);prob=np.concatenate(pp);derr=raw_error(tr,bt,dev_ids);th,grid=calibrate(prob,derr,tr,dev_ids,thresholds,[float(x) for x in tc['direction_threshold_grid']])
    met=evaluate_direction_probe(pe,prob,derr,tr,dev_ids,thresholds,speed_std,int(tc['metric_sample_points']),int(seed)+7000,direction_threshold=th)
    return {'name':name,'seed':int(seed),'checkpoint':str(path),'best_epoch':int(ck['epoch']),'dev_objective':float(ck['dev_objective']),'direction_threshold':float(th),'threshold_grid':grid,'dev':met,'history':history,'model_args':args}


def final_eval(run, cfg, dev, speed_std, va, bv, final_ids, nfva, rec_va, phy_va, hidden_final, use_hidden,use_recent,use_physical, thresholds):
    tc=cfg['query_error_state_diagnostics']['training'];ck=torch.load(run['checkpoint'],map_location=dev,weights_only=False);model=QueryErrorStateNet(**ck['model_args']).to(dev);model.load_state_dict(ck['state_dict']);model.eval();N=int(nfva.shape[1]);dummy=np.zeros((len(final_ids),N,1),np.float16) if not use_hidden else hidden_final
    ee=[];pp=[]
    with torch.no_grad():
        for pos in batches(len(final_ids),int(tc['eval_batch_size'])):
            ids=final_ids[pos];e,l=model(t(nfva[ids],dev),t(dummy[pos],dev),t(rec_va[ids],dev),t(phy_va[ids],dev));ee.append(e.cpu().numpy().astype(np.float32));pp.append(torch.sigmoid(l).cpu().numpy().astype(np.float32))
    pe=np.concatenate(ee);prob=np.concatenate(pp);err=raw_error(va,bv,final_ids)
    return evaluate_direction_probe(pe,prob,err,va,final_ids,thresholds,speed_std,int(tc['metric_sample_points']),int(run['seed'])+9000,direction_threshold=float(run['direction_threshold']))


def aggregate_runs(runs):
    vals=np.asarray([r['dev_objective'] for r in runs],np.float64)
    order=np.argsort(vals);rep=runs[int(order[len(order)//2])]
    return {'mean_dev_objective':float(vals.mean()),'std_dev_objective':float(vals.std()),'representative_seed':int(rep['seed']),'representative_checkpoint':rep['checkpoint'],'representative_run':rep,'runs':runs}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',default='development',choices=['development']);ap.add_argument('--seed',type=int,default=None);a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/presure.yaml')
    if str(cfg.get('version'))!='0.9.5.1':raise RuntimeError(f'expected v0.9.5.1 got {cfg.get("version")}')
    base_seed=int(cfg['experiment']['seed'] if a.seed is None else a.seed);seed_all(base_seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pc=cfg['query_error_state_diagnostics'];lc=pc['layer_sweep'];tc=pc['training'];cache=cache_dir(cfg,a.stage);art=art_dir(cfg,a.stage)
    tr=CacheSplit(cache,'train');va=CacheSplit(cache,'val');_,speed_std=cache_stats(cache)
    nftr=np.load(art/'train_node_features.npy',mmap_mode='r');nfva=np.load(art/'val_node_features.npy',mmap_mode='r');bt=np.load(art/'train_base_pred.npy',mmap_mode='r');bv=np.load(art/'val_base_pred.npy',mmap_mode='r')
    rec_tr=np.load(art/'train_recent_error_state_repaired.npy',mmap_mode='r');rec_va=np.load(art/'val_recent_error_state_repaired.npy',mmap_mode='r');phy_tr=np.load(art/'train_physical_transition.npy',mmap_mode='r');phy_va=np.load(art/'val_physical_transition.npy',mmap_mode='r')
    manifest=json.loads((art/'query_error_state_diagnostics_manifest.json').read_text());catalog=json.loads((art/'patchstg_hidden_layer_catalog.json').read_text())
    split_source,splits=load_v09a_splits(cfg,a.stage);train_ids=splits['train_ids'];dev_ids=splits['dev_ids'];final_ids=splits['final_val_ids']
    thresholds=derive_abs_error_thresholds(raw_error(tr,bt,train_ids),np.asarray(tr.valid[train_ids],bool),tuple(float(x) for x in pc['thresholds']['evaluation_quantiles']));pw,counts=train_class_pos_weight(tr,bt,train_ids,thresholds);q50=t(np.asarray(thresholds['q50'],np.float32)[None,:,None],dev);pwt=torch.tensor(pw,device=dev)
    rd=run_dir(cfg,'QueryErrorStateRepresentationDiagnostics',a.stage,stamp(),base_seed);rd.mkdir(parents=True,exist_ok=True)
    for k,v in splits.items():np.save(rd/f'{k}.npy',v)
    write_json(rd/'train_error_thresholds.json',{**thresholds,'class_counts_q50':counts,'source':'exact v0.9A train_ids only'})
    print('='*80,flush=True);print('PreSuRe v0.9.5.1 Representation Diagnostics',flush=True);print(f'v0.9A split source={split_source}',flush=True);print(f'recent_error_train={manifest["recent_error_train"]}',flush=True);print(f'recent_error_val={manifest["recent_error_val"]}',flush=True);print('Layer sweep uses Train-dev only; Final Validation is untouched until representation selection.',flush=True);print('='*80,flush=True)

    adapter,_,_,_=load_patchstg_adapter(cfg,a.stage,dev);ids_tv=np.concatenate([train_ids,dev_ids]);ntr=len(train_ids);screen=[]
    choices=[HiddenLayerChoice(**x) for x in catalog['sweep_candidates']]
    # Stage A: one-seed layer sweep. Every candidate gets the same query/base features and
    # differs only in the audited PatchSTG activation.
    for li,c in enumerate(choices):
        outw=min(int(lc['projected_dim']),int(c.raw_width));ps=int(lc['projection_seed'])+17*li
        print(f'INFER LAYER {li+1}/{len(choices)} {c.module_name} call={c.call_ordinal} width={c.raw_width}->{outw}',flush=True)
        h=infer_hidden_ids(adapter,cache,'train',ids_tv,c.module_name,c.call_ordinal,c.raw_width,outw,ps,int(lc['inference_batch_size']),int(nftr.shape[1]));ht=h[:ntr];hd=h[ntr:]
        name=f'layer::{c.module_name}::call{c.call_ordinal}'
        r=train_representation(name,base_seed,int(lc['screen_epochs']),int(lc['screen_patience']),cfg,dev,speed_std,tr,bt,train_ids,dev_ids,nftr,rec_tr,phy_tr,ht,hd,True,False,False,thresholds,q50,pwt,rd)
        screen.append({'layer':c.to_dict(),'projected_dim':outw,'projection_seed':ps,'run':r})
        del h,ht,hd;gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None
    screen.sort(key=lambda x:x['run']['dev_objective'],reverse=True)
    top=screen[:max(1,min(int(lc['confirm_top_k']),len(screen)))]
    # Always confirm the exact v0.9.5 reference layer (input_st_fc) with the same
    # epochs/seeds as the Top-K. This makes the hidden-gain diagnostic apples-to-apples.
    reference_item=None
    for x in screen:
        if x['layer']['module_name'].endswith('input_st_fc') or x['layer']['module_name']=='input_st_fc':
            reference_item=x;break
    confirm_items=list(top)
    if reference_item is not None and not any((x['layer']['module_name'],x['layer']['call_ordinal'])==(reference_item['layer']['module_name'],reference_item['layer']['call_ordinal']) for x in confirm_items):
        confirm_items.append(reference_item)
    print('LAYER SWEEP TOP:',[(x['layer']['module_name'],x['run']['dev_objective']) for x in top],flush=True)
    print('LAYER CONFIRM SET:',[(x['layer']['module_name'],x['run']['dev_objective']) for x in confirm_items],flush=True)

    # Stage B: confirm Top-K layers plus the v0.9.5 input_st_fc reference over multiple seeds.
    confirm_seeds=[int(x) for x in lc['confirm_seeds']];layer_confirm={}
    for rank,item in enumerate(confirm_items):
        c=HiddenLayerChoice(**item['layer']);outw=int(item['projected_dim']);ps=int(item['projection_seed']);h=infer_hidden_ids(adapter,cache,'train',ids_tv,c.module_name,c.call_ordinal,c.raw_width,outw,ps,int(lc['inference_batch_size']),int(nftr.shape[1]));ht=h[:ntr];hd=h[ntr:]
        runs=[]
        for s in confirm_seeds:
            runs.append(train_representation(f'confirm::{c.module_name}::call{c.call_ordinal}',s,int(tc['epochs']),int(tc['patience']),cfg,dev,speed_std,tr,bt,train_ids,dev_ids,nftr,rec_tr,phy_tr,ht,hd,True,False,False,thresholds,q50,pwt,rd))
        layer_confirm[f'{c.module_name}::call{c.call_ordinal}']={'layer':c.to_dict(),'projected_dim':outw,'projection_seed':ps,**aggregate_runs(runs)}
        del h,ht,hd;gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None
    best_layer_key=max(layer_confirm,key=lambda k:layer_confirm[k]['mean_dev_objective']);best_layer=layer_confirm[best_layer_key];bc=HiddenLayerChoice(**best_layer['layer'])
    print(f'CONFIRMED BEST HIDDEN {best_layer_key} mean_dev={best_layer["mean_dev_objective"]:.6f}',flush=True)

    # Infer selected hidden once for representation-level comparisons.
    h=infer_hidden_ids(adapter,cache,'train',ids_tv,bc.module_name,bc.call_ordinal,bc.raw_width,int(best_layer['projected_dim']),int(best_layer['projection_seed']),int(lc['inference_batch_size']),int(nftr.shape[1]));ht=h[:ntr];hd=h[ntr:]
    recent_available=bool(manifest.get('recent_error_train',{}).get('available')) and bool(manifest.get('recent_error_val',{}).get('available'))
    specs={
        'base':(False,False,False),
        'physical':(False,False,True),
        'recent_repaired':(False,recent_available,False),
        'physical_recent':(False,recent_available,True),
        'best_hidden':(True,False,False),
        'best_hidden_physical':(True,False,True),
        'full_repaired':(True,recent_available,True),
    }
    rep_results={}
    for name in pc['representation_variants']:
        uh,ur,up=specs[name]
        # Reuse the 3-seed confirmed best-hidden runs when this representation is exactly hidden-only.
        if name=='best_hidden':
            rep_results[name]=best_layer;continue
        runs=[]
        for s in confirm_seeds:
            runs.append(train_representation(name,s,int(tc['epochs']),int(tc['patience']),cfg,dev,speed_std,tr,bt,train_ids,dev_ids,nftr,rec_tr,phy_tr,ht,hd,uh,ur,up,thresholds,q50,pwt,rd))
        rep_results[name]={'spec':{'use_hidden':uh,'use_recent':ur,'use_physical':up},**aggregate_runs(runs)}
    selected=max(rep_results,key=lambda k:rep_results[k]['mean_dev_objective']);sel=rep_results[selected];rep_run=sel['representative_run']
    print(f'SELECTED REPRESENTATION {selected} mean_dev={sel["mean_dev_objective"]:.6f} representative_seed={rep_run["seed"]}',flush=True)

    # Final Validation is touched only now, after both layer and representation selection.
    uh,ur,up=specs[selected]
    if uh:
        hf=infer_hidden_ids(adapter,cache,'val',final_ids,bc.module_name,bc.call_ordinal,bc.raw_width,int(best_layer['projected_dim']),int(best_layer['projection_seed']),int(lc['inference_batch_size']),int(nfva.shape[1]))
    else:
        hf=np.zeros((len(final_ids),int(nfva.shape[1]),1),np.float16)
    fm=final_eval(rep_run,cfg,dev,speed_std,va,bv,final_ids,nfva,rec_va,phy_va,hf,uh,ur,up,thresholds)
    status,checks=decision(fm,cfg)

    # Diagnostic gains are computed on Train-dev only to avoid turning Final Validation
    # into a representation-selection surface.
    v095_ref_key=None
    for k,r in layer_confirm.items():
        if r['layer']['module_name'].endswith('input_st_fc') or r['layer']['module_name']=='input_st_fc':
            v095_ref_key=k;break
    v095_ref=float(layer_confirm[v095_ref_key]['mean_dev_objective']) if v095_ref_key is not None else None
    hidden_gain=(float(best_layer['mean_dev_objective'])-float(v095_ref)) if v095_ref is not None else None
    recent_gain_hidden=float(rep_results['full_repaired']['mean_dev_objective']-rep_results['best_hidden_physical']['mean_dev_objective']) if recent_available else None
    recent_gain_physical=float(rep_results['physical_recent']['mean_dev_objective']-rep_results['physical']['mean_dev_objective']) if recent_available else None
    dcfg=pc['decision'];diag={
        'deeper_hidden_found':not (bc.module_name.endswith('input_st_fc') or bc.module_name=='input_st_fc'),
        'confirmed_v095_reference_key':v095_ref_key,
        'hidden_gain_over_v095_input_fc':hidden_gain,
        'hidden_gain_pass':hidden_gain is not None and hidden_gain>=float(dcfg['minimum_hidden_gain_over_v095_input_fc']),
        'recent_error_available':recent_available,
        'recent_gain_over_best_hidden_physical':recent_gain_hidden,
        'recent_gain_over_physical':recent_gain_physical,
        'recent_gain_pass':recent_gain_hidden is not None and recent_gain_physical is not None and max(recent_gain_hidden,recent_gain_physical)>=float(dcfg['minimum_recent_gain_over_without_recent']),
    }

    summary={'version':'0.9.5.1','implementation_revision':cfg.get('implementation_revision'),'run_dir':str(rd),'split_source_run':str(split_source),
             'artifact_manifest':manifest,'hidden_catalog':catalog,'layer_screen':screen,'layer_confirmation':layer_confirm,'selected_hidden_layer':best_layer_key,
             'representation_selection':'mean Train-dev objective over configured confirmation seeds; Final Validation never selects layer/representation/seed',
             'representation_results':rep_results,'selected_representation':selected,'selected_representative_seed':int(rep_run['seed']),'final_val':fm,
             'diagnostic_findings':diag,'decision':status,'decision_checks':checks}
    write_json(rd/'summary.json',summary);(rd/'DECISION.txt').write_text(status+'\n'+json.dumps({'checks':checks,'diagnostic_findings':diag},indent=2,sort_keys=True)+'\n')
    with (rd/'layer_sweep.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['rank','module','call','exec','depth','leaf','raw_width','projected_dim','screen_dev_objective'])
        for rank,x in enumerate(screen,1):
            c=x['layer'];w.writerow([rank,c['module_name'],c['call_ordinal'],c['execution_index'],c['module_depth'],c['is_leaf'],c['raw_width'],x['projected_dim'],x['run']['dev_objective']])
    with (rd/'representation_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['representation','mean_dev_objective','std_dev_objective','representative_seed','use_recent_effective'])
        for n,r in rep_results.items():w.writerow([n,r['mean_dev_objective'],r['std_dev_objective'],r['representative_seed'],specs[n][1]])
    latest=Path(cfg['project']['exp_root'])/'QueryErrorStateRepresentationDiagnostics'/cfg['experiment']['subgraph']/a.stage/'LATEST.json';latest.parent.mkdir(parents=True,exist_ok=True);write_json(latest,{'run_dir':str(rd),'summary':str(rd/'summary.json'),'selected_hidden_layer':best_layer_key,'selected_representation':selected,'decision':status})
    print('='*80,flush=True);print(f'SELECTED hidden={best_layer_key}',flush=True);print(f'SELECTED representation={selected}',flush=True);print('DIAGNOSTIC FINDINGS:',json.dumps(diag,indent=2),flush=True);print('FINAL masked:',json.dumps(fm['views']['masked'],indent=2),flush=True);print(f'DECISION={status} checks={checks}',flush=True);print(f'SUMMARY {rd/"summary.json"}',flush=True);print('='*80,flush=True)

if __name__=='__main__':main()
