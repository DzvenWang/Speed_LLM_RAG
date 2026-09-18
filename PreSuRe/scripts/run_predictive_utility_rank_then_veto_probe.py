#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse, csv, json, sys
import numpy as np
import torch

ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe')
ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))

from presure.utils import load_yaml, write_json, stamp, seed_all
from presure.contract import cache_dir
from presure.data import CacheSplit, cache_stats
from presure.models import PredictiveUtilityRerankerV09A, StrongHarmVetoHead
from presure.pipeline import load_artifacts, run_dir
from presure.predictive_utility import node_horizon_utility_targets, risk_score
from presure.harm_veto import (
    strong_harm_focal_loss, sequential_veto_choice, frozen_rank_order,
    HarmClassifierAccumulator, RankThenVetoAccumulator,
)
from run_predictive_utility_reranker_probe import batches, t, view_mask, gather_batch


def load_v09a(cfg, stage, dev):
    base=Path(cfg['project']['exp_root'])/'PredictiveUtilityRetrieval'/cfg['experiment']['subgraph']/stage
    lp=base/'LATEST.json'
    if not lp.exists(): raise FileNotFoundError(f'Missing v0.9A LATEST: {lp}')
    info=json.loads(lp.read_text()); cp=Path(info['checkpoint']); rd=Path(info['run_dir'])
    if not cp.exists(): raise FileNotFoundError(cp)
    ck=torch.load(cp,map_location=dev,weights_only=False)
    model=PredictiveUtilityRerankerV09A(**ck['model_args']).to(dev)
    model.load_state_dict(ck['state_dict']); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    beta=float(ck.get('risk_beta',0.5))
    splits={k:np.load(rd/f'{k}.npy') for k in ['train_ids','dev_ids','final_val_ids']}
    return model, ck, cp, rd, beta, splits


def model_outputs(ranker, pack, beta, veto=None):
    qctx,cctx,qs,ss,rr,qbase,sbase,coarse,cm,valid,target,_=pack
    with torch.no_grad():
        role,mu,sigma,blogit=ranker.forward_features(qctx,cctx,qs,ss,rr,qbase,sbase,coarse)
        rscore=risk_score(mu,sigma,beta)
    hlogit=None if veto is None else veto(role.detach(),mu.detach(),sigma.detach(),blogit.detach(),rscore.detach())
    return role,mu,sigma,blogit,rscore,hlogit,cm,valid,target


def eval_classifier(ranker,veto,cfg,ds,ids,cand,node_q,node_mem,residual_mem,base,mean,std,dev,beta,seed):
    tc=cfg['predictive_utility_rank_then_veto']['training']; acc=HarmClassifierAccumulator(seed=seed)
    veto.eval()
    with torch.no_grad():
        for src in batches(ids,int(tc['eval_batch_size'])):
            pack=gather_batch(cfg, None, ds, src, cand, node_q,node_mem,residual_mem,base,mean,std,dev)
            *_,hlogit,cm,valid,target=model_outputs(ranker,pack,beta,veto)
            acc.update(hlogit,target,cm,valid,float(tc['strong_harm_threshold']))
    return acc.summary(0.5)


def policy_grid_eval(ranker,veto,cfg,ds,ids,cand,node_q,node_mem,residual_mem,base,mean,std,dev,beta,views=('masked',)):
    pc=cfg['predictive_utility_rank_then_veto']; tc=pc['training']; pol=pc['policy']
    combos=[(float(th),int(m)) for th in pol['veto_threshold_grid'] for m in pol['max_checks_grid']]
    acc={(th,m,v):RankThenVetoAccumulator(float(cfg['predictive_utility_retrieval']['training']['beneficial_threshold']),float(tc['strong_harm_threshold'])) for th,m in combos for v in views}
    veto.eval()
    with torch.no_grad():
        for src in batches(ids,int(tc['eval_batch_size'])):
            pack=gather_batch(cfg,None,ds,src,cand,node_q,node_mem,residual_mem,base,mean,std,dev)
            role,mu,sigma,blogit,rscore,hlogit,cm,valid,target=model_outputs(ranker,pack,beta,veto)
            hp=torch.sigmoid(hlogit)
            order=frozen_rank_order(rscore,cm,valid)
            for th,m in combos:
                for v in views:
                    vm=t(view_mask(ds,src,v),dev,torch.bool)
                    choice,accept,checks,_=sequential_veto_choice(rscore,hp,cm,vm,th,m,order=order)
                    acc[(th,m,v)].update(choice,accept,checks,target,cm,vm)
    return {(th,m):{v:acc[(th,m,v)].summary() for v in views} for th,m in combos}


def choose_policy(grid,cfg):
    p=cfg['predictive_utility_rank_then_veto']['policy']; best=None
    for (th,m),views in grid.items():
        x=views['masked']; cov=x['coverage']; hexp=cov*x['accepted_harmful_rate']; sexp=cov*x['accepted_strong_harm_rate']
        obj=(x['mean_policy_relative_utility']+float(p['capture_reward'])*x['beneficial_capture_ratio']
             -float(p['harmful_exposure_penalty'])*hexp-float(p['strong_harm_exposure_penalty'])*sexp
             -float(p['regret_penalty'])*x['mean_safe_policy_regret'])
        feasible=cov+1e-9>=float(p['min_coverage'])
        key=(1 if feasible else 0,obj,-x['accepted_strong_harm_rate'],-x['accepted_harmful_rate'],x['coverage'],-m)
        row={'veto_threshold':th,'max_checks':m,'objective':float(obj),'feasible':bool(feasible),**x}
        if best is None or key>best[0]: best=(key,row)
    return best[1]


def fixed_policy_eval(ranker,veto,cfg,ds,ids,cand,node_q,node_mem,residual_mem,base,mean,std,dev,beta,th,max_checks):
    tc=cfg['predictive_utility_rank_then_veto']['training']; bt=float(cfg['predictive_utility_retrieval']['training']['beneficial_threshold'])
    out={v:RankThenVetoAccumulator(bt,float(tc['strong_harm_threshold'])) for v in ['overall','masked','support']}
    base_acc={v:RankThenVetoAccumulator(bt,float(tc['strong_harm_threshold'])) for v in ['overall','masked','support']}
    veto.eval()
    with torch.no_grad():
        for src in batches(ids,int(tc['eval_batch_size'])):
            pack=gather_batch(cfg,None,ds,src,cand,node_q,node_mem,residual_mem,base,mean,std,dev)
            role,mu,sigma,blogit,rscore,hlogit,cm,valid,target=model_outputs(ranker,pack,beta,veto); hp=torch.sigmoid(hlogit)
            for v in ['overall','masked','support']:
                vm=t(view_mask(ds,src,v),dev,torch.bool)
                use=cm[:,:,:,None]&vm[:,:,None,:]; order=frozen_rank_order(rscore,cm,vm)
                choice,accept,checks,_=sequential_veto_choice(rscore,hp,cm,vm,th,max_checks,order=order)
                out[v].update(choice,accept,checks,target,cm,vm)
                # frozen v0.9A no-veto baseline = rank top1, 100% coverage
                top=order[:,:,0,:]; acc0=(use.sum(2)>0)&vm; chk=torch.ones_like(top,dtype=torch.int16)
                base_acc[v].update(top,acc0,chk,target,cm,vm)
    return {'frozen_v09a_no_veto':{v:a.summary() for v,a in base_acc.items()},'rank_then_veto':{v:a.summary() for v,a in out.items()}}


def regret_reduction(a,b):
    x=float(a['mean_safe_policy_regret']); y=float(b['mean_safe_policy_regret']); return 0.0 if x<=1e-12 else (x-y)/x


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development']); ap.add_argument('--seed',type=int,default=None); a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/presure.yaml')
    if str(cfg.get('version')) not in {'0.9.2','0.9.3','0.9.4','0.9.5'}: raise RuntimeError(f'v0.9A.2 requires 0.9.2/0.9.3, got {cfg.get("version")}')
    seed=int(cfg['experiment']['seed'] if a.seed is None else a.seed); seed_all(seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    art=load_artifacts(cfg,a.stage); cache=cache_dir(cfg,a.stage); tr=CacheSplit(cache,'train'); va=CacheSplit(cache,'val'); mean,std=cache_stats(cache)
    node_mem=np.load(art/'train_node_features.npy',mmap_mode='r'); node_val=np.load(art/'val_node_features.npy',mmap_mode='r'); residual=np.load(art/'train_residual.npy',mmap_mode='r')
    train_base=np.load(art/'train_base_pred.npy',mmap_mode='r'); val_base=np.load(art/'val_base_pred.npy',mmap_mode='r'); train_cand=np.load(art/'train_candidates.npz'); val_cand=np.load(art/'val_candidates.npz')
    ranker,rck,rank_cp,rank_rd,beta,splits=load_v09a(cfg,a.stage,dev); train_ids=splits['train_ids']; dev_ids=splits['dev_ids']; fval_ids=splits['final_val_ids']
    rd=run_dir(cfg,'PredictiveUtilityRankThenVeto',a.stage,stamp(),seed); rd.mkdir(parents=True,exist_ok=True)
    for k,v in splits.items(): np.save(rd/f'{k}.npy',v)
    mc=cfg['predictive_utility_rank_then_veto']['model']; veto=StrongHarmVetoHead(96,int(mc['hidden']),float(mc['dropout'])).to(dev)
    tc=cfg['predictive_utility_rank_then_veto']['training']; opt=torch.optim.AdamW(veto.parameters(),lr=float(tc['lr']),weight_decay=float(tc['weight_decay']))
    best=-1e18; best_ep=0; wait=0; best_path=rd/'strong_harm_veto_best.pt'
    print('='*60,flush=True); print('PreSuRe v0.9A.2 Rank-then-Veto',flush=True); print(f'Frozen v0.9A ranker={rank_cp} beta={beta:g}',flush=True); print(f'train={len(train_ids)} dev={len(dev_ids)} final_val={len(fval_ids)} K=24',flush=True); print('Ranker is frozen; veto predicts strong harm only. No Qwen / Dual Memory.',flush=True); print('='*60,flush=True)
    for ep in range(1,int(tc['epochs'])+1):
        veto.train(); losses=[]
        for src in batches(train_ids,int(tc['batch_size']),True,seed+1009*ep):
            pack=gather_batch(cfg,None,tr,src,train_cand,node_mem,node_mem,residual,train_base,mean,std,dev)
            role,mu,sigma,blogit,rscore,_,cm,valid,target=model_outputs(ranker,pack,beta,None)
            hlogit=veto(role.detach(),mu.detach(),sigma.detach(),blogit.detach(),rscore.detach())
            loss,_=strong_harm_focal_loss(hlogit,target,cm,valid,float(tc['strong_harm_threshold']),float(tc['focal_gamma']),float(tc['focal_alpha']),rscore.detach(),float(tc['top_rank_weight']),int(tc['top_rank_k']))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(veto.parameters(),float(tc['grad_clip'])); opt.step(); losses.append(float(loss.detach().cpu()))
        cls=eval_classifier(ranker,veto,cfg,tr,dev_ids,train_cand,node_mem,node_mem,residual,train_base,mean,std,dev,beta,seed+ep*1777)
        grid=policy_grid_eval(ranker,veto,cfg,tr,dev_ids,train_cand,node_mem,node_mem,residual,train_base,mean,std,dev,beta,views=('masked',)); choice=choose_policy(grid,cfg)
        score=float(choice['objective']); print(f'EPOCH {ep:02d} loss={np.mean(losses):.5f} harm_auc={cls["auroc"]:.4f} harm_precision={cls["precision"]:.4f} harm_recall={cls["recall"]:.4f} th={choice["veto_threshold"]:.2f} checks={choice["max_checks"]} coverage={choice["coverage"]:.4f} beneficial={choice["accepted_beneficial_rate"]:.4f} harmful={choice["accepted_harmful_rate"]:.4f} strong_harm={choice["accepted_strong_harm_rate"]:.4f} regret={choice["mean_safe_policy_regret"]:.5f}',flush=True)
        if score>best:
            best=score; best_ep=ep; wait=0; torch.save({'model_type':'strong_harm_veto_v09a2','state_dict':veto.state_dict(),'model_args':{'role_dim':96,'hidden':int(mc['hidden']),'dropout':float(mc['dropout'])},'epoch':ep,'dev_objective':score,'frozen_ranker_checkpoint':str(rank_cp),'frozen_risk_beta':beta,'veto_threshold':float(choice['veto_threshold']),'max_checks':int(choice['max_checks']),'strong_harm_threshold':float(tc['strong_harm_threshold'])},best_path)
        else: wait+=1
        if wait>=int(tc['patience']): print(f'EARLY STOP epoch={ep} best_epoch={best_ep}',flush=True); break
    ck=torch.load(best_path,map_location=dev,weights_only=False); veto.load_state_dict(ck['state_dict']); th=float(ck['veto_threshold']); max_checks=int(ck['max_checks'])
    dev_cls=eval_classifier(ranker,veto,cfg,tr,dev_ids,train_cand,node_mem,node_mem,residual,train_base,mean,std,dev,beta,seed+5551)
    final_cls=eval_classifier(ranker,veto,cfg,va,fval_ids,val_cand,node_val,node_mem,residual,val_base,mean,std,dev,beta,seed+7771)
    final=fixed_policy_eval(ranker,veto,cfg,va,fval_ids,val_cand,node_val,node_mem,residual,val_base,mean,std,dev,beta,th,max_checks)
    b=final['frozen_v09a_no_veto']['masked']; n=final['rank_then_veto']['masked']; rr=regret_reduction(b,n); dc=cfg['predictive_utility_rank_then_veto']['decision']
    checks={'coverage':n['coverage']>=float(dc['coverage_go']),'accepted_beneficial':n['accepted_beneficial_rate']>=float(dc['accepted_beneficial_go']),'accepted_harmful':n['accepted_harmful_rate']<=float(dc['accepted_harmful_go']),'accepted_strong_harm':n['accepted_strong_harm_rate']<=float(dc['accepted_strong_harm_go']),'safe_regret_reduction':rr>=float(dc['safe_regret_reduction_go'])}
    conditional=(n['coverage']>=float(dc['coverage_conditional']) and n['accepted_harmful_rate']<=float(dc['accepted_harmful_conditional']) and n['accepted_strong_harm_rate']<=float(dc['accepted_strong_harm_conditional']) and rr>=float(dc['safe_regret_reduction_conditional']))
    decision='GO' if all(checks.values()) else ('CONDITIONAL' if conditional else 'NO_GO')
    summary={'version':cfg['version'],'implementation_revision':cfg.get('implementation_revision'),'model_type':'rank_then_veto_v09a2','stage':a.stage,'seed':seed,'run_dir':str(rd),'checkpoint':str(best_path),'frozen_ranker_checkpoint':str(rank_cp),'frozen_ranker_run':str(rank_rd),'frozen_risk_beta':beta,'best_epoch':int(ck['epoch']),'chosen_veto_threshold':th,'chosen_max_checks':max_checks,'strong_harm_threshold':float(tc['strong_harm_threshold']),'ranker_frozen':True,'ranking_reordered_by_veto':False,'qwen_used':False,'dual_memory_used':False,'dev_harm_classifier':dev_cls,'final_val_harm_classifier':final_cls,'final_val_policy':final,'masked_safe_regret_reduction_vs_frozen_v09a':rr,'go_checks':checks,'decision':decision}
    write_json(rd/'summary.json',summary)
    with (rd/'policy_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['variant','view','coverage','accepted_beneficial_rate','accepted_harmful_rate','accepted_strong_harm_rate','beneficial_capture_ratio','mean_policy_relative_utility','mean_accepted_relative_utility','mean_safe_policy_regret','mean_checks','fallback_rate'])
        for variant,views in final.items():
            for v,m in views.items(): w.writerow([variant,v,m['coverage'],m['accepted_beneficial_rate'],m['accepted_harmful_rate'],m['accepted_strong_harm_rate'],m['beneficial_capture_ratio'],m['mean_policy_relative_utility'],m['mean_accepted_relative_utility'],m['mean_safe_policy_regret'],m['mean_checks'],m['fallback_rate']])
    (rd/'DECISION.txt').write_text(f'{decision}\nbest_epoch={ck["epoch"]}\nfrozen_beta={beta:g}\nveto_threshold={th:g}\nmax_checks={max_checks}\nmasked_coverage={n["coverage"]:.6f}\nmasked_accepted_beneficial={n["accepted_beneficial_rate"]:.6f}\nmasked_accepted_harmful={n["accepted_harmful_rate"]:.6f}\nmasked_accepted_strong_harm={n["accepted_strong_harm_rate"]:.6f}\nmasked_safe_regret_reduction_vs_v09a={rr:.6f}\n')
    latest=Path(cfg['project']['exp_root'])/'PredictiveUtilityRankThenVeto'/cfg['experiment']['subgraph']/a.stage/'LATEST.json'; latest.parent.mkdir(parents=True,exist_ok=True); write_json(latest,{'run_dir':str(rd),'checkpoint':str(best_path),'summary':str(rd/'summary.json'),'variant':'PredictiveUtilityRankThenVeto','stage':a.stage,'version':cfg['version']})
    print('='*60,flush=True); print(f'BEST epoch={ck["epoch"]} veto_threshold={th:g} max_checks={max_checks}',flush=True); print('FINAL harm classifier:',json.dumps(final_cls,indent=2),flush=True); print('FINAL masked frozen v0.9A:',json.dumps(b,indent=2),flush=True); print('FINAL masked rank-then-veto:',json.dumps(n,indent=2),flush=True); print(f'MASKED safe-regret reduction vs frozen v0.9A = {rr:.4%}',flush=True); print(f'DECISION = {decision} checks={checks}',flush=True); print(f'SUMMARY {rd/"summary.json"}',flush=True); print('='*60,flush=True)

if __name__=='__main__': main()
