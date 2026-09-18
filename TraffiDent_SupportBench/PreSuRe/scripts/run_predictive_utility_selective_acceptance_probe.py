#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse, csv, json, sys
import numpy as np
import torch

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe')
ROOT = ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'scripts'))

from presure.utils import load_yaml, write_json, stamp, seed_all
from presure.contract import cache_dir
from presure.data import CacheSplit, cache_stats
from presure.models import PredictiveUtilityRerankerV09A, SelectedSupportAcceptanceHead
from presure.pipeline import load_artifacts, run_dir
from presure.predictive_utility import risk_score
from presure.selective_acceptance import (
    frozen_top1_acceptance_context, selected_target, selective_acceptance_loss,
    AcceptanceClassifierAccumulator, SelectiveAcceptanceAccumulator,
)
from run_predictive_utility_reranker_probe import batches, t, view_mask, gather_batch


def load_v09a(cfg, stage, dev):
    base = Path(cfg['project']['exp_root'])/'PredictiveUtilityRetrieval'/cfg['experiment']['subgraph']/stage
    lp = base/'LATEST.json'
    if not lp.exists(): raise FileNotFoundError(f'Missing v0.9A LATEST: {lp}')
    info = json.loads(lp.read_text()); cp = Path(info['checkpoint']); rd = Path(info['run_dir'])
    if not cp.exists(): raise FileNotFoundError(cp)
    ck = torch.load(cp, map_location=dev, weights_only=False)
    model = PredictiveUtilityRerankerV09A(**ck['model_args']).to(dev)
    model.load_state_dict(ck['state_dict']); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    beta = float(ck.get('risk_beta', 0.5))
    splits = {k: np.load(rd/f'{k}.npy') for k in ['train_ids','dev_ids','final_val_ids']}
    return model, ck, cp, rd, beta, splits


def frozen_selected_outputs(ranker, pack, beta):
    qctx,cctx,qs,ss,rr,qbase,sbase,coarse,cm,valid,target,_ = pack
    with torch.no_grad():
        role,mu,sigma,blogit = ranker.forward_features(qctx,cctx,qs,ss,rr,qbase,sbase,coarse)
        rscore = risk_score(mu, sigma, beta)
        idx, feat, selected_valid = frozen_top1_acceptance_context(
            role, mu, sigma, blogit, rscore, cm, valid
        )
        su = selected_target(target, idx)
        use = cm[:, :, :, None] & valid[:, :, None, :]
        best = target.masked_fill(~use, -1e9).max(2).values
    return idx, feat.detach(), selected_valid, su.detach(), best.detach(), rscore.detach(), cm, valid, target


def eval_classifier(ranker, head, cfg, ds, ids, cand, node_q, node_mem, residual_mem,
                    base, mean, std, dev, beta, seed):
    tc = cfg['predictive_utility_selective_acceptance']['training']
    out = {v: AcceptanceClassifierAccumulator(float(tc['positive_utility_threshold']), seed=seed+100*i)
           for i,v in enumerate(['overall','masked','support'])}
    head.eval()
    with torch.no_grad():
        for src in batches(ids, int(tc['eval_batch_size'])):
            pack = gather_batch(cfg,None,ds,src,cand,node_q,node_mem,residual_mem,base,mean,std,dev)
            _,feat,sv,su,_,_,_,_,_ = frozen_selected_outputs(ranker,pack,beta)
            logit = head(feat)
            for v in ['overall','masked','support']:
                vm = t(view_mask(ds,src,v),dev,torch.bool)
                out[v].update(logit,su,sv & vm)
    return {v:a.summary(0.5) for v,a in out.items()}


def policy_grid_eval(ranker, head, cfg, ds, ids, cand, node_q, node_mem, residual_mem,
                     base, mean, std, dev, beta, views=('masked',)):
    pc = cfg['predictive_utility_selective_acceptance']; tc=pc['training']; pol=pc['policy']
    bt = float(cfg['predictive_utility_retrieval']['training']['beneficial_threshold'])
    ht = float(cfg['predictive_utility_retrieval']['training']['harmful_threshold'])
    ths = [float(x) for x in pol['threshold_grid']]
    acc = {(th,v): SelectiveAcceptanceAccumulator(bt,ht) for th in ths for v in views}
    head.eval()
    with torch.no_grad():
        for src in batches(ids,int(tc['eval_batch_size'])):
            pack = gather_batch(cfg,None,ds,src,cand,node_q,node_mem,residual_mem,base,mean,std,dev)
            _,feat,sv,su,best,_,_,_,_ = frozen_selected_outputs(ranker,pack,beta)
            prob = torch.sigmoid(head(feat))
            for v in views:
                vm = t(view_mask(ds,src,v),dev,torch.bool)
                vv = sv & vm
                for th in ths:
                    accept = vv & (prob >= th)
                    acc[(th,v)].update(su,accept,best,vv)
    return {th:{v:acc[(th,v)].summary() for v in views} for th in ths}


def choose_threshold(grid,cfg):
    p = cfg['predictive_utility_selective_acceptance']['policy']; best=None
    for th,views in grid.items():
        x=views['masked']; cov=x['coverage']; harm_exp=cov*x['accepted_harmful_rate']
        obj = (x['mean_policy_relative_utility']
               + float(p['capture_reward'])*x['beneficial_capture_ratio']
               - float(p['harmful_exposure_penalty'])*harm_exp)
        feasible = cov + 1e-9 >= float(p['min_coverage'])
        key = (1 if feasible else 0, obj, -x['mean_safe_policy_regret'],
               -x['accepted_harmful_rate'], x['accepted_positive_utility_rate'], cov, th)
        row = {'threshold':float(th),'objective':float(obj),'feasible':bool(feasible),**x}
        if best is None or key > best[0]: best=(key,row)
    return best[1]


def fixed_policy_eval(ranker, head, cfg, ds, ids, cand, node_q, node_mem, residual_mem,
                     base, mean, std, dev, beta, threshold):
    tc=cfg['predictive_utility_selective_acceptance']['training']
    bt=float(cfg['predictive_utility_retrieval']['training']['beneficial_threshold'])
    ht=float(cfg['predictive_utility_retrieval']['training']['harmful_threshold'])
    sel={v:SelectiveAcceptanceAccumulator(bt,ht) for v in ['overall','masked','support']}
    frozen={v:SelectiveAcceptanceAccumulator(bt,ht) for v in ['overall','masked','support']}
    head.eval()
    with torch.no_grad():
        for src in batches(ids,int(tc['eval_batch_size'])):
            pack=gather_batch(cfg,None,ds,src,cand,node_q,node_mem,residual_mem,base,mean,std,dev)
            _,feat,sv,su,best,_,_,_,_=frozen_selected_outputs(ranker,pack,beta)
            prob=torch.sigmoid(head(feat))
            for v in ['overall','masked','support']:
                vm=t(view_mask(ds,src,v),dev,torch.bool); vv=sv&vm
                sel[v].update(su,vv&(prob>=float(threshold)),best,vv)
                frozen[v].update(su,vv,best,vv)
    return {
        'frozen_v09a_top1':{v:a.summary() for v,a in frozen.items()},
        'selective_acceptance':{v:a.summary() for v,a in sel.items()},
    }


def regret_reduction(base,new):
    x=float(base['mean_safe_policy_regret']); y=float(new['mean_safe_policy_regret'])
    return 0.0 if x<=1e-12 else (x-y)/x


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development']); ap.add_argument('--seed',type=int,default=None); a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/presure.yaml')
    if str(cfg.get('version')) not in {'0.9.3','0.9.4'}: raise RuntimeError(f'v0.9A.3 requires 0.9.3+, got {cfg.get("version")}')
    seed=int(cfg['experiment']['seed'] if a.seed is None else a.seed); seed_all(seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    art=load_artifacts(cfg,a.stage); cache=cache_dir(cfg,a.stage); tr=CacheSplit(cache,'train'); va=CacheSplit(cache,'val'); mean,std=cache_stats(cache)
    node_mem=np.load(art/'train_node_features.npy',mmap_mode='r'); node_val=np.load(art/'val_node_features.npy',mmap_mode='r'); residual=np.load(art/'train_residual.npy',mmap_mode='r')
    train_base=np.load(art/'train_base_pred.npy',mmap_mode='r'); val_base=np.load(art/'val_base_pred.npy',mmap_mode='r'); train_cand=np.load(art/'train_candidates.npz'); val_cand=np.load(art/'val_candidates.npz')
    ranker,rck,rank_cp,rank_rd,beta,splits=load_v09a(cfg,a.stage,dev); train_ids=splits['train_ids']; dev_ids=splits['dev_ids']; fval_ids=splits['final_val_ids']

    rd=run_dir(cfg,'PredictiveUtilitySelectiveAcceptance',a.stage,stamp(),seed); rd.mkdir(parents=True,exist_ok=True)
    for k,v in splits.items(): np.save(rd/f'{k}.npy',v)
    pc=cfg['predictive_utility_selective_acceptance']; mc=pc['model']; tc=pc['training']
    head=SelectedSupportAcceptanceHead(role_dim=96,context_dim=12,hidden=int(mc['hidden']),dropout=float(mc['dropout'])).to(dev)
    opt=torch.optim.AdamW(head.parameters(),lr=float(tc['lr']),weight_decay=float(tc['weight_decay']))
    best=-1e18; best_ep=0; wait=0; best_path=rd/'selected_support_acceptance_best.pt'

    print('='*60,flush=True)
    print('PreSuRe v0.9A.3 Selective Utility Acceptance',flush=True)
    print(f'Frozen v0.9A ranker={rank_cp} beta={beta:g}',flush=True)
    print(f'train={len(train_ids)} dev={len(dev_ids)} final_val={len(fval_ids)} K={pc["candidate_k"]}',flush=True)
    print('Acceptance is trained ONLY on frozen Top-1 supports. No fallback / no re-ranking.',flush=True)
    print('Label: selected support utility > 0; near-zero utility receives reduced loss weight.',flush=True)
    print('Qwen / Dual Memory remain excluded.',flush=True)
    print('='*60,flush=True)

    for ep in range(1,int(tc['epochs'])+1):
        head.train(); losses=[]; pos=[]; amb=[]
        for src in batches(train_ids,int(tc['batch_size']),True,seed+1009*ep):
            pack=gather_batch(cfg,None,tr,src,train_cand,node_mem,node_mem,residual,train_base,mean,std,dev)
            _,feat,sv,su,_,_,_,_,_=frozen_selected_outputs(ranker,pack,beta)
            logit=head(feat)
            mm=t(view_mask(tr,src,'masked'),dev,torch.bool); sm=t(view_mask(tr,src,'support'),dev,torch.bool)
            loss,parts=selective_acceptance_loss(logit,su,sv,cfg,mm,sm)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),float(tc['grad_clip'])); opt.step()
            losses.append(float(loss.detach().cpu())); pos.append(parts['positive_rate']); amb.append(parts['ambiguous_rate'])

        cls=eval_classifier(ranker,head,cfg,tr,dev_ids,train_cand,node_mem,node_mem,residual,train_base,mean,std,dev,beta,seed+1777*ep)
        grid=policy_grid_eval(ranker,head,cfg,tr,dev_ids,train_cand,node_mem,node_mem,residual,train_base,mean,std,dev,beta,views=('masked',))
        choice=choose_threshold(grid,cfg); score=float(choice['objective'])
        print(f'EPOCH {ep:02d} loss={np.mean(losses):.5f} train_pos={np.mean(pos):.4f} train_amb={np.mean(amb):.4f} '
              f'accept_auc={cls["masked"]["auroc"]:.4f} th={choice["threshold"]:.2f} coverage={choice["coverage"]:.4f} '
              f'positive={choice["accepted_positive_utility_rate"]:.4f} beneficial={choice["accepted_beneficial_rate"]:.4f} '
              f'harmful={choice["accepted_harmful_rate"]:.4f} regret={choice["mean_safe_policy_regret"]:.5f}',flush=True)
        if score>best+1e-6:
            best=score;best_ep=ep;wait=0
            torch.save({'epoch':ep,'state_dict':head.state_dict(),'model_args':{'role_dim':96,'context_dim':12,'hidden':int(mc['hidden']),'dropout':float(mc['dropout'])},
                        'frozen_ranker_checkpoint':str(rank_cp),'frozen_ranker_run':str(rank_rd),'frozen_risk_beta':beta,
                        'dev_threshold':float(choice['threshold']),'dev_policy':choice,'dev_classifier':cls},best_path)
        else:
            wait+=1
            if wait>=int(tc['patience']): print(f'EARLY STOP epoch={ep} best_epoch={best_ep}',flush=True); break

    ck=torch.load(best_path,map_location=dev,weights_only=False); head.load_state_dict(ck['state_dict'])
    # Recalibrate on Train-dev using the selected checkpoint only.
    dev_grid=policy_grid_eval(ranker,head,cfg,tr,dev_ids,train_cand,node_mem,node_mem,residual,train_base,mean,std,dev,beta,views=('masked',))
    choice=choose_threshold(dev_grid,cfg); th=float(choice['threshold'])
    dev_cls=eval_classifier(ranker,head,cfg,tr,dev_ids,train_cand,node_mem,node_mem,residual,train_base,mean,std,dev,beta,seed+991)
    final_cls=eval_classifier(ranker,head,cfg,va,fval_ids,val_cand,node_val,node_mem,residual,val_base,mean,std,dev,beta,seed+1991)
    final=fixed_policy_eval(ranker,head,cfg,va,fval_ids,val_cand,node_val,node_mem,residual,val_base,mean,std,dev,beta,th)
    b=final['frozen_v09a_top1']['masked']; n=final['selective_acceptance']['masked']; rr=regret_reduction(b,n)
    dc=pc['decision']; checks={
        'coverage':n['coverage']>=float(dc['coverage_go']),
        'accepted_beneficial':n['accepted_beneficial_rate']>=float(dc['accepted_beneficial_go']),
        'accepted_harmful':n['accepted_harmful_rate']<=float(dc['accepted_harmful_go']),
        'safe_regret_reduction':rr>=float(dc['safe_regret_reduction_go']),
    }
    decision='GO' if all(checks.values()) else ('CONDITIONAL' if n['coverage']>=float(dc['coverage_conditional']) and n['accepted_harmful_rate']<=float(dc['accepted_harmful_conditional']) and rr>=float(dc['safe_regret_reduction_conditional']) else 'NO_GO')
    summary={'version':cfg['version'],'model_type':'selective_utility_acceptance_v09a3','stage':a.stage,'seed':seed,'run_dir':str(rd),'checkpoint':str(best_path),
             'frozen_ranker_checkpoint':str(rank_cp),'frozen_ranker_run':str(rank_rd),'frozen_risk_beta':beta,'best_epoch':int(ck['epoch']),
             'chosen_acceptance_threshold':th,'ranker_frozen':True,'top1_only':True,'fallback_used':False,'ranking_reordered':False,'qwen_used':False,'dual_memory_used':False,
             'dev_classifier':dev_cls,'final_val_classifier':final_cls,'dev_policy':choice,'final_val_policy':final,
             'masked_safe_regret_reduction_vs_frozen_v09a':rr,'go_checks':checks,'decision':decision}
    write_json(rd/'summary.json',summary)
    with (rd/'policy_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['variant','view','coverage','accepted_positive_utility_rate','accepted_beneficial_rate','accepted_harmful_rate','accepted_strong_harm_rate','beneficial_capture_ratio','mean_policy_relative_utility','mean_accepted_relative_utility','mean_safe_policy_regret'])
        for variant,views in final.items():
            for v,m in views.items(): w.writerow([variant,v,m['coverage'],m['accepted_positive_utility_rate'],m['accepted_beneficial_rate'],m['accepted_harmful_rate'],m['accepted_strong_harm_rate'],m['beneficial_capture_ratio'],m['mean_policy_relative_utility'],m['mean_accepted_relative_utility'],m['mean_safe_policy_regret']])
    with (rd/'threshold_grid_masked.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['threshold','objective','feasible','coverage','accepted_positive_utility_rate','accepted_beneficial_rate','accepted_harmful_rate','mean_policy_relative_utility','mean_safe_policy_regret'])
        for t0,views in sorted(dev_grid.items()):
            x=views['masked']; harm_exp=x['coverage']*x['accepted_harmful_rate']; obj=x['mean_policy_relative_utility']+float(pc['policy']['capture_reward'])*x['beneficial_capture_ratio']-float(pc['policy']['harmful_exposure_penalty'])*harm_exp
            w.writerow([t0,obj,x['coverage']>=float(pc['policy']['min_coverage']),x['coverage'],x['accepted_positive_utility_rate'],x['accepted_beneficial_rate'],x['accepted_harmful_rate'],x['mean_policy_relative_utility'],x['mean_safe_policy_regret']])
    (rd/'DECISION.txt').write_text(f'{decision}\nbest_epoch={ck["epoch"]}\nfrozen_beta={beta:g}\nacceptance_threshold={th:g}\nmasked_coverage={n["coverage"]:.6f}\nmasked_accepted_positive={n["accepted_positive_utility_rate"]:.6f}\nmasked_accepted_beneficial={n["accepted_beneficial_rate"]:.6f}\nmasked_accepted_harmful={n["accepted_harmful_rate"]:.6f}\nmasked_safe_regret_reduction_vs_v09a={rr:.6f}\n')
    latest=Path(cfg['project']['exp_root'])/'PredictiveUtilitySelectiveAcceptance'/cfg['experiment']['subgraph']/a.stage/'LATEST.json'; latest.parent.mkdir(parents=True,exist_ok=True); write_json(latest,{'run_dir':str(rd),'checkpoint':str(best_path),'summary':str(rd/'summary.json'),'variant':'PredictiveUtilitySelectiveAcceptance','stage':a.stage,'version':cfg['version']})
    print('='*60,flush=True); print(f'BEST epoch={ck["epoch"]} acceptance_threshold={th:g}',flush=True); print('FINAL acceptance classifier masked:',json.dumps(final_cls['masked'],indent=2),flush=True); print('FINAL masked frozen v0.9A:',json.dumps(b,indent=2),flush=True); print('FINAL masked selective acceptance:',json.dumps(n,indent=2),flush=True); print(f'MASKED safe-regret reduction vs frozen v0.9A = {rr:.4%}',flush=True); print(f'DECISION = {decision} checks={checks}',flush=True); print(f'SUMMARY {rd/"summary.json"}',flush=True); print('='*60,flush=True)

if __name__=='__main__': main()
