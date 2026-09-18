#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import sys

import numpy as np
import torch

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe')
ROOT = ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from presure.utils import load_yaml, write_json, stamp, seed_all
from presure.contract import cache_dir
from presure.data import CacheSplit, cache_stats
from presure.features import NODE_SENSE_DIM
from presure.models import PredictiveUtilityRerankerV09A
from presure.need_gated import gather_support_context
from presure.pipeline import load_artifacts, load_m1, run_dir
from presure.predictive_utility import (
    SafePolicyAccumulator,
    UtilitySelectionAccumulator,
    calibrate_abstain_threshold,
    node_horizon_utility_targets,
    normalize_coarse_scores,
    predictive_utility_safe_loss,
    safe_selection_score,
)


def batches(ids, bs, shuffle=False, seed=0):
    x = np.asarray(ids, np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(x)
    for a in range(0, len(x), int(bs)):
        yield x[a:a + int(bs)]


def t(a, dev, dtype=torch.float32):
    return torch.as_tensor(np.array(a, copy=True), device=dev, dtype=dtype)


def view_mask(ds, src, view):
    valid = np.asarray(ds.valid[src], bool)
    if view == 'overall':
        return np.transpose(valid, (0, 2, 1))
    if view == 'masked':
        m = valid & np.asarray(ds.direct[src], bool)[:, None, :]
        return np.transpose(m, (0, 2, 1))
    if view == 'support':
        m = valid & np.asarray(ds.support[src], bool)[:, None, :]
        return np.transpose(m, (0, 2, 1))
    raise ValueError(view)


def split_ids(n, train_n, dev_n, seed):
    ids = np.arange(n, dtype=np.int64)
    np.random.default_rng(seed).shuffle(ids)
    if int(train_n) + int(dev_n) > len(ids):
        raise RuntimeError(f'Need {train_n + dev_n} Train queries, only {len(ids)} available')
    return ids[:int(train_n)], ids[int(train_n):int(train_n) + int(dev_n)]


def final_val_ids(n, count, seed):
    ids = np.arange(n, dtype=np.int64)
    np.random.default_rng(seed + 911).shuffle(ids)
    return ids[:min(int(count), len(ids))]


def model_args(cfg, node_feature_dim, num_nodes, P):
    c = cfg['predictive_utility_safe_selection']['model']
    return {
        'node_feature_dim': int(node_feature_dim),
        'num_nodes': int(num_nodes),
        'P': int(P),
        'encoder_hidden': int(c['encoder_hidden']),
        'state_dim': int(c['state_dim']),
        'node_dim': int(c['node_dim']),
        'sense_dim': int(NODE_SENSE_DIM),
        'residual_dim': int(c['residual_dim']),
        'pair_dim': int(c['pair_dim']),
        'hidden': int(c['hidden']),
        'horizon_dim': int(c['horizon_dim']),
        'attention_heads': int(c['attention_heads']),
        'dropout': float(c['dropout']),
        'min_sigma': float(c['min_sigma']),
        'max_sigma': float(c['max_sigma']),
    }


def gather_batch(cfg, ds, src, cand, node_q, node_mem, residual_mem, base_raw, mean, std, dev):
    pc = cfg['predictive_utility_safe_selection']
    H = int(cfg['experiment']['history'])
    P = int(cfg['experiment']['horizon'])
    K = int(pc['candidate_k'])
    gids_raw = np.asarray(cand['indices'][src, :K], np.int64)
    scores_raw = np.asarray(cand['scores'][src, :K], np.float32)
    query_mask = gids_raw >= 0
    gids = np.where(query_mask, gids_raw, 0)

    qctx_np = np.asarray(node_q[src], np.float32)
    gids_node = np.broadcast_to(
        gids[:, None, :],
        (gids.shape[0], qctx_np.shape[1], gids.shape[1]),
    )
    cctx_np, rr_raw_np = gather_support_context(node_mem, residual_mem, gids_node)
    cm_np = np.broadcast_to(
        query_mask[:, None, :],
        (query_mask.shape[0], qctx_np.shape[1], query_mask.shape[1]),
    ).copy()
    coarse_np = normalize_coarse_scores(scores_raw)

    qctx = t(qctx_np, dev)
    cctx = t(cctx_np, dev)
    rr = t(rr_raw_np / float(std), dev)
    qs = qctx[..., -NODE_SENSE_DIM:]
    ss = cctx[..., -NODE_SENSE_DIM:]
    b = t(np.asarray(base_raw[src], np.float32), dev)
    y = t(np.asarray(ds.y[src], np.float32), dev)
    qbase = (b - float(mean)) / float(std)
    sbase = cctx[..., 4 * H:4 * H + P]
    cm = t(cm_np, dev, torch.bool)
    coarse = t(coarse_np, dev)
    valid = t(np.transpose(np.asarray(ds.valid[src], bool), (0, 2, 1)), dev, torch.bool)
    true_need = (y - b) / float(std)
    target, best_alpha = node_horizon_utility_targets(
        true_need, rr, cm,
        pc['training']['alpha_grid'],
        pc['training']['gain_floor'],
        pc['training']['utility_clip'],
    )
    return qctx, cctx, qs, ss, rr, qbase, sbase, coarse, cm, valid, target, best_alpha


def combo_name(beta, lam):
    return f'combo_b{float(beta):g}_l{float(lam):g}'


def evaluate_candidates(model, cfg, ds, ids, cand, node_q, node_mem, residual_mem,
                        base_raw, mean, std, dev, seed):
    pc = cfg['predictive_utility_safe_selection']
    tc = pc['training']
    betas = [float(x) for x in pc['risk']['beta_grid']]
    lams = [float(x) for x in pc['risk']['benefit_logit_weight_grid']]
    names = ['coarse', 'mu', 'benefit'] + [combo_name(b, l) for b in betas for l in lams]
    acc = {
        name: {
            view: UtilitySelectionAccumulator(
                float(tc['beneficial_threshold']), float(tc['harmful_threshold']),
                sample_cap_per_update=256,
                seed=int(seed + 1000 * ni + 100 * vi),
            )
            for vi, view in enumerate(['overall', 'masked', 'support'])
        }
        for ni, name in enumerate(names)
    }
    sigma_sum = sigma_n = 0
    model.eval()
    with torch.no_grad():
        for src in batches(ids, int(tc['eval_batch_size'])):
            pack = gather_batch(cfg, ds, src, cand, node_q, node_mem, residual_mem, base_raw, mean, std, dev)
            qctx, cctx, qs, ss, rr, qbase, sbase, coarse, cm, valid, target, _ = pack
            mu, sigma, blogit = model(qctx, cctx, qs, ss, rr, qbase, sbase, coarse)
            sigma_sum += float(sigma[cm[:, :, :, None].expand_as(sigma)].sum().cpu())
            sigma_n += int(cm.sum().cpu()) * int(mu.shape[-1])
            B, N, K, P = mu.shape
            coarse_score = coarse[:, None, :, None].expand(B, N, K, P)
            score_map = {
                'coarse': coarse_score,
                'mu': mu,
                'benefit': blogit,
            }
            for beta in betas:
                for lam in lams:
                    score_map[combo_name(beta, lam)] = safe_selection_score(mu, sigma, blogit, beta, lam)
            for view in ['overall', 'masked', 'support']:
                vm = t(view_mask(ds, src, view), dev, torch.bool)
                for name, score in score_map.items():
                    acc[name][view].update(score, target, cm, vm)
    out = {name: {view: obj.summary() for view, obj in views.items()} for name, views in acc.items()}
    out['_mean_sigma'] = float(sigma_sum / max(sigma_n, 1))
    return out


def choose_combo(summary, cfg):
    r = cfg['predictive_utility_safe_selection']['risk']
    best = None
    for beta in [float(x) for x in r['beta_grid']]:
        for lam in [float(x) for x in r['benefit_logit_weight_grid']]:
            key = combo_name(beta, lam)
            m = summary[key]['masked']
            score = (
                float(r['checkpoint_rank_weight']) * float(m['pairwise_rank_accuracy'])
                + float(r['checkpoint_benefit_weight']) * float(m['selected_beneficial_rate'])
                - float(r['checkpoint_harm_weight']) * float(m['selected_harmful_rate'])
                - float(r['checkpoint_strong_harm_weight']) * float(m['selected_strong_harm_rate'])
                - float(r['checkpoint_regret_weight']) * float(m['mean_selection_regret'])
            )
            item = (score, -float(m['selected_harmful_rate']), -float(m['mean_selection_regret']), -beta, -lam)
            if best is None or item > best[0]:
                best = (item, beta, lam, key, score)
    return float(best[1]), float(best[2]), best[3], float(best[4])


def collect_top_records(model, cfg, ds, ids, cand, node_q, node_mem, residual_mem,
                        base_raw, mean, std, dev, beta, lam, view='masked'):
    tc = cfg['predictive_utility_safe_selection']['training']
    score_rows, chosen_rows, best_rows = [], [], []
    model.eval()
    with torch.no_grad():
        for src in batches(ids, int(tc['eval_batch_size'])):
            pack = gather_batch(cfg, ds, src, cand, node_q, node_mem, residual_mem, base_raw, mean, std, dev)
            qctx, cctx, qs, ss, rr, qbase, sbase, coarse, cm, valid, target, _ = pack
            mu, sigma, blogit = model(qctx, cctx, qs, ss, rr, qbase, sbase, coarse)
            score = safe_selection_score(mu, sigma, blogit, beta, lam)
            vm = t(view_mask(ds, src, view), dev, torch.bool)
            use = cm[:, :, :, None] & vm[:, :, None, :]
            ms = score.masked_fill(~use, -1e6)
            mt = target.masked_fill(~use, -1e6)
            pi = ms.argmax(2)
            top_score = ms.max(2).values
            chosen = torch.gather(target, 2, pi[:, :, None, :]).squeeze(2)
            best = mt.max(2).values
            ok = (use.sum(2) > 0) & vm
            if torch.any(ok):
                score_rows.append(top_score[ok].detach().cpu().numpy().astype(np.float32, copy=False))
                chosen_rows.append(chosen[ok].detach().cpu().numpy().astype(np.float32, copy=False))
                best_rows.append(best[ok].detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(score_rows), np.concatenate(chosen_rows), np.concatenate(best_rows)


def evaluate_fixed_policy(model, cfg, ds, ids, cand, node_q, node_mem, residual_mem,
                          base_raw, mean, std, dev, beta, lam, threshold):
    tc = cfg['predictive_utility_safe_selection']['training']
    bthr = float(tc['beneficial_threshold']); hthr = float(tc['harmful_threshold'])
    policy = {v: SafePolicyAccumulator(bthr, hthr, float(threshold)) for v in ['overall','masked','support']}
    coarse = {v: SafePolicyAccumulator(bthr, hthr, -1e9) for v in ['overall','masked','support']}
    noabstain = {v: SafePolicyAccumulator(bthr, hthr, -1e9) for v in ['overall','masked','support']}
    model.eval()
    with torch.no_grad():
        for src in batches(ids, int(tc['eval_batch_size'])):
            pack = gather_batch(cfg, ds, src, cand, node_q, node_mem, residual_mem, base_raw, mean, std, dev)
            qctx, cctx, qs, ss, rr, qbase, sbase, coarse_s, cm, valid, target, _ = pack
            mu, sigma, blogit = model(qctx, cctx, qs, ss, rr, qbase, sbase, coarse_s)
            score = safe_selection_score(mu, sigma, blogit, beta, lam)
            B,N,K,P = score.shape
            coarse_score = coarse_s[:, None, :, None].expand(B,N,K,P)
            for view in ['overall','masked','support']:
                vm = t(view_mask(ds, src, view), dev, torch.bool)
                policy[view].update(score, target, cm, vm)
                noabstain[view].update(score, target, cm, vm)
                coarse[view].update(coarse_score, target, cm, vm)
    return {
        'safe_policy': {v: x.summary() for v,x in policy.items()},
        'no_abstain': {v: x.summary() for v,x in noabstain.items()},
        'coarse_no_abstain': {v: x.summary() for v,x in coarse.items()},
    }


def regret_reduction(base_summary, new_summary):
    b = float(base_summary['mean_safe_policy_regret'])
    n = float(new_summary['mean_safe_policy_regret'])
    return 0.0 if b <= 1e-12 else float((b - n) / b)


def warm_start(model, cfg, stage, dev):
    pc = cfg['predictive_utility_safe_selection']
    if bool(pc.get('initialize_from_v09a', True)):
        lp = Path(cfg['project']['exp_root']) / 'PredictiveUtilityRetrieval' / cfg['experiment']['subgraph'] / stage / 'LATEST.json'
        if lp.exists():
            j = json.loads(lp.read_text())
            cp = Path(j['checkpoint'])
            if cp.exists():
                ck = torch.load(cp, map_location=dev, weights_only=False)
                model.load_state_dict(ck['state_dict'], strict=True)
                print(f'INIT full reranker from v0.9A {cp}', flush=True)
                return str(cp), 'v0.9A_full_reranker'
        if bool(pc.get('require_v09a_checkpoint', True)):
            raise RuntimeError(f'v0.9A warm-start required but no usable LATEST/checkpoint at {lp}')
    m1, _, m1path = load_m1(cfg, stage, dev)
    if set(m1.encoder.state_dict()) != set(model.encoder.state_dict()):
        raise RuntimeError('M1 encoder keys differ from safe-selection reranker')
    model.encoder.load_state_dict(m1.encoder.state_dict())
    print(f'INIT encoder from M1 fallback {m1path}', flush=True)
    return str(m1path), 'M1_encoder_only'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='development', choices=['development'])
    ap.add_argument('--seed', type=int, default=None)
    a = ap.parse_args()

    cfg = load_yaml(ROOT / 'configs/presure.yaml')
    if str(cfg.get('version')) not in {'0.9.1','0.9.2','0.9.3','0.9.4','0.9.5'}:
        raise RuntimeError(f'v0.9A.1 runner requires config version 0.9.1/0.9.2, got {cfg.get("version")}')
    seed = int(cfg['experiment']['seed'] if a.seed is None else a.seed)
    seed_all(seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    art = load_artifacts(cfg, a.stage)
    cache = cache_dir(cfg, a.stage)
    tr = CacheSplit(cache, 'train'); va = CacheSplit(cache, 'val')
    mean, std = cache_stats(cache)

    node_mem = np.load(art / 'train_node_features.npy', mmap_mode='r')
    node_val = np.load(art / 'val_node_features.npy', mmap_mode='r')
    residual_mem = np.load(art / 'train_residual.npy', mmap_mode='r')
    train_base = np.load(art / 'train_base_pred.npy', mmap_mode='r')
    val_base = np.load(art / 'val_base_pred.npy', mmap_mode='r')
    train_cand = np.load(art / 'train_candidates.npz'); val_cand = np.load(art / 'val_candidates.npz')

    sc = cfg['predictive_utility_safe_selection']['split']
    train_ids, dev_ids = split_ids(len(tr), sc['train_queries'], sc['dev_queries'], seed + 101)
    fval_ids = final_val_ids(len(va), sc['final_val_queries'], seed + 303)

    rd = run_dir(cfg, 'PredictiveUtilitySafeSelection', a.stage, stamp(), seed)
    rd.mkdir(parents=True, exist_ok=True)
    np.save(rd / 'train_ids.npy', train_ids); np.save(rd / 'dev_ids.npy', dev_ids); np.save(rd / 'final_val_ids.npy', fval_ids)

    args = model_args(cfg, node_mem.shape[-1], node_mem.shape[1], int(cfg['experiment']['horizon']))
    model = PredictiveUtilityRerankerV09A(**args).to(dev)
    initialized_from, init_mode = warm_start(model, cfg, a.stage, dev)

    tc = cfg['predictive_utility_safe_selection']['training']
    enc_params = list(model.encoder.parameters()); enc_ids = {id(p) for p in enc_params}
    other_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {'params': enc_params, 'lr': float(tc['lr']) * float(tc['encoder_lr_scale'])},
        {'params': other_params, 'lr': float(tc['lr'])},
    ], weight_decay=float(tc['weight_decay']))

    best_score = -1e18; best_epoch = 0; wait = 0
    best_path = rd / 'predictive_utility_safe_selection_best.pt'

    print('============================================================', flush=True)
    print('PreSuRe v0.9A.1 Safe Top-Selection', flush=True)
    print(f'device={dev} train={len(train_ids)} dev={len(dev_ids)} final_val={len(fval_ids)} K={cfg["predictive_utility_safe_selection"]["candidate_k"]}', flush=True)
    print('Warm-start preserves v0.9A ranking; new objectives target Top-1, sign separation, and abstention.', flush=True)
    print('Query future remains supervision/evaluation only.', flush=True)
    print('============================================================', flush=True)

    for ep in range(1, int(tc['epochs']) + 1):
        model.train(); losses=[]; ranks=[]; hards=[]; tops=[]
        for src in batches(train_ids, int(tc['batch_size']), True, seed + ep * 1009):
            pack = gather_batch(cfg, tr, src, train_cand, node_mem, node_mem, residual_mem, train_base, mean, std, dev)
            qctx,cctx,qs,ss,rr,qbase,sbase,coarse_s,cm,valid,target,_ = pack
            mu,sigma,blogit = model(qctx,cctx,qs,ss,rr,qbase,sbase,coarse_s)
            loss, parts = predictive_utility_safe_loss(mu,sigma,blogit,target,cm,valid,coarse_s,cfg)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc['grad_clip'])); opt.step()
            losses.append(float(loss.detach().cpu()))
            if parts['rank_n']: ranks.append(parts['rank_acc'])
            if parts['hard_n']: hards.append(parts['hard_acc'])
            if parts['top1_n']: tops.append(parts['top1_acc'])

        dev_summary = evaluate_candidates(model,cfg,tr,dev_ids,train_cand,node_mem,node_mem,residual_mem,train_base,mean,std,dev,seed+ep*1777)
        beta, lam, key, score = choose_combo(dev_summary,cfg)
        mm = dev_summary[key]['masked']
        print(
            f'EPOCH {ep:02d} loss={np.mean(losses):.5f} '
            f'train_rank={np.mean(ranks) if ranks else 0.0:.4f} '
            f'train_hard={np.mean(hards) if hards else 0.0:.4f} '
            f'train_top1={np.mean(tops) if tops else 0.0:.4f} '
            f'dev_beta={beta:g} dev_lambda={lam:g} rank={mm["pairwise_rank_accuracy"]:.4f} '
            f'top1={mm["top1_agreement"]:.4f} beneficial={mm["selected_beneficial_rate"]:.4f} '
            f'harmful={mm["selected_harmful_rate"]:.4f} strong_harm={mm["selected_strong_harm_rate"]:.4f} '
            f'regret={mm["mean_selection_regret"]:.5f}', flush=True,
        )
        if score > best_score:
            best_score, best_epoch, wait = score, ep, 0
            torch.save({
                'model_type':'predictive_utility_safe_selection_v09a1',
                'state_dict':model.state_dict(), 'model_args':args, 'epoch':ep, 'dev_score':score,
                'risk_beta':beta, 'benefit_logit_weight':lam,
                'initialized_from':initialized_from, 'initialization_mode':init_mode,
                'candidate_k':int(cfg['predictive_utility_safe_selection']['candidate_k']),
                'target':'node_horizon_best_alpha_relative_predictive_utility',
                'future_query_target_is_model_input':False,
            }, best_path)
        else:
            wait += 1
        if wait >= int(tc['patience']):
            print(f'EARLY STOP epoch={ep} best_epoch={best_epoch}', flush=True); break

    ck = torch.load(best_path,map_location=dev,weights_only=False)
    model.load_state_dict(ck['state_dict']); model.eval()
    beta=float(ck['risk_beta']); lam=float(ck['benefit_logit_weight']); key=combo_name(beta,lam)

    dev_summary = evaluate_candidates(model,cfg,tr,dev_ids,train_cand,node_mem,node_mem,residual_mem,train_base,mean,std,dev,seed+5551)
    val_summary = evaluate_candidates(model,cfg,va,fval_ids,val_cand,node_val,node_mem,residual_mem,val_base,mean,std,dev,seed+7771)

    top_s, chosen_u, best_u = collect_top_records(
        model,cfg,tr,dev_ids,train_cand,node_mem,node_mem,residual_mem,train_base,mean,std,dev,beta,lam,'masked')
    policy_choice, policy_grid = calibrate_abstain_threshold(top_s,chosen_u,best_u,cfg)
    threshold=float(policy_choice['threshold'])
    final_policy = evaluate_fixed_policy(
        model,cfg,va,fval_ids,val_cand,node_val,node_mem,residual_mem,val_base,mean,std,dev,beta,lam,threshold)

    coarse_m = final_policy['coarse_no_abstain']['masked']
    noabs_m = final_policy['no_abstain']['masked']
    safe_m = final_policy['safe_policy']['masked']
    rr = regret_reduction(coarse_m,safe_m)
    dc = cfg['predictive_utility_safe_selection']['decision']
    noabs_rank = float(val_summary[key]['masked']['pairwise_rank_accuracy'])
    go_checks = {
        'masked_pairwise_rank': noabs_rank >= float(dc['masked_pairwise_rank_go']),
        'coverage': float(safe_m['coverage']) >= float(dc['coverage_go']),
        'accepted_beneficial': float(safe_m['accepted_beneficial_rate']) >= float(dc['accepted_beneficial_go']),
        'accepted_harmful': float(safe_m['accepted_harmful_rate']) <= float(dc['accepted_harmful_go']),
        'safe_regret_reduction': rr >= float(dc['safe_regret_reduction_go']),
    }
    conditional = (
        float(safe_m['coverage']) >= float(dc['coverage_conditional'])
        and float(safe_m['accepted_harmful_rate']) <= float(dc['accepted_harmful_conditional'])
        and rr >= float(dc['safe_regret_reduction_conditional'])
    )
    decision='GO' if all(go_checks.values()) else ('CONDITIONAL' if conditional else 'NO_GO')

    summary = {
        'version':cfg['version'], 'implementation_revision':cfg.get('implementation_revision'),
        'model_type':'predictive_utility_safe_selection_v09a1', 'stage':a.stage, 'seed':seed,
        'run_dir':str(rd), 'checkpoint':str(best_path), 'best_epoch':int(ck['epoch']),
        'chosen_risk_beta':beta, 'chosen_benefit_logit_weight':lam,
        'chosen_abstain_threshold':threshold, 'candidate_k':int(cfg['predictive_utility_safe_selection']['candidate_k']),
        'initialization':{'source':initialized_from,'mode':init_mode},
        'legacy_m3_top4_used':False, 'qwen_used':False, 'dual_memory_used':False,
        'query_future_used_as_model_input':False,
        'dev_candidate_selection':dev_summary,
        'final_val_candidate_selection':val_summary,
        'dev_policy_calibration':{'chosen':policy_choice,'grid':policy_grid},
        'final_val_policy':final_policy,
        'masked_safe_regret_reduction_vs_coarse':rr,
        'go_checks':go_checks, 'decision':decision,
    }
    write_json(rd/'summary.json',summary)

    with (rd/'comparison.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['variant','view','pairwise_rank_accuracy','top1_agreement','selected_beneficial_rate','selected_harmful_rate','selected_strong_harm_rate','mean_selection_regret','mean_selected_relative_utility'])
        for variant in ['coarse','mu','benefit',key]:
            for view in ['overall','masked','support']:
                m=val_summary[variant][view]
                w.writerow([variant,view,m['pairwise_rank_accuracy'],m['top1_agreement'],m['selected_beneficial_rate'],m['selected_harmful_rate'],m['selected_strong_harm_rate'],m['mean_selection_regret'],m['mean_selected_relative_utility']])
    with (rd/'policy_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['variant','view','coverage','accepted_beneficial_rate','accepted_harmful_rate','accepted_strong_harm_rate','beneficial_capture_ratio','mean_policy_relative_utility','mean_accepted_relative_utility','mean_safe_policy_regret'])
        for variant in ['coarse_no_abstain','no_abstain','safe_policy']:
            for view in ['overall','masked','support']:
                m=final_policy[variant][view]
                w.writerow([variant,view,m['coverage'],m['accepted_beneficial_rate'],m['accepted_harmful_rate'],m['accepted_strong_harm_rate'],m['beneficial_capture_ratio'],m['mean_policy_relative_utility'],m['mean_accepted_relative_utility'],m['mean_safe_policy_regret']])

    (rd/'DECISION.txt').write_text(
        f'{decision}\n'
        f'best_epoch={ck["epoch"]}\n'
        f'beta={beta:g}\nbenefit_logit_weight={lam:g}\nabstain_threshold={threshold:.8f}\n'
        f'masked_pairwise_rank={noabs_rank:.6f}\n'
        f'masked_coverage={safe_m["coverage"]:.6f}\n'
        f'masked_accepted_beneficial={safe_m["accepted_beneficial_rate"]:.6f}\n'
        f'masked_accepted_harmful={safe_m["accepted_harmful_rate"]:.6f}\n'
        f'masked_accepted_strong_harm={safe_m["accepted_strong_harm_rate"]:.6f}\n'
        f'masked_safe_regret_reduction_vs_coarse={rr:.6f}\n'
    )
    latest=Path(cfg['project']['exp_root'])/'PredictiveUtilitySafeSelection'/cfg['experiment']['subgraph']/a.stage/'LATEST.json'
    latest.parent.mkdir(parents=True,exist_ok=True)
    write_json(latest,{'run_dir':str(rd),'checkpoint':str(best_path),'summary':str(rd/'summary.json'),'variant':'PredictiveUtilitySafeSelection','stage':a.stage,'version':cfg['version']})

    print('============================================================',flush=True)
    print(f'BEST epoch={ck["epoch"]} beta={beta:g} lambda={lam:g}',flush=True)
    print('DEV abstain calibration:',json.dumps(policy_choice,indent=2),flush=True)
    print('FINAL masked coarse policy:',json.dumps(coarse_m,indent=2),flush=True)
    print('FINAL masked v0.9A.1 no-abstain:',json.dumps(noabs_m,indent=2),flush=True)
    print('FINAL masked v0.9A.1 safe policy:',json.dumps(safe_m,indent=2),flush=True)
    print(f'MASKED safe-regret reduction vs coarse = {rr:.4%}',flush=True)
    print(f'DECISION = {decision} checks={go_checks}',flush=True)
    print(f'SUMMARY {rd / "summary.json"}',flush=True)
    print('============================================================',flush=True)


if __name__ == '__main__':
    main()
