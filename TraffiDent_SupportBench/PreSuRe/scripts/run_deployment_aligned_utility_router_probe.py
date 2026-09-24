#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import copy
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
from presure.data import CacheSplit
from presure.need_gated import instantiate_utility_a, load_latest
from presure.probes import SupportUtilityProbeNet
from presure.utility_specialist import CandidateMetricAccumulator, true_q90_mask
from presure.deployment_routing import (
    ExpertPreferenceRouter,
    RouterAccuracyAccumulator,
    deployment_specialist_loss,
    need_thresholds_by_horizon,
    predicted_need_mask,
    router_features,
    router_targets,
    router_loss,
    route_candidate_scores,
    oracle_expert_scores,
)
from presure.pipeline import load_artifacts, run_dir
from scripts.run_need_conditioned_utility_specialist_probe import (
    batches, t, load_state_q90, predict_need_q90, load_top4, load_state_top4,
    view_np, specialist_model_from_utility_a, batch_scores,
)


EXPERT_ORDER = ['generic', 'hard', 'pred10', 'pred20']


def load_v074_hard(v074_rd, dev):
    p = v074_rd / 'hard_specialist_best.pt'
    ck = torch.load(p, map_location=dev, weights_only=False)
    if ck.get('model_type') != 'need_conditioned_utility_specialist_v074' or ck.get('mode') != 'hard':
        raise RuntimeError(f'v0.7.4 hard checkpoint contract mismatch: {p}')
    m = SupportUtilityProbeNet(**ck['model_args']).to(dev)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m, ck


def fixed_subset_masks(ds, ids, need_prob, true_hard, thresholds):
    out = {}
    pm10 = predicted_need_mask(need_prob, thresholds['pred10'])
    pm20 = predicted_need_mask(need_prob, thresholds['pred20'])
    for view in ['overall', 'masked', 'support']:
        vm = view_np(ds, ids, view)
        out[view] = {
            'all': vm,
            'pred_top10': vm & pm10,
            'pred_top20': vm & pm20,
            'true_q90': vm & true_hard,
        }
    return out


def eval_one_specialist(model, cfg, art, ds, ids, top4, need_prob, deployment_threshold, dev, seed):
    c = cfg['deployment_aligned_utility_router_probe']
    uc = cfg['predictive_need_probe']['utility_model']
    bs = int(c['evaluation']['eval_batch_size'])
    acc = CandidateMetricAccumulator(float(uc['beneficial_threshold']),
                                     int(c['evaluation']['sample_points_per_batch']), seed)
    for m in [model]:
        m.eval()
    with torch.no_grad():
        for bi in batches(np.arange(len(ids)), bs):
            src = ids[bi]
            pred, target, cm, valid, _ = batch_scores(model, cfg, art, ds, src, top4, bi, need_prob, dev, True)
            dep = predicted_need_mask(np.asarray(need_prob[bi]), deployment_threshold,
                                      np.asarray(ds.valid[src], bool))
            cell = torch.as_tensor(np.transpose(dep, (0, 2, 1)), device=dev, dtype=torch.bool)
            acc.update(pred, target, cm, cell)
    return acc.summary()


def train_deployment_specialist(name, keep_fraction, thresholds, cfg, art, tr,
                                ids_train, ids_dev, top_train, top_dev,
                                need_train, need_dev, utility_ck, rd, dev, seed):
    c = cfg['deployment_aligned_utility_router_probe']['specialist_training']
    model = specialist_model_from_utility_a(utility_ck, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c['lr']), weight_decay=float(c['weight_decay']))
    best_score, wait = -1e9, 0
    best_path = rd / f'{name}_specialist_best.pt'
    threshold = thresholds[name]

    for ep in range(1, int(c['epochs']) + 1):
        model.train()
        losses, ranks = [], []
        for bi in batches(np.arange(len(ids_train)), int(c['batch_size']), True, seed + ep * 101):
            src = ids_train[bi]
            pred, target, cm, valid, _ = batch_scores(model, cfg, art, tr, src, top_train, bi, need_train, dev, True)
            dep_np = predicted_need_mask(np.asarray(need_train[bi]), threshold,
                                         np.asarray(tr.valid[src], bool))
            dep = t(dep_np, dev, torch.bool)
            loss, rank, cls, racc, rn = deployment_specialist_loss(pred, target, cm, valid, dep, cfg)
            if rn == 0:
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(c['grad_clip']))
            opt.step()
            losses.append(float(loss.detach())); ranks.append(float(racc.detach()))

        devm = eval_one_specialist(model, cfg, art, tr, ids_dev, top_dev, need_dev,
                                   threshold, dev, seed + 9000 + ep)
        rank = float(devm['pairwise_rank_accuracy'])
        harm = float(devm['selected_harmful_rate'])
        benef = float(devm['selected_beneficial_rate'])
        score = rank + 0.20 * benef - 0.20 * harm
        print(
            f'DEPLOY-SPECIALIST-{name.upper()} epoch={ep:03d} '
            f'loss={np.mean(losses) if losses else 0.0:.5f} '
            f'train_rank={np.mean(ranks) if ranks else 0.0:.3f} '
            f'dev_rank={rank:.3f} dev_benef={benef:.3f} dev_harm={harm:.3f}', flush=True,
        )
        if score > best_score:
            best_score, wait = score, 0
            torch.save({
                'state_dict': model.state_dict(), 'model_args': utility_ck['model_args'],
                'model_type': 'deployment_aligned_utility_specialist_v075',
                'mode': name, 'keep_fraction': float(keep_fraction),
                'need_thresholds_by_horizon': np.asarray(threshold, np.float32).tolist(),
                'epoch': ep, 'dev_score': score,
                'initialized_from': 'v0.7.1 Utility-A',
                'deployment_distribution': f'Train predicted Need top {int(100*float(keep_fraction))}%',
            }, best_path)
        else:
            wait += 1
        if wait >= int(c['patience']):
            break

    ck = torch.load(best_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck['state_dict']); model.eval()
    return model, ck


def get_expert_scores(models, cfg, art, ds, src, top4, bi, need_prob, dev):
    scores = {}
    target = cm = valid = None
    with torch.no_grad():
        for name in EXPERT_ORDER:
            cond = name != 'generic'
            pred, tar, cmm, vv, _ = batch_scores(models[name], cfg, art, ds, src, top4, bi, need_prob, dev, cond)
            scores[name] = pred
            if target is None:
                target, cm, valid = tar, cmm, vv
    return scores, target, cm, valid


def train_router(models, cfg, art, tr, ids_dev, top_dev, need_dev, rd, dev, seed):
    c = cfg['deployment_aligned_utility_router_probe']['router']
    rng = np.random.default_rng(seed + 3001)
    order = rng.permutation(len(ids_dev))
    ntrain = max(1, int(round(len(order) * float(c['train_fraction']))))
    rid_train, rid_hold = order[:ntrain], order[ntrain:]
    if len(rid_hold) == 0:
        rid_hold = order[-max(1, len(order)//5):]
        rid_train = order[:max(1, len(order)-len(rid_hold))]

    router = ExpertPreferenceRouter(len(EXPERT_ORDER), int(c['hidden']), float(c['dropout'])).to(dev)
    opt = torch.optim.AdamW(router.parameters(), lr=float(c['lr']), weight_decay=float(c['weight_decay']))
    temp = float(cfg['predictive_need_probe']['utility_model']['classification_temperature'])
    best, wait = -1e9, 0
    best_path = rd / 'expert_preference_router_best.pt'

    def one_pass(local_ids, train):
        if train: router.train()
        else: router.eval()
        losses, accs, counts = [], [], []
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for pos in batches(local_ids, int(c['batch_size']), shuffle=train, seed=seed + (19 if train else 23)):
                src = ids_dev[pos]
                scores, target, cm, valid = get_expert_scores(models, cfg, art, tr, src, top_dev, pos, need_dev, dev)
                need_t = t(np.asarray(need_dev[pos], np.float32), dev)
                conf_t = t(np.asarray(top_dev['conf'][pos], np.float32), dev)
                feat, _ = router_features(scores, EXPERT_ORDER, need_t, conf_t, cm, temp)
                labels, use, _ = router_targets(scores, EXPERT_ORDER, target, cm, valid, float(c['target_min_gap']))
                logits = router(feat)
                loss, acc, n = router_loss(logits, labels, use, need_t, float(c['need_weight_lambda']))
                if n == 0: continue
                if train:
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(router.parameters(), float(c['grad_clip']))
                    opt.step()
                losses.append(float(loss.detach())); accs.append(float(acc.detach())); counts.append(n)
        return (float(np.mean(losses)) if losses else 0.0,
                float(np.average(accs, weights=counts)) if counts else 0.0,
                int(np.sum(counts)))

    for ep in range(1, int(c['epochs']) + 1):
        tl, ta, tn = one_pass(rid_train, True)
        dl, da, dn = one_pass(rid_hold, False)
        print(f'EXPERT-ROUTER epoch={ep:03d} train_loss={tl:.5f} train_acc={ta:.3f} '
              f'dev_loss={dl:.5f} dev_acc={da:.3f} dev_n={dn}', flush=True)
        if da > best:
            best, wait = da, 0
            torch.save({
                'state_dict': router.state_dict(),
                'model_type': 'expert_preference_router_v075',
                'expert_order': EXPERT_ORDER,
                'model_args': {'n_experts': len(EXPERT_ORDER), 'hidden': int(c['hidden']), 'dropout': float(c['dropout'])},
                'epoch': ep, 'dev_expert_accuracy': da,
                'router_train_query_count': int(len(rid_train)),
                'router_holdout_query_count': int(len(rid_hold)),
                'future_usage': 'Train-only utility_dev expert-preference labels',
            }, best_path)
        else:
            wait += 1
        if wait >= int(c['patience']): break

    ck = torch.load(best_path, map_location=dev, weights_only=False)
    router.load_state_dict(ck['state_dict']); router.eval()
    return router, ck, {'router_train_local_ids': rid_train.tolist(), 'router_holdout_local_ids': rid_hold.tolist()}


def evaluate_all(models, router, cfg, art, ds, ids, top4, need_prob, q90_raw, thresholds, dev, seed):
    c = cfg['deployment_aligned_utility_router_probe']
    uc = cfg['predictive_need_probe']['utility_model']
    temp = float(uc['classification_temperature'])
    bs = int(c['evaluation']['eval_batch_size'])
    base = np.load(art / ('val_base_pred.npy' if ds.split == 'val' else 'train_base_pred.npy'), mmap_mode='r')
    true_hard = true_q90_mask(np.asarray(base[ids]), np.asarray(ds.y[ids]), q90_raw)
    masks = fixed_subset_masks(ds, ids, need_prob, true_hard, thresholds)
    variants = EXPERT_ORDER + ['router', 'router_soft', 'oracle_router']
    acc = {v: {w: {s: CandidateMetricAccumulator(float(uc['beneficial_threshold']),
                                                    int(c['evaluation']['sample_points_per_batch']),
                                                    seed + 1000*i + 100*j + 7*k)
                       for k, s in enumerate(masks[w])}
               for j, w in enumerate(['overall','masked','support'])}
           for i, v in enumerate(variants)}
    racc = {w: {s: RouterAccuracyAccumulator() for s in masks[w]} for w in masks}

    for m in models.values(): m.eval()
    router.eval()
    with torch.no_grad():
        for bi in batches(np.arange(len(ids)), bs):
            src = ids[bi]
            scores, target, cm, valid = get_expert_scores(models, cfg, art, ds, src, top4, bi, need_prob, dev)
            need_t = t(np.asarray(need_prob[bi], np.float32), dev)
            conf_t = t(np.asarray(top4['conf'][bi], np.float32), dev)
            feat, _ = router_features(scores, EXPERT_ORDER, need_t, conf_t, cm, temp)
            logits = router(feat)
            labels, use_router, _ = router_targets(scores, EXPERT_ORDER, target, cm, valid,
                                                    float(c['router']['target_min_gap']))
            routed = route_candidate_scores(logits, scores, EXPERT_ORDER, temp, soft=False)
            routed_soft = route_candidate_scores(logits, scores, EXPERT_ORDER, temp, soft=True)
            oracle = oracle_expert_scores(scores, EXPERT_ORDER, target, cm, temp)
            vscores = {**scores, 'router': routed, 'router_soft': routed_soft, 'oracle_router': oracle}

            for view in ['overall','masked','support']:
                for sub, full_mask in masks[view].items():
                    local = np.asarray(full_mask[bi], bool)
                    cell = torch.as_tensor(np.transpose(local, (0,2,1)), device=dev, dtype=torch.bool)
                    for name in variants:
                        acc[name][view][sub].update(vscores[name], target, cm, cell)
                    racc[view][sub].update(logits, labels, use_router & cell)

    return {
        'variants': {v: {'views': {w: {s: a.summary() for s, a in ss.items()} for w, ss in ww.items()}}
                     for v, ww in acc.items()},
        'router_accuracy': {w: {s: a.summary() for s, a in ss.items()} for w, ss in racc.items()},
        'fixed_train_only_predicted_need_thresholds': {k: np.asarray(v, np.float32).tolist() for k,v in thresholds.items()},
        'oracle_router_is_privileged_validation_diagnostic_only': True,
        'true_q90_is_privileged_validation_diagnostic_only': True,
    }


def decision(summary, cfg):
    d = cfg['deployment_aligned_utility_router_probe']['decision']
    vv = summary['final_validation']['variants']
    generic = vv['generic']['views']['masked']['pred_top10']
    candidates = ['pred10','pred20','router','router_soft']
    best_name = max(candidates, key=lambda n: vv[n]['views']['masked']['pred_top10']['pairwise_rank_accuracy'])
    best = vv[best_name]['views']['masked']['pred_top10']
    gr = float(generic['pairwise_rank_accuracy']); br = float(best['pairwise_rank_accuracy'])
    gain = br - gr; benef = float(best['selected_beneficial_rate']); harm = float(best['selected_harmful_rate'])
    go = br >= float(d['masked_pred_top10_rank_go']) and gain >= float(d['rank_gain_go']) and \
         benef >= float(d['selected_beneficial_go']) and harm <= float(d['selected_harmful_go'])
    cond = br >= float(d['masked_pred_top10_rank_conditional']) and gain >= float(d['rank_gain_conditional']) and \
           harm <= float(d['selected_harmful_conditional'])
    status = 'GO_V08_DEPLOYMENT_ROUTER' if go else ('CONDITIONAL_V08_DEPLOYMENT_ROUTER' if cond else 'NO_GO_V08_DEPLOYMENT_ROUTER')
    oracle = vv['oracle_router']['views']['masked']['pred_top10']
    diag = {
        'best_variant': best_name,
        'generic_masked_pred_top10_rank': gr,
        'best_masked_pred_top10_rank': br,
        'rank_gain_over_generic': gain,
        'best_selected_beneficial_rate': benef,
        'best_selected_harmful_rate': harm,
        'oracle_router_masked_pred_top10_rank': float(oracle['pairwise_rank_accuracy']),
        'oracle_router_masked_pred_top10_beneficial': float(oracle['selected_beneficial_rate']),
        'oracle_router_masked_pred_top10_harmful': float(oracle['selected_harmful_rate']),
        'router_masked_pred_top10_expert_accuracy': summary['final_validation']['router_accuracy']['masked']['pred_top10']['expert_preference_accuracy'],
        'thresholds': d,
    }
    return status, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='development', choices=['development'])
    ap.add_argument('--seed', type=int, default=10)
    a = ap.parse_args()
    cfg = load_yaml(ROOT / 'configs/presure.yaml'); seed_all(a.seed)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    art = load_artifacts(cfg, a.stage); cache = cache_dir(cfg, a.stage)
    tr = CacheSplit(cache, 'train'); va = CacheSplit(cache, 'val')

    _, exact_rd = load_latest(cfg['project']['exp_root'], 'PredictiveNeedProbe', cfg['experiment']['subgraph'], a.stage)
    _, state_rd = load_latest(cfg['project']['exp_root'], 'PredictiveNeedStateProbe', cfg['experiment']['subgraph'], a.stage)
    _, v074_rd = load_latest(cfg['project']['exp_root'], 'NeedConditionedUtilitySpecialistProbe', cfg['experiment']['subgraph'], a.stage)
    state_summary = json.loads((state_rd / 'summary.json').read_text())
    q90_raw = np.asarray(state_summary['train_only_thresholds']['q90'], np.float32)

    generic, utility_ck = instantiate_utility_a(exact_rd / 'utility_A_best.pt', dev)
    hard, hard_ck = load_v074_hard(v074_rd, dev)
    q90_model, _ = load_state_q90(state_rd, dev)

    ids_train = np.load(exact_rd / 'utility_train_ids.npy').astype(np.int64)
    ids_dev = np.load(exact_rd / 'utility_dev_ids.npy').astype(np.int64)
    top_train = load_top4(exact_rd, 'utility_train'); top_dev = load_top4(exact_rd, 'utility_dev')
    nft = np.load(art / 'train_node_features.npy', mmap_mode='r')
    bs_need = int(cfg['deployment_aligned_utility_router_probe']['evaluation']['need_eval_batch_size'])
    need_train = predict_need_q90(q90_model, nft, ids_train, bs_need, dev)
    need_dev = predict_need_q90(q90_model, nft, ids_dev, bs_need, dev)
    train_valid = np.asarray(tr.valid[ids_train], bool)
    thresholds = {
        'pred10': need_thresholds_by_horizon(need_train, train_valid, 0.10),
        'pred20': need_thresholds_by_horizon(need_train, train_valid, 0.20),
    }

    rd = run_dir(cfg, 'DeploymentAlignedUtilityRouterProbe', a.stage, stamp(), a.seed); rd.mkdir(parents=True, exist_ok=True)
    lineage = {
        'variant': 'DeploymentAlignedUtilityRouterProbe', 'stage': a.stage, 'seed': a.seed,
        'implementation_revision': cfg.get('implementation_revision'),
        'v071_utility_A': str(exact_rd), 'v072_need_q90': str(state_rd), 'v074_hard_specialist': str(v074_rd),
        'deployment_specialist_masks': 'Train predicted Need-q90 only',
        'expert_router_labels': 'Train-only utility_dev future-grounded expert preference',
        'validation_future_used_for_reporting_only': True,
    }
    write_json(rd/'RUN_MANIFEST.json', {**lineage, 'status':'STARTED'})
    write_json(rd/'train_only_predicted_need_thresholds.json', {k: np.asarray(v).tolist() for k,v in thresholds.items()})

    try:
        pred10, ck10 = train_deployment_specialist('pred10', 0.10, thresholds, cfg, art, tr,
                                                   ids_train, ids_dev, top_train, top_dev, need_train, need_dev,
                                                   utility_ck, rd, dev, a.seed + 10)
        pred20, ck20 = train_deployment_specialist('pred20', 0.20, thresholds, cfg, art, tr,
                                                   ids_train, ids_dev, top_train, top_dev, need_train, need_dev,
                                                   utility_ck, rd, dev, a.seed + 20)
        models = {'generic':generic, 'hard':hard, 'pred10':pred10, 'pred20':pred20}
        router, rck, rsplit = train_router(models, cfg, art, tr, ids_dev, top_dev, need_dev, rd, dev, a.seed + 30)
        write_json(rd/'router_train_only_split.json', rsplit)

        state_split = json.loads((state_rd/'split_manifest.json').read_text())
        val_ids = np.asarray(state_split['final_val'], np.int64)
        need_val = np.asarray(np.load(state_rd/'prob_need_q90_final_val.npy', mmap_mode='r'), np.float32)[...,1]
        top_val = load_state_top4(state_rd)
        final = evaluate_all(models, router, cfg, art, va, val_ids, top_val, need_val, q90_raw, thresholds, dev, a.seed + 5000)
        write_json(rd/'deployment_router_validation_summary.json', final)

        summary = {
            'variant':'DeploymentAlignedUtilityRouterProbe', 'stage':a.stage, 'seed':a.seed,
            'sources': {'v071_utility_A':str(exact_rd), 'v072_need_q90':str(state_rd), 'v074_hard_specialist':str(v074_rd)},
            'training': {
                'pred10_best_epoch': int(ck10['epoch']), 'pred20_best_epoch': int(ck20['epoch']),
                'router_best_epoch': int(rck['epoch']), 'router_dev_expert_accuracy': float(rck['dev_expert_accuracy']),
                'future_usage':'Train utility supervision / Train-only expert-preference labels only',
                'deployment_inputs':'predicted Need-q90 + expert scores/confidence + M3 confidence; no current-query future',
            },
            'train_only_predicted_need_thresholds': {k: np.asarray(v).tolist() for k,v in thresholds.items()},
            'final_validation': final,
            'test_evaluated': False,
        }
        status, diag = decision(summary, cfg)
        summary['v08_deployment_router_recommendation'] = status; summary['decision_diagnostics'] = diag
        write_json(rd/'summary.json', summary)
        (rd/'DECISION.txt').write_text(status+'\n'+json.dumps(diag,indent=2,sort_keys=True)+'\n')

        rows=[]
        for variant,obj in final['variants'].items():
            for view,subs in obj['views'].items():
                for subset,met in subs.items(): rows.append({'variant':variant,'view':view,'subset':subset,**met})
        keys=sorted(set().union(*(r.keys() for r in rows)))
        with open(rd/'deployment_router_comparison.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

        latest=Path(cfg['project']['exp_root'])/'DeploymentAlignedUtilityRouterProbe'/cfg['experiment']['subgraph']/a.stage
        latest.mkdir(parents=True,exist_ok=True)
        write_json(latest/'LATEST.json',{'run_dir':str(rd),'summary':str(rd/'summary.json'),'recommendation':status,'implementation_revision':cfg.get('implementation_revision')})
        write_json(rd/'RUN_MANIFEST.json',{**lineage,'status':'FINISHED','recommendation':status})
        print('DEPLOYMENT-ALIGNED UTILITY ROUTER FINAL',status,flush=True)
        print(json.dumps(summary,indent=2,sort_keys=True),flush=True)
        print(f'DeploymentAlignedUtilityRouterProbe PASS run_dir={rd}',flush=True)
    except Exception as e:
        write_json(rd/'RUN_MANIFEST.json',{**lineage,'status':'FAILED','error':repr(e)})
        raise


if __name__=='__main__':
    main()
