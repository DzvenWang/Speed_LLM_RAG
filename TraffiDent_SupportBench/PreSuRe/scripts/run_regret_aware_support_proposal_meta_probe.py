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
from presure.need_gated import instantiate_utility_a, load_latest
from presure.probes import SupportUtilityProbeNet
from presure.deployment_routing import (
    ExpertPreferenceRouter, router_features, route_candidate_scores,
    predicted_need_mask,
)
from presure.proposal_meta import (
    SupportProposalMetaRanker,
    ProposalMetricAccumulator,
    build_support_proposals,
    meta_ranker_loss,
)
from presure.utility_specialist import true_q90_mask
from presure.pipeline import load_artifacts, run_dir
from scripts.run_need_conditioned_utility_specialist_probe import (
    batches, t, load_state_q90, predict_need_q90, load_top4, load_state_top4, view_np, batch_scores,
)
from scripts.run_deployment_aligned_utility_router_probe import load_v074_hard


EXPERT_ORDER = ['generic', 'hard', 'pred10', 'pred20']


def load_v075(v075_rd: Path, dev):
    models = {}
    for mode in ['pred10', 'pred20']:
        p = v075_rd / f'{mode}_specialist_best.pt'
        ck = torch.load(p, map_location=dev, weights_only=False)
        if ck.get('model_type') != 'deployment_aligned_utility_specialist_v075' or ck.get('mode') != mode:
            raise RuntimeError(f'v0.7.5 {mode} checkpoint contract mismatch: {p}')
        m = SupportUtilityProbeNet(**ck['model_args']).to(dev)
        m.load_state_dict(ck['state_dict']); m.eval()
        models[mode] = m

    rp = v075_rd / 'expert_preference_router_best.pt'
    rck = torch.load(rp, map_location=dev, weights_only=False)
    if rck.get('model_type') != 'expert_preference_router_v075':
        raise RuntimeError(f'v0.7.5 router checkpoint contract mismatch: {rp}')
    if list(rck.get('expert_order', [])) != EXPERT_ORDER:
        raise RuntimeError(f'v0.7.5 expert order mismatch: {rck.get("expert_order")}')
    router = ExpertPreferenceRouter(**rck['model_args']).to(dev)
    router.load_state_dict(rck['state_dict']); router.eval()
    return models, router, rck


def get_expert_batch(models, cfg, art, ds, src, top4, local_idx, need_prob, dev):
    scores = {}
    target = cm = valid = rr_raw = None
    with torch.no_grad():
        for name in EXPERT_ORDER:
            cond = name != 'generic'
            pred, tar, cmm, vv, rr = batch_scores(
                models[name], cfg, art, ds, src, top4, local_idx, need_prob, dev, cond
            )
            scores[name] = pred
            if target is None:
                target, cm, valid, rr_raw = tar, cmm, vv, rr
    return scores, target, cm, valid, rr_raw


def proposal_score_from_candidate_score(candidate_score, proposal_idx):
    """candidate_score [B,N,L,P], proposal_idx [B,N,P,S] -> [B,N,P,S]."""
    x = candidate_score.permute(0, 1, 3, 2)  # [B,N,P,L]
    return torch.gather(x, -1, proposal_idx)


def proposal_masks(ds, ids, need_prob, q90_raw, thresholds, art):
    base = np.load(art / ('val_base_pred.npy' if ds.split == 'val' else 'train_base_pred.npy'), mmap_mode='r')
    true_hard = true_q90_mask(np.asarray(base[ids]), np.asarray(ds.y[ids]), q90_raw)
    p10 = predicted_need_mask(need_prob, np.asarray(thresholds['pred10'], np.float32))
    p20 = predicted_need_mask(need_prob, np.asarray(thresholds['pred20'], np.float32))
    out = {}
    for view in ['overall', 'masked', 'support']:
        vm = view_np(ds, ids, view)
        out[view] = {
            'all': vm,
            'pred_top10': vm & p10,
            'pred_top20': vm & p20,
            'true_q90': vm & true_hard,
        }
    return out


def build_bundle(models, cfg, art, ds, ids, top4, local_idx, need_prob, dev, residual_std):
    src = ids[local_idx]
    scores, target, cm, valid, rr_raw = get_expert_batch(models, cfg, art, ds, src, top4, local_idx, need_prob, dev)
    need_t = t(np.asarray(need_prob[local_idx], np.float32), dev)
    conf_t = t(np.asarray(top4['conf'][local_idx], np.float32), dev)
    temp = float(cfg['predictive_need_probe']['utility_model']['classification_temperature'])
    node_mem = np.load(art / 'train_node_features.npy', mmap_mode='r')
    node_q = np.load(art / ('val_node_features.npy' if ds.split == 'val' else 'train_node_features.npy'), mmap_mode='r')
    qctx_np = np.asarray(node_q[src], np.float32)
    gids = np.asarray(top4['ids'][local_idx], np.int64)
    ni = np.arange(gids.shape[1], dtype=np.int64)[None, :, None]
    cctx_np = np.asarray(node_mem)[gids, ni, :].astype(np.float32, copy=False)
    qctx_t = t(qctx_np, dev)
    cctx_t = t(cctx_np, dev)
    bundle = build_support_proposals(
        scores, EXPERT_ORDER, target, rr_raw, conf_t, cm, need_t, temp, residual_std,
        query_context=qctx_t, support_context=cctx_t,
    )
    return scores, target, cm, valid, need_t, conf_t, bundle


def train_meta_ranker(models, v075_router, cfg, art, tr, ids_dev, top_dev, need_dev, thresholds, rd, dev, seed):
    c = cfg['proposal_meta_ranking_probe']['training']
    _, residual_std = cache_stats(cache_dir(cfg, 'development'))
    rng = np.random.default_rng(seed + 7601)
    order = rng.permutation(len(ids_dev))
    ntrain = max(1, int(round(len(order) * float(c['train_fraction']))))
    rid_train, rid_hold = order[:ntrain], order[ntrain:]
    if len(rid_hold) == 0:
        rid_hold = order[-max(1, len(order)//5):]
        rid_train = order[:max(1, len(order)-len(rid_hold))]

    # Infer feature dimensionality from one mini-batch.
    probe = np.asarray(rid_train[:1], np.int64)
    _, _, _, valid0, _, _, b0 = build_bundle(models, cfg, art, tr, ids_dev, top_dev, probe, need_dev, dev, residual_std)
    input_dim = int(b0['features'].shape[-1])
    model = SupportProposalMetaRanker(input_dim, int(c['hidden']), float(c['dropout'])).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c['lr']), weight_decay=float(c['weight_decay']))
    best_score, wait = -1e18, 0
    best_path = rd / 'proposal_meta_ranker_best.pt'

    def one_pass(local_ids, train, epoch):
        if train: model.train()
        else: model.eval()
        losses, rankacc, counts = [], [], []
        metric = ProposalMetricAccumulator(
            float(cfg['predictive_need_probe']['utility_model']['beneficial_threshold']),
            float(cfg['proposal_meta_ranking_probe']['evaluation']['near_oracle_tolerance']),
            int(cfg['proposal_meta_ranking_probe']['evaluation']['regret_sample_points_per_batch']),
            seed + 9000 + epoch + (0 if train else 500),
        )
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for pos in batches(local_ids, int(c['batch_size']), shuffle=train, seed=seed + epoch * 137 + (1 if train else 2)):
                _, _, _, valid, need_t, _, bundle = build_bundle(
                    models, cfg, art, tr, ids_dev, top_dev, pos, need_dev, dev, residual_std
                )
                pred = model(bundle['features'])
                # direct_mask is allowed only as Train weighting / evaluation view, never as an input feature.
                direct_np = np.asarray(tr.direct[ids_dev[pos]], bool)
                masked_valid = valid & t(direct_np[:, None, :], dev, torch.bool)
                loss, parts = meta_ranker_loss(pred, bundle['target'], bundle['mask'], masked_valid, need_t, cfg)
                if train:
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(c['grad_clip']))
                    opt.step()
                losses.append(float(loss.detach()))
                if parts['cells']:
                    rankacc.append(parts['rank_acc']); counts.append(parts['cells'])
                cell = masked_valid.bool().permute(0, 2, 1)
                if not train:
                    dep_np = predicted_need_mask(
                        np.asarray(need_dev[pos], np.float32),
                        np.asarray(thresholds['pred10'], np.float32),
                        np.asarray(tr.valid[ids_dev[pos]], bool),
                    )
                    cell &= t(np.transpose(dep_np, (0, 2, 1)), dev, torch.bool)
                metric.update(pred.detach(), bundle['target'], bundle['mask'], cell)
        sm = metric.summary()
        return float(np.mean(losses)) if losses else 0.0, sm, (
            float(np.average(rankacc, weights=counts)) if counts else 0.0
        )

    for ep in range(1, int(c['epochs']) + 1):
        tl, tm, tra = one_pass(rid_train, True, ep)
        dl, dm, dra = one_pass(rid_hold, False, ep)
        # Selection quality is primary; regret is explicitly penalized.
        score = (
            float(dm['pairwise_rank_accuracy'])
            + 0.30 * float(dm['selected_beneficial_rate'])
            - 0.30 * float(dm['selected_harmful_rate'])
            - 0.75 * float(dm['mean_selection_regret'])
        )
        print(
            f'PROPOSAL-META epoch={ep:03d} train_loss={tl:.5f} train_rank={tra:.3f} '
            f'dev_rank={dm["pairwise_rank_accuracy"]:.3f} dev_benef={dm["selected_beneficial_rate"]:.3f} '
            f'dev_harm={dm["selected_harmful_rate"]:.3f} dev_regret={dm["mean_selection_regret"]:.5f}',
            flush=True,
        )
        if score > best_score:
            best_score, wait = score, 0
            torch.save({
                'state_dict': model.state_dict(),
                'model_type': 'regret_aware_support_proposal_meta_ranker_v076',
                'model_args': {'input_dim': input_dim, 'hidden': int(c['hidden']), 'dropout': float(c['dropout'])},
                'expert_order': EXPERT_ORDER,
                'epoch': ep,
                'dev_score': score,
                'dev_metrics': dm,
                'training_objective': 'pairwise + utility + beneficial/harmful + expected selection regret',
                'future_usage': 'Train-only utility targets; no current-query future in meta-ranker input',
            }, best_path)
        else:
            wait += 1
        if wait >= int(c['patience']):
            break

    ck = torch.load(best_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck['state_dict']); model.eval()
    return model, ck, {
        'meta_train_local_ids': rid_train.tolist(),
        'meta_holdout_local_ids': rid_hold.tolist(),
        'meta_train_query_count': int(len(rid_train)),
        'meta_holdout_query_count': int(len(rid_hold)),
    }


def evaluate(models, v075_router, meta, cfg, art, ds, ids, top4, need_prob, q90_raw, thresholds, dev, seed):
    _, residual_std = cache_stats(cache_dir(cfg, 'development'))
    c = cfg['proposal_meta_ranking_probe']
    bs = int(c['evaluation']['eval_batch_size'])
    masks = proposal_masks(ds, ids, need_prob, q90_raw, thresholds, art)
    variants = ['generic', 'hard', 'pred10', 'pred20', 'v075_router', 'v075_router_soft', 'meta_ranker', 'oracle_proposal']
    acc = {
        v: {
            view: {
                subset: ProposalMetricAccumulator(
                    float(cfg['predictive_need_probe']['utility_model']['beneficial_threshold']),
                    float(c['evaluation']['near_oracle_tolerance']),
                    int(c['evaluation']['regret_sample_points_per_batch']),
                    seed + 10000 * vi + 1000 * wi + 31 * si,
                )
                for si, subset in enumerate(masks[view])
            }
            for wi, view in enumerate(['overall', 'masked', 'support'])
        }
        for vi, v in enumerate(variants)
    }
    temp = float(cfg['predictive_need_probe']['utility_model']['classification_temperature'])

    with torch.no_grad():
        for pos in batches(np.arange(len(ids)), bs):
            scores, _, cm, valid, need_t, conf_t, bundle = build_bundle(
                models, cfg, art, ds, ids, top4, pos, need_prob, dev, residual_std
            )
            pidx = bundle['candidate_idx']
            pscore = {
                name: proposal_score_from_candidate_score(torch.sigmoid(scores[name] / temp), pidx)
                for name in EXPERT_ORDER
            }
            feat075, _ = router_features(scores, EXPERT_ORDER, need_t, conf_t, cm, temp)
            logits075 = v075_router(feat075)
            routed = route_candidate_scores(logits075, scores, EXPERT_ORDER, temp, soft=False)
            routed_soft = route_candidate_scores(logits075, scores, EXPERT_ORDER, temp, soft=True)
            pscore['v075_router'] = proposal_score_from_candidate_score(routed, pidx)
            pscore['v075_router_soft'] = proposal_score_from_candidate_score(routed_soft, pidx)
            pscore['meta_ranker'] = meta(bundle['features'])
            pscore['oracle_proposal'] = bundle['target']

            for view in ['overall', 'masked', 'support']:
                for subset, full_mask in masks[view].items():
                    local = np.asarray(full_mask[pos], bool)
                    cell = t(np.transpose(local, (0, 2, 1)), dev, torch.bool)
                    for name in variants:
                        acc[name][view][subset].update(
                            pscore[name], bundle['target'], bundle['mask'], cell
                        )

    return {
        'variants': {
            v: {'views': {w: {s: a.summary() for s, a in subs.items()} for w, subs in views.items()}}
            for v, views in acc.items()
        },
        'proposal_pool': {
            'experts': EXPERT_ORDER,
            'max_proposals_per_cell': len(EXPERT_ORDER),
            'duplicates_merged': True,
            'oracle_proposal_is_privileged_validation_diagnostic_only': True,
        },
        'true_q90_is_privileged_validation_diagnostic_only': True,
        'fixed_train_only_predicted_need_thresholds': {k: np.asarray(v, np.float32).tolist() for k, v in thresholds.items()},
    }


def decision(final, cfg):
    d = cfg['proposal_meta_ranking_probe']['decision']
    vv = final['variants']
    g = vv['generic']['views']['masked']['pred_top10']
    m = vv['meta_ranker']['views']['masked']['pred_top10']
    o = vv['oracle_proposal']['views']['masked']['pred_top10']
    gr = float(g['pairwise_rank_accuracy']); mr = float(m['pairwise_rank_accuracy'])
    gain = mr - gr
    greg = max(float(g['mean_selection_regret']), 1e-12)
    mreg = float(m['mean_selection_regret'])
    regret_reduction = (greg - mreg) / greg
    benef = float(m['selected_beneficial_rate']); harm = float(m['selected_harmful_rate'])
    hreg = float(m['harmful_regret_rate'])
    go = (
        mr >= float(d['masked_pred_top10_rank_go'])
        and gain >= float(d['rank_gain_go'])
        and benef >= float(d['selected_beneficial_go'])
        and harm <= float(d['selected_harmful_go'])
        and regret_reduction >= float(d['mean_regret_reduction_go'])
        and hreg <= float(d['harmful_regret_go'])
    )
    cond = (
        mr >= float(d['masked_pred_top10_rank_conditional'])
        and gain >= float(d['rank_gain_conditional'])
        and harm <= float(d['selected_harmful_conditional'])
        and regret_reduction >= float(d['mean_regret_reduction_conditional'])
    )
    status = 'GO_V08_PROPOSAL_META' if go else ('CONDITIONAL_V08_PROPOSAL_META' if cond else 'NO_GO_V08_PROPOSAL_META')
    return status, {
        'generic_masked_pred_top10_rank': gr,
        'meta_masked_pred_top10_rank': mr,
        'rank_gain_over_generic': gain,
        'generic_mean_selection_regret': float(g['mean_selection_regret']),
        'meta_mean_selection_regret': mreg,
        'mean_regret_reduction_fraction': regret_reduction,
        'meta_selected_beneficial_rate': benef,
        'meta_selected_harmful_rate': harm,
        'meta_harmful_regret_rate': hreg,
        'oracle_proposal_masked_pred_top10_rank': float(o['pairwise_rank_accuracy']),
        'oracle_proposal_selected_beneficial_rate': float(o['selected_beneficial_rate']),
        'oracle_proposal_selected_harmful_rate': float(o['selected_harmful_rate']),
        'oracle_proposal_mean_selection_regret': float(o['mean_selection_regret']),
        'thresholds': d,
    }


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
    _, v075_rd = load_latest(cfg['project']['exp_root'], 'DeploymentAlignedUtilityRouterProbe', cfg['experiment']['subgraph'], a.stage)

    generic, _ = instantiate_utility_a(exact_rd / 'utility_A_best.pt', dev)
    hard, _ = load_v074_hard(v074_rd, dev)
    v075_models, v075_router, v075_router_ck = load_v075(v075_rd, dev)
    models = {'generic': generic, 'hard': hard, **v075_models}
    q90_model, _ = load_state_q90(state_rd, dev)

    state_summary = json.loads((state_rd / 'summary.json').read_text())
    q90_raw = np.asarray(state_summary['train_only_thresholds']['q90'], np.float32)
    v075_summary = json.loads((v075_rd / 'summary.json').read_text())
    thresholds = {
        k: np.asarray(v, np.float32)
        for k, v in v075_summary['train_only_predicted_need_thresholds'].items()
    }

    ids_dev = np.load(exact_rd / 'utility_dev_ids.npy').astype(np.int64)
    top_dev = load_top4(exact_rd, 'utility_dev')
    nft = np.load(art / 'train_node_features.npy', mmap_mode='r')
    need_dev = predict_need_q90(
        q90_model, nft, ids_dev,
        int(cfg['proposal_meta_ranking_probe']['evaluation']['need_eval_batch_size']), dev,
    )

    rd = run_dir(cfg, 'RegretAwareSupportProposalMetaRankingProbe', a.stage, stamp(), a.seed)
    rd.mkdir(parents=True, exist_ok=True)
    lineage = {
        'variant': 'RegretAwareSupportProposalMetaRankingProbe',
        'stage': a.stage, 'seed': a.seed,
        'implementation_revision': cfg.get('implementation_revision'),
        'v071_utility_A': str(exact_rd),
        'v072_need_q90': str(state_rd),
        'v074_hard_specialist': str(v074_rd),
        'v075_deployment_router': str(v075_rd),
        'experts_frozen': EXPERT_ORDER,
        'meta_training_split': 'Train-only utility_dev split; direct_mask used only as training/evaluation weighting',
        'validation_future_used_for_reporting_only': True,
    }
    write_json(rd / 'RUN_MANIFEST.json', {**lineage, 'status': 'STARTED'})

    try:
        meta, mck, split = train_meta_ranker(
            models, v075_router, cfg, art, tr, ids_dev, top_dev, need_dev, thresholds, rd, dev, a.seed + 76
        )
        write_json(rd / 'meta_ranker_train_only_split.json', split)

        state_split = json.loads((state_rd / 'split_manifest.json').read_text())
        val_ids = np.asarray(state_split['final_val'], np.int64)
        need_val = np.asarray(np.load(state_rd / 'prob_need_q90_final_val.npy', mmap_mode='r'), np.float32)[..., 1]
        top_val = load_state_top4(state_rd)
        final = evaluate(
            models, v075_router, meta, cfg, art, va, val_ids, top_val, need_val,
            q90_raw, thresholds, dev, a.seed + 7600,
        )
        write_json(rd / 'proposal_meta_validation_summary.json', final)

        reference = {
            'v075_decision': v075_summary.get('v08_deployment_router_recommendation'),
            'v075_masked_pred_top10': {
                k: v075_summary['final_validation']['variants'][k]['views']['masked']['pred_top10']
                for k in ['generic', 'router', 'router_soft', 'oracle_router']
            },
        }
        summary = {
            'variant': 'RegretAwareSupportProposalMetaRankingProbe',
            'stage': a.stage, 'seed': a.seed,
            'sources': lineage,
            'training': {
                'meta_best_epoch': int(mck['epoch']),
                'meta_dev_score': float(mck['dev_score']),
                'future_usage': 'Train-only proposal utility supervision; current-query future never enters meta-ranker features',
                'deployment_inputs': 'Need-q90 + frozen expert score/confidence + M3 confidence + historical support residual + expert agreement',
                'v075_router_best_epoch': int(v075_router_ck['epoch']),
            },
            'reference_v075_full_top4_metrics': reference,
            'final_validation': final,
            'test_evaluated': False,
        }
        status, diag = decision(final, cfg)
        summary['v08_proposal_meta_recommendation'] = status
        summary['decision_diagnostics'] = diag
        write_json(rd / 'summary.json', summary)
        (rd / 'DECISION.txt').write_text(status + '\n' + json.dumps(diag, indent=2, sort_keys=True) + '\n')

        rows = []
        for variant, obj in final['variants'].items():
            for view, subsets in obj['views'].items():
                for subset, met in subsets.items():
                    rows.append({'variant': variant, 'view': view, 'subset': subset, **met})
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with open(rd / 'proposal_meta_comparison.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

        latest = Path(cfg['project']['exp_root']) / 'RegretAwareSupportProposalMetaRankingProbe' / cfg['experiment']['subgraph'] / a.stage
        latest.mkdir(parents=True, exist_ok=True)
        write_json(latest / 'LATEST.json', {
            'run_dir': str(rd), 'summary': str(rd / 'summary.json'),
            'recommendation': status, 'implementation_revision': cfg.get('implementation_revision'),
        })
        write_json(rd / 'RUN_MANIFEST.json', {**lineage, 'status': 'FINISHED', 'recommendation': status})
        print('REGRET-AWARE SUPPORT PROPOSAL META-RANKING FINAL', status, flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f'RegretAwareSupportProposalMetaRankingProbe PASS run_dir={rd}', flush=True)
    except Exception as e:
        write_json(rd / 'RUN_MANIFEST.json', {**lineage, 'status': 'FAILED', 'error': repr(e)})
        raise


if __name__ == '__main__':
    main()
