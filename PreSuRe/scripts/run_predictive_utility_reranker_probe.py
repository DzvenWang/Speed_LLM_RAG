#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
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
    UtilitySelectionAccumulator,
    node_horizon_utility_targets,
    normalize_coarse_scores,
    predictive_utility_loss,
    risk_score,
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
    c = cfg['predictive_utility_retrieval']['model']
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


def gather_batch(cfg, art, ds, src, cand, node_q, node_mem, residual_mem, base_raw, mean, std, dev):
    pc = cfg['predictive_utility_retrieval']
    H = int(cfg['experiment']['history'])
    P = int(cfg['experiment']['horizon'])
    K = int(pc['candidate_k'])
    gids_raw = np.asarray(cand['indices'][src, :K], np.int64)
    scores_raw = np.asarray(cand['scores'][src, :K], np.float32)
    query_mask = gids_raw >= 0
    gids = np.where(query_mask, gids_raw, 0)

    qctx_np = np.asarray(node_q[src], np.float32)
    # gather_support_context is a legacy node-local helper whose contract is
    # gids=[B,N,K].  v0.9A candidates are episode-level [B,K] and are shared
    # across nodes, so expand only the integer index view before gathering.
    # This does not duplicate node/residual memory and preserves the same K=24
    # candidate set for every node.
    gids_node = np.broadcast_to(
        gids[:, None, :],
        (gids.shape[0], qctx_np.shape[1], gids.shape[1]),
    )
    cctx_np, rr_raw_np = gather_support_context(node_mem, residual_mem, gids_node)
    # [B,N,K] availability; the episode-level candidate set is shared across nodes.
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


def evaluate(model, cfg, art, ds, ids, cand, node_q, node_mem, residual_mem, base_raw, mean, std, dev, seed):
    pc = cfg['predictive_utility_retrieval']
    tc = pc['training']
    betas = [float(x) for x in pc['risk']['beta_grid']]
    names = ['coarse', 'mu', 'benefit'] + [f'risk_{b:g}' for b in betas]
    acc = {
        name: {
            view: UtilitySelectionAccumulator(
                float(tc['beneficial_threshold']), float(tc['harmful_threshold']),
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
            pack = gather_batch(cfg, art, ds, src, cand, node_q, node_mem, residual_mem, base_raw, mean, std, dev)
            qctx, cctx, qs, ss, rr, qbase, sbase, coarse, cm, valid, target, _ = pack
            mu, sigma, blogit = model(qctx, cctx, qs, ss, rr, qbase, sbase, coarse)
            sigma_sum += float(sigma[cm[:, :, :, None].expand_as(sigma)].sum().cpu())
            sigma_n += int(cm.sum().cpu()) * int(mu.shape[-1])
            B, N, K, P = mu.shape
            coarse_score = coarse[:, None, :, None].expand(B, N, K, P)
            score_map = {
                'coarse': coarse_score,
                'mu': mu,
                'benefit': torch.sigmoid(blogit / float(tc['classification_temperature'])),
            }
            for beta in betas:
                score_map[f'risk_{beta:g}'] = risk_score(mu, sigma, beta)
            for view in ['overall', 'masked', 'support']:
                vm = t(view_mask(ds, src, view), dev, torch.bool)
                for name, score in score_map.items():
                    acc[name][view].update(score, target, cm, vm)

    out = {
        name: {view: obj.summary() for view, obj in views.items()}
        for name, views in acc.items()
    }
    out['_mean_sigma'] = float(sigma_sum / max(sigma_n, 1))
    return out


def choose_beta(summary, cfg):
    r = cfg['predictive_utility_retrieval']['risk']
    best = None
    for beta in [float(x) for x in r['beta_grid']]:
        key = f'risk_{beta:g}'
        m = summary[key]['masked']
        score = (
            float(r['dev_score_rank_weight']) * float(m['pairwise_rank_accuracy'])
            + float(r['dev_score_benefit_weight']) * float(m['selected_beneficial_rate'])
            - float(r['dev_score_harm_weight']) * float(m['selected_harmful_rate'])
        )
        item = (score, -beta, beta, key)
        if best is None or item > best:
            best = item
    return float(best[2]), best[3], float(best[0])


def regret_reduction(base_summary, new_summary):
    b = float(base_summary['mean_selection_regret'])
    n = float(new_summary['mean_selection_regret'])
    return 0.0 if b <= 1e-12 else float((b - n) / b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='development', choices=['development'])
    ap.add_argument('--seed', type=int, default=None)
    a = ap.parse_args()

    cfg = load_yaml(ROOT / 'configs/presure.yaml')
    if str(cfg.get('version')) not in {'0.9.0','0.9.1','0.9.2','0.9.3','0.9.4','0.9.5'}:
        raise RuntimeError(f'v0.9A runner requires config version 0.9.0/0.9.1/0.9.2, got {cfg.get("version")}')
    seed = int(cfg['experiment']['seed'] if a.seed is None else a.seed)
    seed_all(seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    art = load_artifacts(cfg, a.stage)
    cache = cache_dir(cfg, a.stage)
    tr = CacheSplit(cache, 'train')
    va = CacheSplit(cache, 'val')
    mean, std = cache_stats(cache)

    node_mem = np.load(art / 'train_node_features.npy', mmap_mode='r')
    node_val = np.load(art / 'val_node_features.npy', mmap_mode='r')
    residual_mem = np.load(art / 'train_residual.npy', mmap_mode='r')
    train_base = np.load(art / 'train_base_pred.npy', mmap_mode='r')
    val_base = np.load(art / 'val_base_pred.npy', mmap_mode='r')
    train_cand = np.load(art / 'train_candidates.npz')
    val_cand = np.load(art / 'val_candidates.npz')

    sc = cfg['predictive_utility_retrieval']['split']
    train_ids, dev_ids = split_ids(len(tr), sc['train_queries'], sc['dev_queries'], seed + 101)
    fval_ids = final_val_ids(len(va), sc['final_val_queries'], seed + 303)

    rd = run_dir(cfg, 'PredictiveUtilityRetrieval', a.stage, stamp(), seed)
    rd.mkdir(parents=True, exist_ok=True)
    np.save(rd / 'train_ids.npy', train_ids)
    np.save(rd / 'dev_ids.npy', dev_ids)
    np.save(rd / 'final_val_ids.npy', fval_ids)

    args = model_args(cfg, node_mem.shape[-1], node_mem.shape[1], int(cfg['experiment']['horizon']))
    model = PredictiveUtilityRerankerV09A(**args).to(dev)
    initialized_from = None
    if bool(cfg['predictive_utility_retrieval'].get('initialize_encoder_from_m1', True)):
        m1, m1ck, m1path = load_m1(cfg, a.stage, dev)
        if set(m1.encoder.state_dict()) != set(model.encoder.state_dict()):
            raise RuntimeError('M1 encoder keys differ from v0.9A encoder; keep state_dim/node_dim/encoder_hidden compatible')
        for k, v in m1.encoder.state_dict().items():
            if model.encoder.state_dict()[k].shape != v.shape:
                raise RuntimeError(f'M1 encoder tensor shape mismatch for {k}')
        model.encoder.load_state_dict(m1.encoder.state_dict())
        initialized_from = str(m1path)
        print(f'INIT encoder from M1 {m1path}', flush=True)

    tc = cfg['predictive_utility_retrieval']['training']
    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    other_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {'params': enc_params, 'lr': float(tc['lr']) * float(tc['encoder_lr_scale'])},
        {'params': other_params, 'lr': float(tc['lr'])},
    ], weight_decay=float(tc['weight_decay']))

    best_score = -1e18
    best_epoch = 0
    wait = 0
    best_path = rd / 'predictive_utility_reranker_best.pt'

    print('============================================================', flush=True)
    print('PreSuRe v0.9A Predictive-Utility Retrieval', flush=True)
    print(f'device={dev} train={len(train_ids)} dev={len(dev_ids)} final_val={len(fval_ids)} K={cfg["predictive_utility_retrieval"]["candidate_k"]}', flush=True)
    print('Future target is supervision/evaluation only; model inputs are deployment-safe.', flush=True)
    print('============================================================', flush=True)

    for ep in range(1, int(tc['epochs']) + 1):
        model.train()
        losses, ranks, hards = [], [], []
        rank_n = hard_n = 0
        for src in batches(train_ids, int(tc['batch_size']), True, seed + ep * 1009):
            pack = gather_batch(cfg, art, tr, src, train_cand, node_mem, node_mem, residual_mem, train_base, mean, std, dev)
            qctx, cctx, qs, ss, rr, qbase, sbase, coarse, cm, valid, target, _ = pack
            mu, sigma, blogit = model(qctx, cctx, qs, ss, rr, qbase, sbase, coarse)
            loss, parts = predictive_utility_loss(mu, sigma, blogit, target, cm, valid, coarse, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tc['grad_clip']))
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if parts['rank_n']:
                ranks.append(parts['rank_acc']); rank_n += int(parts['rank_n'])
            if parts['hard_n']:
                hards.append(parts['hard_acc']); hard_n += int(parts['hard_n'])

        dev_summary = evaluate(
            model, cfg, art, tr, dev_ids, train_cand, node_mem, node_mem,
            residual_mem, train_base, mean, std, dev, seed + ep * 1777,
        )
        beta, beta_key, score = choose_beta(dev_summary, cfg)
        mm = dev_summary[beta_key]['masked']
        print(
            f'EPOCH {ep:02d} loss={np.mean(losses):.5f} '
            f'train_rank={np.mean(ranks) if ranks else 0.0:.4f} '
            f'train_hard={np.mean(hards) if hards else 0.0:.4f} '
            f'dev_beta={beta:g} masked_rank={mm["pairwise_rank_accuracy"]:.4f} '
            f'beneficial={mm["selected_beneficial_rate"]:.4f} harmful={mm["selected_harmful_rate"]:.4f} '
            f'regret={mm["mean_selection_regret"]:.5f}', flush=True,
        )
        if score > best_score:
            best_score, best_epoch, wait = score, ep, 0
            torch.save({
                'model_type': 'predictive_utility_reranker_v09a',
                'state_dict': model.state_dict(),
                'model_args': args,
                'epoch': ep,
                'dev_score': score,
                'risk_beta': beta,
                'initialized_encoder_from': initialized_from,
                'candidate_k': int(cfg['predictive_utility_retrieval']['candidate_k']),
                'target': 'node_horizon_best_alpha_relative_predictive_utility',
                'future_query_target_is_model_input': False,
            }, best_path)
        else:
            wait += 1
        if wait >= int(tc['patience']):
            print(f'EARLY STOP epoch={ep} best_epoch={best_epoch}', flush=True)
            break

    ck = torch.load(best_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    chosen_beta = float(ck['risk_beta'])

    dev_summary = evaluate(
        model, cfg, art, tr, dev_ids, train_cand, node_mem, node_mem,
        residual_mem, train_base, mean, std, dev, seed + 5551,
    )
    val_summary = evaluate(
        model, cfg, art, va, fval_ids, val_cand, node_val, node_mem,
        residual_mem, val_base, mean, std, dev, seed + 7771,
    )
    beta_key = f'risk_{chosen_beta:g}'
    base_m = val_summary['coarse']['masked']
    new_m = val_summary[beta_key]['masked']
    rr = regret_reduction(base_m, new_m)
    dc = cfg['predictive_utility_retrieval']['decision']
    go_checks = {
        'masked_pairwise_rank': float(new_m['pairwise_rank_accuracy']) >= float(dc['masked_pairwise_rank_go']),
        'selected_beneficial': float(new_m['selected_beneficial_rate']) >= float(dc['selected_beneficial_go']),
        'selected_harmful': float(new_m['selected_harmful_rate']) <= float(dc['selected_harmful_go']),
        'regret_reduction': rr >= float(dc['regret_reduction_go']),
    }
    conditional = (
        float(new_m['pairwise_rank_accuracy']) >= float(dc['masked_pairwise_rank_conditional'])
        and float(new_m['selected_harmful_rate']) <= float(dc['selected_harmful_conditional'])
    )
    decision = 'GO' if all(go_checks.values()) else ('CONDITIONAL' if conditional else 'NO_GO')

    summary = {
        'version': cfg['version'],
        'implementation_revision': cfg.get('implementation_revision'),
        'model_type': 'predictive_utility_reranker_v09a',
        'stage': a.stage,
        'seed': seed,
        'run_dir': str(rd),
        'checkpoint': str(best_path),
        'best_epoch': int(ck['epoch']),
        'chosen_risk_beta': chosen_beta,
        'candidate_k': int(cfg['predictive_utility_retrieval']['candidate_k']),
        'legacy_m3_top4_used': False,
        'need_probability_used_as_reranker_input': False,
        'query_future_used_as_model_input': False,
        'query_future_used_for_train_target_and_diagnostic_eval_only': True,
        'encoder_initialization': initialized_from,
        'dev': dev_summary,
        'final_val': val_summary,
        'masked_regret_reduction_vs_coarse': rr,
        'go_checks': go_checks,
        'decision': decision,
    }
    write_json(rd / 'summary.json', summary)
    # Compact comparison table for rapid log/package review.
    import csv
    with (rd / 'comparison.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant','view','pairwise_rank_accuracy','selected_beneficial_rate','selected_harmful_rate','mean_selection_regret','mean_selected_relative_utility'])
        for variant in ['coarse','mu',beta_key,'benefit']:
            for view in ['overall','masked','support']:
                m = val_summary[variant][view]
                w.writerow([variant,view,m['pairwise_rank_accuracy'],m['selected_beneficial_rate'],m['selected_harmful_rate'],m['mean_selection_regret'],m['mean_selected_relative_utility']])
    (rd / 'DECISION.txt').write_text(
        f'{decision}\n'
        f'best_epoch={ck["epoch"]}\n'
        f'risk_beta={chosen_beta:g}\n'
        f'masked_pairwise_rank={new_m["pairwise_rank_accuracy"]:.6f}\n'
        f'masked_selected_beneficial={new_m["selected_beneficial_rate"]:.6f}\n'
        f'masked_selected_harmful={new_m["selected_harmful_rate"]:.6f}\n'
        f'masked_regret_reduction_vs_coarse={rr:.6f}\n'
    )
    latest = Path(cfg['project']['exp_root']) / 'PredictiveUtilityRetrieval' / cfg['experiment']['subgraph'] / a.stage / 'LATEST.json'
    latest.parent.mkdir(parents=True, exist_ok=True)
    write_json(latest, {
        'run_dir': str(rd), 'checkpoint': str(best_path), 'summary': str(rd / 'summary.json'),
        'variant': 'PredictiveUtilityRetrieval', 'stage': a.stage,
        'version': cfg['version'], 'implementation_revision': cfg.get('implementation_revision'),
    })

    print('============================================================', flush=True)
    print(f'BEST epoch={ck["epoch"]} beta={chosen_beta:g}', flush=True)
    print('FINAL masked coarse:', json.dumps(base_m, indent=2), flush=True)
    print('FINAL masked v0.9A:', json.dumps(new_m, indent=2), flush=True)
    print(f'MASKED regret reduction vs coarse = {rr:.4%}', flush=True)
    print(f'DECISION = {decision} checks={go_checks}', flush=True)
    print(f'SUMMARY {rd / "summary.json"}', flush=True)
    print('============================================================', flush=True)


if __name__ == '__main__':
    main()
