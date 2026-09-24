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
from presure.data import CacheSplit, cache_stats
from presure.models import NodeUsefulnessNet
from presure.need_state import PredictiveNeedStateNet
from presure.need_gated import instantiate_utility_a, load_latest, quantile_threshold
from presure.probes import SupportUtilityProbeNet
from presure.utility_specialist import (
    CandidateMetricAccumulator,
    specialist_loss,
    true_q90_mask,
    two_expert_score,
    utility_specialist_batch,
)
from presure.pipeline import load_artifacts, run_dir


def batches(ids, bs, shuffle=False, seed=0):
    ids = np.asarray(ids, np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(ids)
    for a in range(0, len(ids), int(bs)):
        yield ids[a:a + int(bs)]


def t(a, dev, dtype=torch.float32):
    return torch.as_tensor(np.array(a, copy=True), device=dev, dtype=dtype)


def load_state_q90(state_rd, dev):
    p = state_rd / 'need_q90_best.pt'
    ck = torch.load(p, map_location=dev, weights_only=False)
    if ck.get('model_type') != 'predictive_need_state_probe_v072' or ck.get('task') != 'need_q90':
        raise RuntimeError(f'Need-q90 checkpoint mismatch: {p}')
    m = PredictiveNeedStateNet(**ck['model_args']).to(dev)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    return m, ck


def predict_need_q90(model, node_feat, ids, batch_size, dev):
    out = []
    model.eval()
    with torch.no_grad():
        for bi in batches(np.arange(len(ids)), batch_size):
            src = ids[bi]
            p = torch.softmax(model(t(node_feat[src], dev)), dim=-1)[..., 1]
            out.append(p.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0) if out else np.empty((0, 0, 0), np.float32)


def load_top4(rd, tag):
    return {
        'ids': np.load(rd / f'top4_ids_{tag}.npy', mmap_mode='r'),
        'conf': np.load(rd / f'top4_conf_{tag}.npy', mmap_mode='r'),
        'mask': np.load(rd / f'top4_mask_{tag}.npy', mmap_mode='r'),
    }


def load_state_top4(state_rd):
    return {
        'ids': np.load(state_rd / 'top4_ids_final_val.npy', mmap_mode='r'),
        'conf': np.load(state_rd / 'top4_conf_final_val.npy', mmap_mode='r'),
        'mask': np.load(state_rd / 'top4_mask_final_val.npy', mmap_mode='r'),
    }


def view_np(ds, ids, view):
    valid = np.asarray(ds.valid[ids], bool)
    if view == 'overall':
        return valid
    if view == 'masked':
        return valid & np.asarray(ds.direct[ids], bool)[:, None, :]
    if view == 'support':
        return valid & np.asarray(ds.support[ids], bool)[:, None, :]
    raise ValueError(view)


def specialist_model_from_utility_a(utility_ck, dev):
    model = SupportUtilityProbeNet(**utility_ck['model_args']).to(dev)
    model.load_state_dict(copy.deepcopy(utility_ck['state_dict']))
    return model


def batch_scores(model, cfg, art, ds, src, top4, top_idx, need_prob, dev, condition_on_need=True):
    node_mem = np.load(art / 'train_node_features.npy', mmap_mode='r')
    node_q = np.load(art / ('val_node_features.npy' if ds.split == 'val' else 'train_node_features.npy'), mmap_mode='r')
    residual = np.load(art / 'train_residual.npy', mmap_mode='r')
    base = np.load(art / ('val_base_pred.npy' if ds.split == 'val' else 'train_base_pred.npy'), mmap_mode='r')
    mean, std = cache_stats(cache_dir(cfg, 'development'))
    return utility_specialist_batch(
        model, node_q, node_mem, residual, base, ds.y, ds.valid, src,
        np.asarray(top4['ids'][top_idx]), np.asarray(top4['conf'][top_idx]), np.asarray(top4['mask'][top_idx]),
        np.asarray(need_prob[top_idx], np.float32), mean, std,
        int(cfg['experiment']['history']), int(cfg['experiment']['horizon']), cfg, dev,
        condition_on_need=condition_on_need,
    )


def train_specialist(mode, cfg, art, tr, ids_train, ids_dev, top_train, top_dev, need_train, need_dev, q90_raw, utility_ck, rd, dev, seed):
    c = cfg['need_conditioned_utility_specialist_probe']['training']
    model = specialist_model_from_utility_a(utility_ck, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c['lr']), weight_decay=float(c['weight_decay']))
    base_train = np.load(art / 'train_base_pred.npy', mmap_mode='r')
    best_score = -1e9
    best_epoch = 0
    wait = 0
    best_path = rd / f'{mode}_specialist_best.pt'

    for ep in range(1, int(c['epochs']) + 1):
        model.train()
        losses, ranks, counts = [], [], []
        for bi in batches(np.arange(len(ids_train)), int(c['batch_size']), True, seed + 7100 + ep):
            src = ids_train[bi]
            pred, target, cm, valid, _ = batch_scores(model, cfg, art, tr, src, top_train, bi, need_train, dev, True)
            hard_np = true_q90_mask(np.asarray(base_train[src]), np.asarray(tr.y[src]), q90_raw)
            hard = t(hard_np, dev, torch.bool)
            npb = t(np.asarray(need_train[bi], np.float32), dev)
            loss, rank, cls, racc, rn = specialist_loss(pred, target, cm, valid, hard, npb, cfg, mode)
            if rn == 0:
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(c['grad_clip']))
            opt.step()
            losses.append(float(loss.detach()))
            ranks.append(float(racc.detach()))
            counts.append(rn)

        dev_summary = evaluate_variants(
            {'specialist': model}, cfg, art, tr, ids_dev, top_dev, need_dev, q90_raw, dev,
            seed + 7300 + ep, generic_model=None, two_expert_cfg=None,
        )
        dm = dev_summary['variants']['specialist']['views']['overall']
        true_rank = float(dm['true_q90']['pairwise_rank_accuracy'])
        pred_rank = float(dm['pred_top10']['pairwise_rank_accuracy'])
        harm = float(dm['pred_top10']['selected_harmful_rate'])
        score = true_rank + 0.50 * pred_rank - 0.20 * harm
        print(
            f'SPECIALIST-{mode.upper()} epoch={ep:03d} '
            f'loss={np.mean(losses) if losses else 0.0:.5f} '
            f'train_rank={np.mean(ranks) if ranks else 0.0:.3f} '
            f'dev_true_q90_rank={true_rank:.3f} dev_pred_top10_rank={pred_rank:.3f} '
            f'dev_pred_top10_harm={harm:.3f}', flush=True,
        )
        if score > best_score:
            best_score, best_epoch, wait = score, ep, 0
            torch.save({
                'state_dict': model.state_dict(), 'model_args': utility_ck['model_args'],
                'model_type': 'need_conditioned_utility_specialist_v074', 'mode': mode,
                'epoch': ep, 'dev_score': score,
                'initialized_from': 'v0.7.1 Utility-A',
                'deployable_need_input': 'v0.7.2 predicted Need-q90 probability',
            }, best_path)
        else:
            wait += 1
        if wait >= int(c['patience']):
            break

    ck = torch.load(best_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, ck


def _subset_masks(ds, ids, need_prob, true_hard, cfg):
    c = cfg['need_conditioned_utility_specialist_probe']['evaluation']
    out = {}
    valid = np.asarray(ds.valid[ids], bool)
    for view in ['overall', 'masked', 'support']:
        vm = view_np(ds, ids, view)
        vv = {'all': vm, 'true_q90': vm & true_hard}
        for frac in c['predicted_need_keep_fractions']:
            key = f'pred_top{int(round(100 * float(frac)))}'
            th = quantile_threshold(need_prob, vm, float(frac))
            vv[key] = vm & (need_prob >= th)
        out[view] = vv
    return out


def evaluate_variants(models, cfg, art, ds, ids, top4, need_prob, q90_raw, dev, seed, generic_model=None, two_expert_cfg=None):
    c = cfg['need_conditioned_utility_specialist_probe']
    uc = cfg['predictive_need_probe']['utility_model']
    bs = int(c['evaluation']['eval_batch_size'])
    base = np.load(art / ('val_base_pred.npy' if ds.split == 'val' else 'train_base_pred.npy'), mmap_mode='r')
    true_hard_all = true_q90_mask(np.asarray(base[ids]), np.asarray(ds.y[ids]), q90_raw)
    masks = _subset_masks(ds, ids, need_prob, true_hard_all, cfg)
    variant_names = list(models.keys())
    if generic_model is not None and 'generic' not in variant_names:
        variant_names = ['generic'] + variant_names
    if two_expert_cfg is not None:
        variant_names = variant_names + ['two_expert']
    acc = {}
    for vi, v in enumerate(variant_names):
        acc[v] = {}
        for wi, view in enumerate(['overall', 'masked', 'support']):
            acc[v][view] = {}
            for si, sub in enumerate(masks[view]):
                acc[v][view][sub] = CandidateMetricAccumulator(
                    float(uc['beneficial_threshold']),
                    int(c['evaluation']['sample_points_per_batch']),
                    int(seed + 1000 * vi + 100 * wi + 7 * si),
                )

    all_models = {**models}
    if generic_model is not None:
        all_models['generic'] = generic_model
    for m in all_models.values():
        m.eval()

    with torch.no_grad():
        for bi in batches(np.arange(len(ids)), bs):
            src = ids[bi]
            # Target is identical for all variants. Generic uses zero Need; specialists use predicted Need-q90.
            specialist_scores = {}
            target = cm = valid = None
            for name, model in all_models.items():
                cond = name != 'generic'
                pred, tar, cmm, vv, _ = batch_scores(model, cfg, art, ds, src, top4, bi, need_prob, dev, cond)
                specialist_scores[name] = pred
                if target is None:
                    target, cm, valid = tar, cmm, vv
            if two_expert_cfg is not None:
                specialist_scores['two_expert'] = two_expert_score(
                    specialist_scores['generic'], specialist_scores['hard'],
                    t(np.asarray(need_prob[bi], np.float32), dev),
                    two_expert_cfg['need_threshold'], two_expert_cfg['hard_mix_weight'],
                    float(uc['classification_temperature']),
                )

            for view in ['overall', 'masked', 'support']:
                for sub, full_mask in masks[view].items():
                    local = np.asarray(full_mask[bi], bool)
                    cell = torch.as_tensor(np.transpose(local, (0, 2, 1)), device=dev, dtype=torch.bool)
                    for name in variant_names:
                        acc[name][view][sub].update(specialist_scores[name], target, cm, cell)

    return {
        'variants': {
            name: {'views': {view: {sub: a.summary() for sub, a in subs.items()} for view, subs in views.items()}}
            for name, views in acc.items()
        },
        'privileged_true_q90_subset_is_evaluation_only': True,
    }


def calibrate_two_expert(generic_model, hard_model, cfg, art, tr, ids_dev, top_dev, need_dev, q90_raw, dev, seed):
    c = cfg['need_conditioned_utility_specialist_probe']['two_expert']
    valid = np.asarray(tr.valid[ids_dev], bool)
    rows = []
    for frac in c['need_keep_fractions']:
        nth = quantile_threshold(need_dev, valid, float(frac))
        for mix in c['hard_mix_weights']:
            cc = {'need_threshold': float(nth), 'hard_mix_weight': float(mix), 'need_keep_fraction': float(frac)}
            sm = evaluate_variants(
                {'hard': hard_model}, cfg, art, tr, ids_dev, top_dev, need_dev, q90_raw, dev,
                seed + int(1000 * frac) + int(100 * mix), generic_model=generic_model, two_expert_cfg=cc,
            )
            v = sm['variants']['two_expert']['views']
            # Train-only calibration may use direct/support masks as weighting, never as inference features.
            r_true = 0.25 * v['overall']['true_q90']['pairwise_rank_accuracy'] + 0.50 * v['masked']['true_q90']['pairwise_rank_accuracy'] + 0.25 * v['support']['true_q90']['pairwise_rank_accuracy']
            r_pred = 0.25 * v['overall']['pred_top10']['pairwise_rank_accuracy'] + 0.50 * v['masked']['pred_top10']['pairwise_rank_accuracy'] + 0.25 * v['support']['pred_top10']['pairwise_rank_accuracy']
            harm = v['masked']['pred_top10']['selected_harmful_rate']
            objective = float(r_true + 0.5 * r_pred - 0.2 * harm)
            rows.append({**cc, 'objective': objective, 'masked_true_q90_rank': v['masked']['true_q90']['pairwise_rank_accuracy'], 'masked_pred_top10_rank': v['masked']['pred_top10']['pairwise_rank_accuracy'], 'masked_pred_top10_harm': harm})
    best = max(rows, key=lambda x: x['objective'])
    return best, rows


def decision(summary, cfg):
    d = cfg['need_conditioned_utility_specialist_probe']['decision']
    generic = summary['final_validation']['variants']['generic']['views']['masked']
    candidates = ['hard', 'weighted', 'two_expert']
    best_name = max(candidates, key=lambda n: summary['final_validation']['variants'][n]['views']['masked']['pred_top10']['pairwise_rank_accuracy'])
    best = summary['final_validation']['variants'][best_name]['views']['masked']
    generic_top10 = float(generic['pred_top10']['pairwise_rank_accuracy'])
    top10 = float(best['pred_top10']['pairwise_rank_accuracy'])
    true_rank = float(best['true_q90']['pairwise_rank_accuracy'])
    benef = float(best['pred_top10']['selected_beneficial_rate'])
    harm = float(best['pred_top10']['selected_harmful_rate'])
    gain = top10 - generic_top10
    go = (
        top10 >= float(d['masked_pred_top10_rank_go']) and
        true_rank >= float(d['masked_true_q90_rank_go']) and
        gain >= float(d['rank_gain_over_generic_go']) and
        benef >= float(d['selected_beneficial_go']) and
        harm <= float(d['selected_harmful_go'])
    )
    conditional = (
        top10 >= float(d['masked_pred_top10_rank_conditional']) and
        gain >= float(d['rank_gain_over_generic_conditional']) and
        harm <= float(d['selected_harmful_conditional'])
    )
    status = 'GO_V08_HARD_SPECIALIST' if go else ('CONDITIONAL_V08_HARD_SPECIALIST' if conditional else 'NO_GO_V08_HARD_SPECIALIST')
    diag = {
        'best_variant': best_name,
        'generic_masked_pred_top10_rank': generic_top10,
        'best_masked_pred_top10_rank': top10,
        'rank_gain_over_generic': gain,
        'best_masked_true_q90_rank': true_rank,
        'best_masked_pred_top10_selected_beneficial_rate': benef,
        'best_masked_pred_top10_selected_harmful_rate': harm,
        'thresholds': d,
    }
    return status, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='development', choices=['development'])
    ap.add_argument('--seed', type=int, default=10)
    a = ap.parse_args()
    cfg = load_yaml(ROOT / 'configs/presure.yaml')
    seed_all(a.seed)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    art = load_artifacts(cfg, a.stage)
    cache = cache_dir(cfg, a.stage)
    tr = CacheSplit(cache, 'train')
    va = CacheSplit(cache, 'val')

    _, exact_rd = load_latest(cfg['project']['exp_root'], 'PredictiveNeedProbe', cfg['experiment']['subgraph'], a.stage)
    _, state_rd = load_latest(cfg['project']['exp_root'], 'PredictiveNeedStateProbe', cfg['experiment']['subgraph'], a.stage)
    _, gate_rd = load_latest(cfg['project']['exp_root'], 'NeedGatedUtilityProbe', cfg['experiment']['subgraph'], a.stage)
    state_summary = json.loads((state_rd / 'summary.json').read_text())
    q90_raw = np.asarray(state_summary['train_only_thresholds']['q90'], np.float32)

    generic_model, utility_ck = instantiate_utility_a(exact_rd / 'utility_A_best.pt', dev)
    q90_model, _ = load_state_q90(state_rd, dev)

    ids_train = np.load(exact_rd / 'utility_train_ids.npy').astype(np.int64)
    ids_dev = np.load(exact_rd / 'utility_dev_ids.npy').astype(np.int64)
    top_train = load_top4(exact_rd, 'utility_train')
    top_dev = load_top4(exact_rd, 'utility_dev')
    nft = np.load(art / 'train_node_features.npy', mmap_mode='r')
    need_train = predict_need_q90(q90_model, nft, ids_train, int(cfg['need_conditioned_utility_specialist_probe']['evaluation']['need_eval_batch_size']), dev)
    need_dev = predict_need_q90(q90_model, nft, ids_dev, int(cfg['need_conditioned_utility_specialist_probe']['evaluation']['need_eval_batch_size']), dev)

    rd = run_dir(cfg, 'NeedConditionedUtilitySpecialistProbe', a.stage, stamp(), a.seed)
    rd.mkdir(parents=True, exist_ok=True)
    lineage = {
        'variant': 'NeedConditionedUtilitySpecialistProbe', 'stage': a.stage, 'seed': a.seed,
        'implementation_revision': cfg.get('implementation_revision'),
        'utility_A_initialization': str(exact_rd / 'utility_A_best.pt'),
        'need_q90_source': str(state_rd),
        'v073_reference': str(gate_rd),
        'true_q90_used_only_as_train_supervision_and_privileged_validation_subset': True,
        'deployable_models_use_predicted_need_q90_only': True,
        'validation_future_used_for_reporting_only': True,
    }
    write_json(rd / 'RUN_MANIFEST.json', {**lineage, 'status': 'STARTED'})

    try:
        hard_model, hard_ck = train_specialist('hard', cfg, art, tr, ids_train, ids_dev, top_train, top_dev, need_train, need_dev, q90_raw, utility_ck, rd, dev, a.seed)
        weighted_model, weighted_ck = train_specialist('weighted', cfg, art, tr, ids_train, ids_dev, top_train, top_dev, need_train, need_dev, q90_raw, utility_ck, rd, dev, a.seed + 101)

        two_best, two_grid = calibrate_two_expert(generic_model, hard_model, cfg, art, tr, ids_dev, top_dev, need_dev, q90_raw, dev, a.seed + 2200)
        write_json(rd / 'two_expert_train_only_calibration.json', {'best': two_best, 'grid': two_grid})
        print('SPECIALIST TWO-EXPERT CALIBRATION BEST', json.dumps(two_best, sort_keys=True), flush=True)

        state_split = json.loads((state_rd / 'split_manifest.json').read_text())
        val_ids = np.asarray(state_split['final_val'], np.int64)
        need_val = np.asarray(np.load(state_rd / 'prob_need_q90_final_val.npy', mmap_mode='r'), np.float32)[..., 1]
        top_val = load_state_top4(state_rd)
        final = evaluate_variants(
            {'hard': hard_model, 'weighted': weighted_model}, cfg, art, va, val_ids, top_val, need_val, q90_raw, dev,
            a.seed + 5000, generic_model=generic_model, two_expert_cfg=two_best,
        )
        write_json(rd / 'specialist_validation_summary.json', final)

        summary = {
            'variant': 'NeedConditionedUtilitySpecialistProbe',
            'stage': a.stage, 'seed': a.seed,
            'sources': {
                'v071_utility_A': str(exact_rd),
                'v072_need_q90': str(state_rd),
                'v073_need_gated_reference': str(gate_rd),
            },
            'training': {
                'hard_specialist_best_epoch': int(hard_ck['epoch']),
                'weighted_specialist_best_epoch': int(weighted_ck['epoch']),
                'true_q90_threshold_source': 'v0.7.2 Train-only horizon-specific q90',
                'future_usage': 'Train utility supervision only',
                'specialist_inputs': 'deployment-safe query/support context + historical residual + M3 confidence + predicted Need-q90 probability',
            },
            'two_expert_train_only_calibration': two_best,
            'final_validation': final,
            'test_evaluated': False,
        }
        status, diag = decision(summary, cfg)
        summary['v08_hard_specialist_recommendation'] = status
        summary['decision_diagnostics'] = diag
        write_json(rd / 'summary.json', summary)
        (rd / 'DECISION.txt').write_text(status + '\n' + json.dumps(diag, indent=2, sort_keys=True) + '\n')

        rows = []
        for variant, obj in final['variants'].items():
            for view, subsets in obj['views'].items():
                for subset, met in subsets.items():
                    rows.append({'variant': variant, 'view': view, 'subset': subset, **met})
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with open(rd / 'specialist_comparison.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

        latest = Path(cfg['project']['exp_root']) / 'NeedConditionedUtilitySpecialistProbe' / cfg['experiment']['subgraph'] / a.stage
        latest.mkdir(parents=True, exist_ok=True)
        write_json(latest / 'LATEST.json', {'run_dir': str(rd), 'summary': str(rd / 'summary.json'), 'recommendation': status, 'implementation_revision': cfg.get('implementation_revision')})
        write_json(rd / 'RUN_MANIFEST.json', {**lineage, 'status': 'FINISHED', 'recommendation': status})
        print('NEED-CONDITIONED UTILITY SPECIALIST FINAL', status, flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f'NeedConditionedUtilitySpecialistProbe PASS run_dir={rd}', flush=True)
    except Exception as e:
        write_json(rd / 'RUN_MANIFEST.json', {**lineage, 'status': 'FAILED', 'error': repr(e)})
        raise


if __name__ == '__main__':
    main()
