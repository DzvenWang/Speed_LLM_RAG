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
from presure.need_state import PredictiveNeedStateNet
from presure.need_gated import (
    RankAccumulator,
    activation_diagnostics,
    choose_support,
    gather_chosen_residual,
    instantiate_utility_a,
    load_latest,
    mae_views,
    quantile_threshold,
    relative_gains,
    utility_a_batch,
)
from presure.pipeline import load_artifacts, run_dir


def batches(ids, bs):
    ids = np.asarray(ids, np.int64)
    for a in range(0, len(ids), int(bs)):
        yield ids[a : a + int(bs)]


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


def view_mask_np(ds, ids, view):
    valid = np.asarray(ds.valid[ids], bool)
    if view == 'overall':
        return valid
    if view == 'masked':
        return valid & np.asarray(ds.direct[ids], bool)[:, None, :]
    if view == 'support':
        return valid & np.asarray(ds.support[ids], bool)[:, None, :]
    raise ValueError(view)


def score_split(
    utility_model, cfg, art, ds, ids, top4, need_prob, dev,
    collect_policy=False,
):
    """Evaluate frozen Utility-A and optionally collect arrays for policy calibration."""
    c = cfg['need_gated_utility_probe']
    uc = cfg['predictive_need_probe']['utility_model']
    nft = np.load(art / 'train_node_features.npy', mmap_mode='r')
    nodeq = np.load(art / ('val_node_features.npy' if ds.split == 'val' else 'train_node_features.npy'), mmap_mode='r')
    res = np.load(art / 'train_residual.npy', mmap_mode='r')
    base = np.load(art / ('val_base_pred.npy' if ds.split == 'val' else 'train_base_pred.npy'), mmap_mode='r')
    mean, std = cache_stats(cache_dir(cfg, 'development'))
    H = int(cfg['experiment']['history'])
    P = int(cfg['experiment']['horizon'])
    bs = int(c['eval_batch_size'])
    beneficial_threshold = float(uc['beneficial_threshold'])

    # Need thresholds are based only on deployment-safe q90 scores, separately per view.
    gate_fractions = [float(x) for x in c['gate_fractions']]
    gate_thresholds = {}
    for view in ['overall', 'masked', 'support']:
        vm = view_mask_np(ds, ids, view)
        gate_thresholds[view] = {
            f'top{int(round(100*f))}': quantile_threshold(need_prob, vm, f)
            for f in gate_fractions
        }

    acc = {
        view: {'all': RankAccumulator(), **{f'top{int(round(100*f))}': RankAccumulator() for f in gate_fractions}}
        for view in ['overall', 'masked', 'support']
    }

    coll = {k: [] for k in ['need', 'max_utility', 'chosen_rr', 'base', 'truth', 'valid', 'direct', 'support']} if collect_policy else None

    utility_model.eval()
    with torch.no_grad():
        for bi in batches(np.arange(len(ids)), bs):
            src = ids[bi]
            score, target, rr_raw, cm, valid = utility_a_batch(
                utility_model, nodeq, nft, res, base, ds.y, ds.valid, src,
                np.asarray(top4['ids'][bi]), np.asarray(top4['conf'][bi]), np.asarray(top4['mask'][bi]),
                mean, std, H, P, cfg, dev,
            )
            pi, max_score_bn = choose_support(score, cm)
            chosen_rr = gather_chosen_residual(rr_raw, pi)  # [B,P,N]
            max_score = max_score_bn.permute(0, 2, 1).contiguous()  # [B,P,N]
            np_need = np.asarray(need_prob[bi], np.float32)

            direct = np.asarray(ds.direct[src], bool)
            support = np.asarray(ds.support[src], bool)
            vm_np = np.asarray(ds.valid[src], bool)
            view_np = {
                'overall': vm_np,
                'masked': vm_np & direct[:, None, :],
                'support': vm_np & support[:, None, :],
            }
            for view in ['overall', 'masked', 'support']:
                base_cell = torch.as_tensor(view_np[view].transpose(0, 2, 1), device=dev, dtype=torch.bool)
                # transpose above is wrong shape [B,N,P] only when axes are reordered explicitly.
                base_cell = torch.as_tensor(np.transpose(view_np[view], (0, 2, 1)), device=dev, dtype=torch.bool)
                acc[view]['all'].update(score, target, cm, base_cell, beneficial_threshold)
                for f in gate_fractions:
                    key = f'top{int(round(100*f))}'
                    thr = gate_thresholds[view][key]
                    gm = view_np[view] & (np_need >= thr)
                    gcell = torch.as_tensor(np.transpose(gm, (0, 2, 1)), device=dev, dtype=torch.bool)
                    acc[view][key].update(score, target, cm, gcell, beneficial_threshold)

            if collect_policy:
                coll['need'].append(np_need.astype(np.float32))
                coll['max_utility'].append(max_score.cpu().numpy().astype(np.float32))
                coll['chosen_rr'].append(chosen_rr.cpu().numpy().astype(np.float32))
                coll['base'].append(np.asarray(base[src], np.float32))
                coll['truth'].append(np.asarray(ds.y[src], np.float32))
                coll['valid'].append(vm_np)
                coll['direct'].append(direct)
                coll['support'].append(support)

    summary = {
        'gate_thresholds_by_view': gate_thresholds,
        'views': {view: {k: a.summary() for k, a in vv.items()} for view, vv in acc.items()},
    }
    if not collect_policy:
        return summary, None
    arrays = {k: np.concatenate(v, axis=0) for k, v in coll.items()}
    return summary, arrays


def policy_metrics(arr, need_thr, utility_thr, alpha):
    gate = (
        np.asarray(arr['valid'], bool)
        & (np.asarray(arr['need'], np.float32) >= float(need_thr))
        & (np.asarray(arr['max_utility'], np.float32) >= float(utility_thr))
    )
    base = np.asarray(arr['base'], np.float32)
    truth = np.asarray(arr['truth'], np.float32)
    corrected = base + gate.astype(np.float32) * float(alpha) * np.asarray(arr['chosen_rr'], np.float32)
    bmae = mae_views(base, truth, arr['valid'], arr['direct'], arr['support'])
    cmae = mae_views(corrected, truth, arr['valid'], arr['direct'], arr['support'])
    gains = relative_gains(bmae, cmae)
    act = activation_diagnostics(base, truth, corrected, gate, arr['valid'], arr['direct'], arr['support'])
    return {
        'need_threshold': float(need_thr),
        'utility_threshold': float(utility_thr),
        'alpha': float(alpha),
        'base_mae': bmae,
        'corrected_mae': cmae,
        'relative_gain': gains,
        'activation': act,
    }


def calibrate_policy(arr, cfg):
    c = cfg['need_gated_utility_probe']['policy_calibration']
    valid = np.asarray(arr['valid'], bool)
    need = np.asarray(arr['need'], np.float32)
    util = np.asarray(arr['max_utility'], np.float32)
    weights = c['objective_weights']
    tol = float(c['view_harm_tolerance'])
    rows = []
    for nf in c['need_keep_fractions']:
        nth = quantile_threshold(need, valid, float(nf))
        ng = valid & (need >= nth)
        for uf in c['utility_keep_fractions']:
            uth = quantile_threshold(util, ng, float(uf))
            for alpha in c['alpha_grid']:
                m = policy_metrics(arr, nth, uth, float(alpha))
                g = m['relative_gain']
                obj = sum(float(weights[k]) * float(g[k] or 0.0) for k in ['overall', 'masked', 'support'])
                safe = all((g[k] is not None and float(g[k]) >= -tol) for k in ['overall', 'masked', 'support'])
                act = m['activation']['overall']['activation_rate']
                rows.append({
                    'need_keep_fraction': float(nf), 'utility_keep_fraction': float(uf),
                    'objective': float(obj), 'safe': bool(safe), 'overall_activation_rate': float(act),
                    **m,
                })
    # Explicit no-op policy makes calibration fail safe.
    noop = policy_metrics(arr, 1.0e9, 1.0e9, 0.0)
    rows.append({'need_keep_fraction': 0.0, 'utility_keep_fraction': 0.0, 'objective': 0.0, 'safe': True, 'overall_activation_rate': 0.0, **noop})
    safe_rows = [r for r in rows if r['safe']]
    best = max(safe_rows, key=lambda r: (r['objective'], -r['overall_activation_rate']))
    return best, rows


def decision(rank_summary, policy_val, state_summary, cfg):
    d = cfg['need_gated_utility_probe']['decision']
    masked = rank_summary['views']['masked']
    all_rank = float(masked['all']['pairwise_rank_accuracy'])
    top10_rank = float(masked['top10']['pairwise_rank_accuracy'])
    rank_gain = top10_rank - all_rank
    q90_auc = float(state_summary['state_tasks']['need_q90']['validation']['masked']['auroc'])
    pg = policy_val['relative_gain']
    harm = policy_val['activation']['masked']['harmful_fraction']
    rank_ok = top10_rank >= float(d['masked_top10_rank_min']) and rank_gain >= float(d['masked_top10_rank_gain_min'])
    policy_go = (
        float(pg['masked']) >= float(d['masked_policy_gain_go'])
        and float(pg['overall']) >= -float(d['view_harm_tolerance'])
        and float(pg['support']) >= -float(d['view_harm_tolerance'])
    )
    policy_cond = float(pg['masked']) >= float(d['masked_policy_gain_conditional'])
    harm_ok = harm is None or float(harm) <= float(d['masked_activated_harm_max'])
    need_ok = q90_auc >= float(d['need_q90_masked_auroc_min'])
    if need_ok and rank_ok and policy_go and harm_ok:
        status = 'GO_V08_NEED_GATED'
    elif need_ok and harm_ok and (rank_ok or policy_cond):
        status = 'CONDITIONAL_V08_NEED_GATED'
    else:
        status = 'NO_GO_V08_NEED_GATED'
    diag = {
        'need_q90_masked_auroc': q90_auc,
        'masked_all_rank_accuracy': all_rank,
        'masked_top10_rank_accuracy': top10_rank,
        'masked_top10_rank_gain': rank_gain,
        'validation_policy_relative_gain': pg,
        'validation_masked_activated_harmful_fraction': harm,
        'need_signal_pass': need_ok,
        'rank_concentration_pass': rank_ok,
        'policy_go_pass': policy_go,
        'policy_conditional_pass': policy_cond,
        'activated_harm_pass': harm_ok,
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
    exact_latest, exact_rd = load_latest(cfg['project']['exp_root'], 'PredictiveNeedProbe', cfg['experiment']['subgraph'], a.stage)
    state_latest, state_rd = load_latest(cfg['project']['exp_root'], 'PredictiveNeedStateProbe', cfg['experiment']['subgraph'], a.stage)
    exact_summary = json.loads((exact_rd / 'summary.json').read_text())
    state_summary = json.loads((state_rd / 'summary.json').read_text())

    utility_model, utility_ck = instantiate_utility_a(exact_rd / 'utility_A_best.pt', dev)
    q90_model, q90_ck = load_state_q90(state_rd, dev)

    rd = run_dir(cfg, 'NeedGatedUtilityProbe', a.stage, stamp(), a.seed)
    rd.mkdir(parents=True, exist_ok=True)
    lineage = {
        'variant': 'NeedGatedUtilityProbe', 'stage': a.stage, 'seed': a.seed,
        'implementation_revision': cfg.get('implementation_revision'),
        'reuse_only_no_model_training': True,
        'utility_A_source': str(exact_rd),
        'need_q90_source': str(state_rd),
        'policy_calibrated_on_train_utility_dev_only': True,
        'validation_future_used_for_diagnostic_only': True,
    }
    write_json(rd / 'RUN_MANIFEST.json', {**lineage, 'status': 'STARTED'})

    try:
        # Train-only calibration split from v0.7.1 Utility-A run.
        cal_ids = np.load(exact_rd / 'utility_dev_ids.npy').astype(np.int64)
        cal_top4 = load_top4(exact_rd, 'utility_dev')
        nft = np.load(art / 'train_node_features.npy', mmap_mode='r')
        need_cal = predict_need_q90(q90_model, nft, cal_ids, int(cfg['need_gated_utility_probe']['need_eval_batch_size']), dev)
        cal_rank, cal_arr = score_split(utility_model, cfg, art, tr, cal_ids, cal_top4, need_cal, dev, collect_policy=True)
        best_policy, policy_grid = calibrate_policy(cal_arr, cfg)
        write_json(rd / 'train_only_policy_calibration.json', {'best_policy': best_policy, 'grid': policy_grid})
        print('NEED-GATED CALIBRATION BEST', json.dumps(best_policy, sort_keys=True), flush=True)

        # Final validation is exactly the v0.7.2 state-probe subset, so q90 probabilities and Top-4 are aligned.
        state_split = json.loads((state_rd / 'split_manifest.json').read_text())
        val_ids = np.asarray(state_split['final_val'], np.int64)
        need_val = np.asarray(np.load(state_rd / 'prob_need_q90_final_val.npy', mmap_mode='r'), np.float32)[..., 1]
        val_top4 = load_state_top4(state_rd)
        val_rank, val_arr = score_split(utility_model, cfg, art, va, val_ids, val_top4, need_val, dev, collect_policy=True)
        write_json(rd / 'conditional_utility_by_need_gate.json', val_rank)

        policy_val = policy_metrics(
            val_arr,
            best_policy['need_threshold'], best_policy['utility_threshold'], best_policy['alpha']
        )
        write_json(rd / 'validation_policy_result.json', policy_val)
        print('NEED-GATED VALIDATION POLICY', json.dumps(policy_val, sort_keys=True), flush=True)

        status, diagnostics = decision(val_rank, policy_val, state_summary, cfg)
        summary = {
            'variant': 'NeedGatedUtilityProbe', 'stage': a.stage, 'seed': a.seed,
            'sources': {
                'v071_predictive_need_probe': str(exact_rd),
                'v072_predictive_need_state_probe': str(state_rd),
                'utility_A_checkpoint': str(exact_rd / 'utility_A_best.pt'),
                'need_q90_checkpoint': str(state_rd / 'need_q90_best.pt'),
            },
            'reference': {
                'v071_utility_A_original_validation': exact_summary['support_utility_validation']['A'],
                'v072_need_q90_masked': state_summary['state_tasks']['need_q90']['validation']['masked'],
            },
            'train_only_policy_calibration_best': best_policy,
            'validation_conditional_utility': val_rank,
            'validation_deployable_policy': policy_val,
            'v08_need_gated_recommendation': status,
            'decision_diagnostics': diagnostics,
            'interpretation': {
                'Need_q90': 'answers WHEN/WHERE external support is likely needed',
                'Utility_A': 'answers WHICH historical support appears useful without current-query future input',
                'conditional_rank': 'tests whether Utility-A becomes more reliable inside high-Need cells',
                'deployable_policy': 'Need threshold + Utility-A threshold + fixed residual alpha, all calibrated on Train-only utility_dev',
            },
            'test_evaluated': False,
        }
        write_json(rd / 'summary.json', summary)
        (rd / 'DECISION.txt').write_text(status + '\n' + json.dumps(diagnostics, indent=2, sort_keys=True) + '\n')

        rows = []
        for view, gates in val_rank['views'].items():
            for gate, met in gates.items():
                rows.append({'view': view, 'gate': gate, **met})
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with open(rd / 'need_gated_rank_summary.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)

        latest_dir = Path(cfg['project']['exp_root']) / 'NeedGatedUtilityProbe' / cfg['experiment']['subgraph'] / a.stage
        latest_dir.mkdir(parents=True, exist_ok=True)
        write_json(latest_dir / 'LATEST.json', {
            'run_dir': str(rd), 'summary': str(rd / 'summary.json'),
            'recommendation': status, 'implementation_revision': cfg.get('implementation_revision'),
        })
        write_json(rd / 'RUN_MANIFEST.json', {**lineage, 'status': 'FINISHED', 'recommendation': status})
        print('NEED-GATED UTILITY PROBE FINAL', status, flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f'NeedGatedUtilityProbe PASS run_dir={rd}', flush=True)
    except Exception as e:
        write_json(rd / 'RUN_MANIFEST.json', {**lineage, 'status': 'FAILED', 'error': repr(e)})
        raise


if __name__ == '__main__':
    main()
