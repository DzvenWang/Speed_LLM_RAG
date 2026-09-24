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
from presure.need_state import (
    PredictiveNeedStateNet,
    class_weights_from_counts,
    confusion_matrix_np,
    macro_f1_from_cm,
    balanced_accuracy_from_cm,
    accuracy_from_cm,
    binary_metrics,
    top_fraction_recall_lift,
)
from presure.probes import exact_spearman_sample, pearson, binary_auroc
from presure.pipeline import load_artifacts, load_m1, load_m3, run_dir


def t(a, dev, dtype=torch.float32):
    return torch.as_tensor(np.array(a, copy=True), device=dev, dtype=dtype)


def batches(ids, bs, shuffle=False, seed=0):
    ids = np.asarray(ids, np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(ids)
    for a in range(0, len(ids), int(bs)):
        yield ids[a : a + int(bs)]


def _sense(x):
    return np.asarray(x[..., -NODE_SENSE_DIM:], np.float32)


def _split_ids(n, cfg, seed):
    c = cfg['predictive_need_state_probe']['split']
    counts = [int(c['state_train_queries']), int(c['state_dev_queries'])]
    if sum(counts) > n:
        raise RuntimeError(f'Need-State split needs {sum(counts)} Train queries but only {n} exist')
    ids = np.arange(n, dtype=np.int64)
    np.random.default_rng(seed + 2701).shuffle(ids)
    return {
        'state_train': np.sort(ids[: counts[0]]),
        'state_dev': np.sort(ids[counts[0] : counts[0] + counts[1]]),
    }


def _final_val_ids(n, cfg, seed):
    k = min(int(cfg['predictive_need_state_probe']['split']['final_val_queries']), int(n))
    ids = np.arange(n, dtype=np.int64)
    np.random.default_rng(seed + 2702).shuffle(ids)
    return np.sort(ids[:k])


def _error(ds, base, ids):
    return np.asarray(ds.y[ids], np.float32) - np.asarray(base[ids], np.float32)


def derive_train_thresholds(tr, base, ids, cfg):
    """Derive every diagnostic threshold from Train only, independently per horizon."""
    P = int(cfg['experiment']['horizon'])
    qc = cfg['predictive_need_state_probe']['thresholds']
    quantiles = sorted(set([
        float(qc['magnitude_q_low']),
        float(qc['need_q75']),
        float(qc['magnitude_q_high']),
        float(qc['need_q90']),
    ]))
    out = {f'q{int(round(q*100)):02d}': [] for q in quantiles}
    centers = []
    signed_pos_rate_q75 = []
    for h in range(P):
        yy = np.asarray(tr.y[ids, h, :], np.float32)
        bb = np.asarray(base[ids, h, :], np.float32)
        vv = np.asarray(tr.valid[ids, h, :], bool)
        ae = np.abs((yy - bb)[vv]).astype(np.float32)
        if len(ae) == 0:
            raise RuntimeError(f'No valid cells for threshold derivation at horizon {h}')
        vals = {q: float(np.quantile(ae, q)) for q in quantiles}
        for q in quantiles:
            out[f'q{int(round(q*100)):02d}'].append(vals[q])
        qlo = vals[float(qc['magnitude_q_low'])]
        qhi = vals[float(qc['magnitude_q_high'])]
        bands = [ae < qlo, (ae >= qlo) & (ae < qhi), ae >= qhi]
        cc = []
        for m in bands:
            if np.any(m):
                cc.append(float(np.median(ae[m])))
            else:
                cc.append(float(np.median(ae)))
        centers.append(cc)
        ee = (yy - bb)[vv]
        q75 = vals[float(qc['need_q75'])]
        hard = np.abs(ee) >= q75
        signed_pos_rate_q75.append(float(np.mean(ee[hard] >= 0)) if np.any(hard) else 0.5)
    out['magnitude_band_centers_raw'] = centers
    out['positive_rate_in_q75_hard_cells'] = signed_pos_rate_q75
    out['source'] = 'Train state_train split only; horizon-specific absolute PatchSTG error quantiles'
    return out


def _thr(thresholds, q, dev=None):
    arr = np.asarray(thresholds[f'q{int(round(float(q)*100)):02d}'], np.float32)[None, :, None]
    if dev is None:
        return arr
    return t(arr, dev)


def make_task_target(task, err, valid, thresholds, cfg):
    """Return integer target [B,P,N] and eligible mask [B,P,N]."""
    e = np.asarray(err, np.float32)
    v = np.asarray(valid, bool)
    ae = np.abs(e)
    tc = cfg['predictive_need_state_probe']['thresholds']
    q50 = _thr(thresholds, tc['magnitude_q_low'])
    q75 = _thr(thresholds, tc['need_q75'])
    q80 = _thr(thresholds, tc['magnitude_q_high'])
    q90 = _thr(thresholds, tc['need_q90'])
    if task == 'need_q75':
        return (ae >= q75).astype(np.int64), v
    if task == 'need_q90':
        return (ae >= q90).astype(np.int64), v
    if task == 'direction_q75':
        return (e >= 0).astype(np.int64), v & (ae >= q75)
    if task == 'direction_q90':
        return (e >= 0).astype(np.int64), v & (ae >= q90)
    if task == 'magnitude_3':
        y = np.zeros(e.shape, np.int64)
        y[(ae >= q50) & (ae < q80)] = 1
        y[ae >= q80] = 2
        return y, v
    if task == 'signed_state_q75':
        y = np.zeros(e.shape, np.int64)  # 0=no hard need, 1=positive, 2=negative
        hard = ae >= q75
        y[hard & (e >= 0)] = 1
        y[hard & (e < 0)] = 2
        return y, v
    raise ValueError(task)


def _num_classes(task):
    return 3 if task in {'magnitude_3', 'signed_state_q75'} else 2


def task_class_counts(task, tr, base, ids, thresholds, cfg):
    c = cfg['predictive_need_state_probe']['model']
    counts = np.zeros(_num_classes(task), np.int64)
    for bi in batches(ids, int(c['eval_batch_size'])):
        err = _error(tr, base, bi)
        y, use = make_task_target(task, err, np.asarray(tr.valid[bi], bool), thresholds, cfg)
        if np.any(use):
            counts += np.bincount(y[use], minlength=len(counts))
    return counts


def _view_mask(ds, ids, view, P):
    if view == 'overall':
        return np.ones((len(ids), P, np.asarray(ds.valid[ids]).shape[-1]), bool)
    if view == 'masked':
        return np.asarray(ds.direct[ids], bool)[:, None, :].repeat(P, axis=1)
    if view == 'support':
        return np.asarray(ds.support[ids], bool)[:, None, :].repeat(P, axis=1)
    raise ValueError(view)


def _multiclass_metrics(prob, target, eligible, num_classes):
    if not np.any(eligible):
        return {'count': 0, 'accuracy': None, 'balanced_accuracy': None, 'macro_f1': None, 'confusion_matrix': []}
    pred = np.asarray(prob).argmax(-1)
    cm = confusion_matrix_np(target[eligible], pred[eligible], num_classes)
    return {
        'count': int(eligible.sum()),
        'accuracy': accuracy_from_cm(cm),
        'balanced_accuracy': balanced_accuracy_from_cm(cm),
        'macro_f1': macro_f1_from_cm(cm),
        'confusion_matrix': cm.tolist(),
    }


def evaluate_task(task, prob, err, ds, ids, thresholds, cfg, seed):
    P = int(cfg['experiment']['horizon'])
    target, eligible = make_task_target(task, err, np.asarray(ds.valid[ids], bool), thresholds, cfg)
    out = {}
    for view in ['overall', 'masked', 'support']:
        use = eligible & _view_mask(ds, ids, view, P)
        if _num_classes(task) == 2:
            score = np.asarray(prob[..., 1], np.float32)
            met = binary_metrics(
                score[use], target[use] > 0,
                threshold=0.5,
                max_points=int(cfg['predictive_need_state_probe']['model']['metric_sample_points']),
                seed=seed + {'overall': 1, 'masked': 2, 'support': 3}[view],
            )
            met['count'] = int(use.sum())
            met['top10'] = top_fraction_recall_lift(score[use], target[use] > 0, 0.10)
            met['top20'] = top_fraction_recall_lift(score[use], target[use] > 0, 0.20)
            out[view] = met
        else:
            out[view] = _multiclass_metrics(prob, target, use, _num_classes(task))
    return out


def _dev_score(task, metrics):
    m = metrics['overall']
    if _num_classes(task) == 2:
        auc = m.get('auroc')
        return -(float(auc) if auc is not None else 0.0)
    return -float(m.get('macro_f1') or 0.0)


def train_one_task(task, cfg, stage, seed, art, tr, va, ids, val_ids, thresholds, rd, dev):
    c = cfg['predictive_need_state_probe']['model']
    nft = np.load(art / 'train_node_features.npy', mmap_mode='r')
    nfv = np.load(art / 'val_node_features.npy', mmap_mode='r')
    bt = np.load(art / 'train_base_pred.npy', mmap_mode='r')
    bv = np.load(art / 'val_base_pred.npy', mmap_mode='r')
    H = int(cfg['experiment']['history'])
    P = int(cfg['experiment']['horizon'])
    args = dict(
        din=nft.shape[-1], N=nft.shape[1], P=P, base_start=4 * H,
        num_classes=_num_classes(task), hidden=int(c['hidden']),
        demb=int(c['embedding_dim']), node_dim=int(c['node_embedding_dim']),
        horizon_dim=int(c['horizon_embedding_dim']), dropout=float(c['dropout']),
    )
    counts = task_class_counts(task, tr, bt, ids['state_train'], thresholds, cfg)
    weights = t(class_weights_from_counts(counts), dev)
    seed_all(seed + 4100 + sum(map(ord, task)))
    model = PredictiveNeedStateNet(**args).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c['lr']), weight_decay=float(c['weight_decay']))
    best = 1e99
    wait = 0
    bestp = rd / f'{task}_best.pt'
    for ep in range(1, int(c['epochs']) + 1):
        model.train()
        losses = []
        for bi in batches(ids['state_train'], int(c['batch_size']), True, seed + 4200 + ep):
            x = t(nft[bi], dev)
            err = _error(tr, bt, bi)
            yy, use = make_task_target(task, err, np.asarray(tr.valid[bi], bool), thresholds, cfg)
            logits = model(x)
            use_t = t(use, dev, torch.bool)
            y_t = t(yy, dev, torch.long)
            if not torch.any(use_t):
                continue
            loss = torch.nn.functional.cross_entropy(logits[use_t], y_t[use_t], weight=weights)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(c['grad_clip']))
            opt.step()
            losses.append(float(loss.detach()))
        # Dev predictions and selection metric; still Train-only queries.
        dev_prob = predict_task(model, nft, ids['state_dev'], c, dev)
        dev_err = _error(tr, bt, ids['state_dev'])
        dev_metrics = evaluate_task(task, dev_prob, dev_err, tr, ids['state_dev'], thresholds, cfg, seed + ep)
        score = _dev_score(task, dev_metrics)
        primary = dev_metrics['overall'].get('auroc') if _num_classes(task) == 2 else dev_metrics['overall'].get('macro_f1')
        print(f'STATE-{task} epoch={ep:03d} loss={np.mean(losses) if losses else 0:.5f} dev_primary={primary}', flush=True)
        if score < best:
            best = score
            wait = 0
            torch.save({
                'state_dict': model.state_dict(), 'model_args': args,
                'model_type': 'predictive_need_state_probe_v072', 'task': task,
                'epoch': ep, 'dev_score': score, 'class_counts': counts.tolist(),
                'class_weights': weights.detach().cpu().tolist(),
            }, bestp)
        else:
            wait += 1
        if wait >= int(c['patience']):
            break
    ck = torch.load(bestp, map_location=dev, weights_only=False)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    prob = predict_task(model, nfv, val_ids, c, dev)
    path = rd / f'prob_{task}_final_val.npy'
    np.save(path, prob.astype(np.float16))
    val_err = _error(va, bv, val_ids)
    metrics = evaluate_task(task, prob, val_err, va, val_ids, thresholds, cfg, seed + 4600)
    summary = {
        'task': task,
        'model_type': ck['model_type'],
        'best_epoch': int(ck['epoch']),
        'class_counts_train': ck['class_counts'],
        'class_weights': ck['class_weights'],
        'validation': metrics,
        'probability_file': str(path),
        'deployable_inputs_only': True,
        'validation_future_used_for_diagnostic_only': True,
    }
    write_json(rd / f'{task}_summary.json', summary)
    print('STATE FINAL', task, json.dumps(metrics, sort_keys=True), flush=True)
    del model, prob, dev_prob
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return summary


def predict_task(model, node_feat, ids, c, dev):
    out = []
    model.eval()
    with torch.no_grad():
        for bi in batches(ids, int(c['eval_batch_size'])):
            p = torch.softmax(model(t(node_feat[bi], dev)), dim=-1).cpu().numpy()
            out.append(p.astype(np.float32))
    return np.concatenate(out, axis=0)


def hard_error_group_analysis(cfg, thresholds, err, va, val_ids, rd):
    """Profile predicted state scores by Train-defined true error groups."""
    tc = cfg['predictive_need_state_probe']['thresholds']
    q50 = _thr(thresholds, tc['magnitude_q_low'])
    q75 = _thr(thresholds, tc['need_q75'])
    q90 = _thr(thresholds, tc['need_q90'])
    ae = np.abs(err)
    groups = {
        'Q00_Q50': ae < q50,
        'Q50_Q75': (ae >= q50) & (ae < q75),
        'Q75_Q90': (ae >= q75) & (ae < q90),
        'Q90_Q100': ae >= q90,
    }
    need75 = np.load(rd / 'prob_need_q75_final_val.npy', mmap_mode='r')[..., 1].astype(np.float32)
    need90 = np.load(rd / 'prob_need_q90_final_val.npy', mmap_mode='r')[..., 1].astype(np.float32)
    dir75 = np.load(rd / 'prob_direction_q75_final_val.npy', mmap_mode='r')[..., 1].astype(np.float32)
    out = {}
    for view in ['overall', 'masked', 'support']:
        vm = _view_mask(va, val_ids, view, int(cfg['experiment']['horizon'])) & np.asarray(va.valid[val_ids], bool)
        vv = {}
        for name, gm in groups.items():
            m = vm & gm
            if not np.any(m):
                vv[name] = {'count': 0}
                continue
            item = {
                'count': int(m.sum()),
                'true_abs_error_mean': float(ae[m].mean()),
                'pred_need_q75_mean': float(need75[m].mean()),
                'pred_need_q90_mean': float(need90[m].mean()),
            }
            # Direction is meaningful mainly in large-error groups, but report all for monotonic inspection.
            item['direction_accuracy_from_q75_model'] = float(np.mean((dir75[m] >= 0.5) == (err[m] >= 0)))
            vv[name] = item
        out[view] = vv
    write_json(rd / 'hard_error_group_summary.json', out)
    return out


def _load_embeddings(art):
    a = art / 'train_node_embeddings_v05.npy'
    b = art / 'val_node_embeddings_v05.npy'
    if not a.exists() or not b.exists():
        raise RuntimeError('Need-State probe requires cached M1 node embeddings from completed v0.7 M3 run')
    return np.load(a, mmap_mode='r'), np.load(b, mmap_mode='r')


def _m3_top4(m3, zmem, zq, node_mem, node_q, cand, ids, cfg, dev):
    K = min(int(cfg['node_routing']['utility_candidate_k']), cand['indices'].shape[1])
    L = int(cfg['node_routing']['m4_top_k'])
    ci = np.asarray(cand['indices'][ids, :K], np.int64)
    sim = np.asarray(cand['scores'][ids, :K], np.float32)
    q = t(zq[ids], dev)
    c = t(zmem[ci], dev)
    qs = t(_sense(node_q[ids]), dev)
    cs = t(_sense(node_mem[ci]), dev)
    with torch.no_grad():
        logits, score, _ = m3(q, c, t(sim, dev), qs, cs)
    p = torch.softmax(logits, -1)[..., 2]
    combined = score + torch.log(p + 1e-6)
    vals, loc = torch.topk(combined, min(L, K), dim=1)
    loc_bn = loc.permute(0, 2, 1).contiguous()
    p_bn = p.permute(0, 2, 1)
    psel = torch.gather(p_bn, 2, loc_bn)
    conf = torch.softmax(vals, dim=1).permute(0, 2, 1).contiguous()
    mask = psel >= float(cfg['model']['m3'].get('router_use_threshold', cfg['model']['m3']['use_threshold']))
    cids = t(ci, dev, torch.long)
    gid = torch.gather(cids[:, None, :].expand(-1, loc_bn.shape[1], -1), 2, loc_bn).cpu().numpy().astype(np.int32)
    return gid, conf.cpu().numpy().astype(np.float16), mask.cpu().numpy().astype(bool)


def prepare_final_top4(cfg, stage, art, val_ids, rd, dev):
    m3, _, _ = load_m3(cfg, stage, dev)
    m3.eval()
    zt, zv = _load_embeddings(art)
    nft = np.load(art / 'train_node_features.npy', mmap_mode='r')
    nfv = np.load(art / 'val_node_features.npy', mmap_mode='r')
    vc = np.load(art / 'val_candidates.npz')
    bs = int(cfg['predictive_need_state_probe']['utility']['selection_batch_size'])
    N = nft.shape[1]
    L = int(cfg['node_routing']['m4_top_k'])
    gp = rd / 'top4_ids_final_val.npy'
    cp = rd / 'top4_conf_final_val.npy'
    mp = rd / 'top4_mask_final_val.npy'
    gmm = np.lib.format.open_memmap(gp, mode='w+', dtype=np.int32, shape=(len(val_ids), N, L))
    cmm = np.lib.format.open_memmap(cp, mode='w+', dtype=np.float16, shape=(len(val_ids), N, L))
    mmm = np.lib.format.open_memmap(mp, mode='w+', dtype=np.bool_, shape=(len(val_ids), N, L))
    for bi in batches(np.arange(len(val_ids)), bs):
        src = val_ids[bi]
        g, cf, mm = _m3_top4(m3, zt, zv, nft, nfv, vc, src, cfg, dev)
        gmm[bi] = g
        cmm[bi] = cf
        mmm[bi] = mm
    del gmm, cmm, mmm
    out = {'ids': str(gp), 'conf': str(cp), 'mask': str(mp)}
    write_json(rd / 'top4_state_probe_manifest.json', out)
    return out


def _gather_support_residual(res, gids):
    # residual memory [T,P,N], gids [B,N,L] -> [B,N,L,P]
    N = gids.shape[1]
    ni = np.arange(N, dtype=np.int64)[None, :, None]
    return np.asarray(res)[gids, :, ni].astype(np.float32, copy=False)


def utility_from_need(need_norm, rr_norm, candidate_mask, cfg):
    """Analytic best-alpha relative utility for a supplied need estimate."""
    c = cfg['predictive_need_probe']['utility_model']
    n = need_norm.permute(0, 2, 1)[:, :, None, :]
    base = torch.abs(n)
    best = torch.full_like(rr_norm, -1e9)
    for a in c['alpha_grid']:
        gain = base - torch.abs(n - float(a) * rr_norm)
        best = torch.maximum(best, gain)
    rel = best / (base + float(c['gain_floor']))
    rel = torch.clamp(rel, -float(c['utility_clip']), float(c['utility_clip']))
    return rel.masked_fill(~candidate_mask[:, :, :, None], -float(c['utility_clip']))


def _utility_metric_accumulator_init():
    return dict(top1_ok=0, top1_n=0, rank_ok=0, rank_n=0, benef=0, benef_n=0, sample_s=[], sample_y=[])


def _utility_metric_update(acc, score, target, cm, valid, cfg, rng):
    c = cfg['predictive_need_probe']['utility_model']
    vm = valid.bool().permute(0, 2, 1)
    use = cm[:, :, :, None] & vm[:, :, None, :]
    masked_score = score.masked_fill(~use, -1e6)
    masked_tar = target.masked_fill(~use, -1e6)
    pi = masked_score.argmax(2)
    ti = masked_tar.argmax(2)
    ok = (use.sum(2) > 0) & vm
    acc['top1_ok'] += int(((pi == ti) & ok).sum().cpu())
    acc['top1_n'] += int(ok.sum().cpu())
    chosen = torch.gather(target, 2, pi[:, :, None, :]).squeeze(2)
    acc['benef'] += int(((chosen > float(c['beneficial_threshold'])) & ok).sum().cpu())
    acc['benef_n'] += int(ok.sum().cpu())
    hi = masked_tar.argmax(2)
    lo = target.masked_fill(~use, 1e6).argmin(2)
    sp = torch.gather(score, 2, hi[:, :, None, :]).squeeze(2)
    sn = torch.gather(score, 2, lo[:, :, None, :]).squeeze(2)
    tp = torch.gather(target, 2, hi[:, :, None, :]).squeeze(2)
    tn = torch.gather(target, 2, lo[:, :, None, :]).squeeze(2)
    en = (use.sum(2) >= 2) & (tp - tn > float(c['rank_min_gap'])) & vm
    acc['rank_ok'] += int(((sp > sn) & en).sum().cpu())
    acc['rank_n'] += int(en.sum().cpu())
    pp = score[use].detach().cpu().numpy()
    yy = target[use].detach().cpu().numpy()
    take = min(len(pp), int(c['sample_points_per_batch']))
    if take > 0:
        ii = rng.choice(len(pp), size=take, replace=False) if len(pp) > take else np.arange(len(pp))
        acc['sample_s'].append(pp[ii])
        acc['sample_y'].append(yy[ii])


def _utility_metric_finish(acc, cfg, seed, mode, deployable):
    c = cfg['predictive_need_probe']['utility_model']
    ss = np.concatenate(acc['sample_s']) if acc['sample_s'] else np.array([], np.float32)
    yy = np.concatenate(acc['sample_y']) if acc['sample_y'] else np.array([], np.float32)
    lab = yy > float(c['beneficial_threshold'])
    return {
        'mode': mode,
        'deployable': bool(deployable),
        'top1_agreement': float(acc['top1_ok'] / max(acc['top1_n'], 1)),
        'pairwise_rank_accuracy': float(acc['rank_ok'] / max(acc['rank_n'], 1)),
        'beneficial_top1_hit_rate': float(acc['benef'] / max(acc['benef_n'], 1)),
        'pearson': pearson(ss, yy),
        'spearman_sample': exact_spearman_sample(ss, yy, int(c['correlation_sample_points']), seed + 31),
        'beneficial_auroc': binary_auroc(ss, lab, int(c['auroc_sample_points']), seed + 32),
        'sampled_cells': int(len(ss)),
    }


def state_analytic_utility(cfg, stage, seed, art, va, val_ids, thresholds, rd, topcache, dev):
    """Test whether coarse predicted state is sufficient for Top-4 support matching."""
    _, std = cache_stats(cache_dir(cfg, stage))
    res = np.load(art / 'train_residual.npy', mmap_mode='r')
    bv = np.load(art / 'val_base_pred.npy', mmap_mode='r')
    top_ids = np.load(topcache['ids'], mmap_mode='r')
    top_conf = np.load(topcache['conf'], mmap_mode='r')
    top_mask = np.load(topcache['mask'], mmap_mode='r')
    p_need = np.load(rd / 'prob_need_q75_final_val.npy', mmap_mode='r')[..., 1]
    p_dir = np.load(rd / 'prob_direction_q75_final_val.npy', mmap_mode='r')[..., 1]
    p_mag = np.load(rd / 'prob_magnitude_3_final_val.npy', mmap_mode='r')
    centers = np.asarray(thresholds['magnitude_band_centers_raw'], np.float32)  # [P,3]
    q75 = _thr(thresholds, cfg['predictive_need_state_probe']['thresholds']['need_q75'])
    q50 = _thr(thresholds, cfg['predictive_need_state_probe']['thresholds']['magnitude_q_low'])
    q80 = _thr(thresholds, cfg['predictive_need_state_probe']['thresholds']['magnitude_q_high'])
    bs = int(cfg['predictive_need_state_probe']['utility']['eval_batch_size'])
    rngs = {
        'M3_CONF': np.random.default_rng(seed + 7101),
        'PREDICTED_STATE_ANALYTIC': np.random.default_rng(seed + 7102),
        'TRUE_STATE_ANALYTIC': np.random.default_rng(seed + 7103),
    }
    accs = {k: _utility_metric_accumulator_init() for k in rngs}
    for bi in batches(np.arange(len(val_ids)), bs):
        src = val_ids[bi]
        gids = np.asarray(top_ids[bi], np.int64)
        cm = t(top_mask[bi], dev, torch.bool)
        rr = t(_gather_support_residual(res, gids), dev) / float(std)
        base = t(bv[src], dev)
        true_need = (t(va.y[src], dev) - base) / float(std)
        valid = t(va.valid[src], dev, torch.bool)
        target = utility_from_need(true_need, rr, cm, cfg)

        pn = np.asarray(p_need[bi], np.float32)
        pd = np.asarray(p_dir[bi], np.float32)
        pm = np.asarray(p_mag[bi], np.float32)
        expected_mag = np.sum(pm * centers[None, :, None, :], axis=-1)
        predicted_state_need = pn * (2.0 * pd - 1.0) * expected_mag / float(std)

        err = _error(va, bv, src)
        ae = np.abs(err)
        hard = ae >= q75
        mag_class = np.zeros(err.shape, np.int64)
        mag_class[(ae >= q50) & (ae < q80)] = 1
        mag_class[ae >= q80] = 2
        center_grid = centers[None, :, None, :]
        true_center = np.take_along_axis(center_grid.repeat(len(src), axis=0).repeat(err.shape[2], axis=2), mag_class[..., None], axis=-1)[..., 0]
        true_state_need = hard.astype(np.float32) * np.where(err >= 0, 1.0, -1.0).astype(np.float32) * true_center / float(std)

        pred_score = utility_from_need(t(predicted_state_need, dev), rr, cm, cfg)
        true_state_score = utility_from_need(t(true_state_need, dev), rr, cm, cfg)
        conf_score = t(np.asarray(top_conf[bi], np.float32), dev)[:, :, :, None].expand_as(target)
        for mode, score in [('M3_CONF', conf_score), ('PREDICTED_STATE_ANALYTIC', pred_score), ('TRUE_STATE_ANALYTIC', true_state_score)]:
            _utility_metric_update(accs[mode], score, target, cm, valid, cfg, rngs[mode])
    out = {k: _utility_metric_finish(v, cfg, seed + i * 100, k, k != 'TRUE_STATE_ANALYTIC') for i, (k, v) in enumerate(accs.items())}
    out['PREDICTED_STATE_ANALYTIC']['description'] = 'Analytic support utility from predicted q75 need probability, predicted direction, and expected magnitude band.'
    out['TRUE_STATE_ANALYTIC']['description'] = 'Privileged coarse-state ceiling using true q75 need/direction and Train-derived magnitude-band centers; no exact residual magnitude.'
    out['M3_CONF']['description'] = 'M3 Top-4 candidate confidence broadcast across horizons; deployable reference baseline.'
    write_json(rd / 'state_analytic_utility_summary.json', out)
    print('STATE UTILITY FINAL', json.dumps(out, sort_keys=True), flush=True)
    return out


def load_v071_reference(cfg):
    p = Path(cfg['project']['exp_root']) / 'PredictiveNeedProbe' / cfg['experiment']['subgraph'] / 'development' / 'LATEST.json'
    if not p.exists():
        return None
    try:
        latest = json.loads(p.read_text())
        s = json.loads(Path(latest['summary']).read_text())
        return {
            'run_dir': latest.get('run_dir'),
            'recommendation': s.get('v08_recommendation'),
            'exact_need_overall': s.get('predictive_need_validation', {}).get('overall'),
            'exact_need_masked': s.get('predictive_need_validation', {}).get('masked'),
            'utility_A': s.get('support_utility_validation', {}).get('A'),
            'utility_B': s.get('support_utility_validation', {}).get('B'),
            'utility_C': s.get('support_utility_validation', {}).get('C'),
            'utility_B_ANALYTIC': s.get('analytic_predicted_need_utility_validation'),
        }
    except Exception as e:
        return {'error': repr(e), 'latest_path': str(p)}


def state_decision(task_summaries, state_utility, cfg):
    th = cfg['predictive_need_state_probe']['decision']
    nq75 = task_summaries['need_q75']['validation']['masked']
    nq90 = task_summaries['need_q90']['validation']['masked']
    dq75 = task_summaries['direction_q75']['validation']['masked']
    dq90 = task_summaries['direction_q90']['validation']['masked']
    u = state_utility['PREDICTED_STATE_ANALYTIC']
    ub = state_utility['M3_CONF']
    uc = state_utility['TRUE_STATE_ANALYTIC']
    need_pass = ((nq75.get('auroc') or 0) >= float(th['need_q75_masked_auroc_min']) or
                 (nq90.get('auroc') or 0) >= float(th['need_q90_masked_auroc_min']))
    direction_pass = ((dq75.get('balanced_accuracy') or 0) >= float(th['direction_q75_masked_balanced_accuracy_min']) or
                      (dq90.get('balanced_accuracy') or 0) >= float(th['direction_q90_masked_balanced_accuracy_min']))
    rank_gain = float(u['pairwise_rank_accuracy']) - float(ub['pairwise_rank_accuracy'])
    utility_pass = (float(u['pairwise_rank_accuracy']) >= float(th['predicted_state_rank_min']) and
                    rank_gain >= float(th['predicted_state_rank_gain_over_m3_min']))
    ceiling_pass = float(uc['pairwise_rank_accuracy']) >= float(th['true_state_rank_min'])
    passed = int(need_pass) + int(direction_pass) + int(utility_pass)
    if ceiling_pass and passed == 3:
        status = 'GO_V08_STATE'
    elif ceiling_pass and (passed >= 2 or float(u['pairwise_rank_accuracy']) >= float(th['predicted_state_rank_conditional'])):
        status = 'CONDITIONAL_V08_STATE'
    else:
        status = 'NO_GO_V08_STATE'
    diag = {
        'need_state_signal_pass': bool(need_pass),
        'direction_signal_pass': bool(direction_pass),
        'predicted_state_utility_pass': bool(utility_pass),
        'coarse_true_state_ceiling_pass': bool(ceiling_pass),
        'predicted_state_rank_gain_over_m3_conf': rank_gain,
        'masked_need_q75_auroc': nq75.get('auroc'),
        'masked_need_q90_auroc': nq90.get('auroc'),
        'masked_direction_q75_balanced_accuracy': dq75.get('balanced_accuracy'),
        'masked_direction_q90_balanced_accuracy': dq90.get('balanced_accuracy'),
        'predicted_state_pairwise_rank_accuracy': u['pairwise_rank_accuracy'],
        'true_state_pairwise_rank_accuracy': uc['pairwise_rank_accuracy'],
        'm3_conf_pairwise_rank_accuracy': ub['pairwise_rank_accuracy'],
        'thresholds': th,
    }
    return status, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='development', choices=['development'])
    ap.add_argument('--seed', type=int, default=10)
    a = ap.parse_args()
    cfg = load_yaml(ROOT / 'configs/presure.yaml')
    seed_all(a.seed)
    art = load_artifacts(cfg, a.stage)
    cache = cache_dir(cfg, a.stage)
    tr = CacheSplit(cache, 'train')
    va = CacheSplit(cache, 'val')
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rd = run_dir(cfg, 'PredictiveNeedStateProbe', a.stage, stamp(), a.seed)
    rd.mkdir(parents=True, exist_ok=True)
    ids = _split_ids(len(tr), cfg, a.seed)
    val_ids = _final_val_ids(len(va), cfg, a.seed)
    write_json(rd / 'split_manifest.json', {**{k: v.tolist() for k, v in ids.items()}, 'final_val': val_ids.tolist()})
    lineage = {
        'variant': 'PredictiveNeedStateProbe', 'stage': a.stage, 'seed': a.seed,
        'implementation_revision': cfg.get('implementation_revision'),
        'purpose': 'reformulate predictive need as coarse state after v0.7.1 exact-residual no-go',
        'deployable_inputs_exclude_current_future_direct_support_masks': True,
        'thresholds_derived_from_train_only': True,
        'validation_future_used_for_diagnostic_only': True,
    }
    write_json(rd / 'RUN_MANIFEST.json', {**lineage, 'status': 'STARTED'})
    try:
        load_m1(cfg, a.stage, dev)
        load_m3(cfg, a.stage, dev)
        bt = np.load(art / 'train_base_pred.npy', mmap_mode='r')
        bv = np.load(art / 'val_base_pred.npy', mmap_mode='r')
        thresholds = derive_train_thresholds(tr, bt, ids['state_train'], cfg)
        write_json(rd / 'train_only_state_thresholds.json', thresholds)
        tasks = ['need_q75', 'need_q90', 'direction_q75', 'direction_q90', 'magnitude_3', 'signed_state_q75']
        task_summaries = {}
        for task in tasks:
            task_summaries[task] = train_one_task(task, cfg, a.stage, a.seed, art, tr, va, ids, val_ids, thresholds, rd, dev)
        val_err = _error(va, bv, val_ids)
        hard_groups = hard_error_group_analysis(cfg, thresholds, val_err, va, val_ids, rd)
        topcache = prepare_final_top4(cfg, a.stage, art, val_ids, rd, dev)
        state_utility = state_analytic_utility(cfg, a.stage, a.seed, art, va, val_ids, thresholds, rd, topcache, dev)
        status, diagnostics = state_decision(task_summaries, state_utility, cfg)
        v071 = load_v071_reference(cfg)
        summary = {
            'variant': 'PredictiveNeedStateProbe', 'stage': a.stage, 'seed': a.seed,
            'exact_residual_route': 'NO_GO_FROM_V071',
            'state_tasks': task_summaries,
            'hard_error_groups': hard_groups,
            'state_analytic_utility': state_utility,
            'train_only_thresholds': thresholds,
            'v071_reference': v071,
            'v08_state_recommendation': status,
            'decision_diagnostics': diagnostics,
            'interpretation': {
                'need_q75': 'detect Train-defined top-25% absolute PatchSTG errors per horizon',
                'need_q90': 'detect Train-defined top-10% absolute PatchSTG errors per horizon',
                'direction_q75': 'predict error sign only on true q75 hard-error cells',
                'direction_q90': 'predict error sign only on true q90 hard-error cells',
                'magnitude_3': 'small/medium/large error band using Train q50/q80 thresholds',
                'signed_state_q75': 'no hard need / positive hard need / negative hard need',
                'PREDICTED_STATE_ANALYTIC': 'coarse predicted state -> analytic Top-4 residual utility',
                'TRUE_STATE_ANALYTIC': 'privileged coarse-state ceiling; still does not use exact residual magnitude',
            },
            'test_evaluated': False,
        }
        write_json(rd / 'summary.json', summary)
        (rd / 'DECISION.txt').write_text(status + '\n' + json.dumps(diagnostics, indent=2, sort_keys=True) + '\n')
        rows = []
        for task, s in task_summaries.items():
            for view, met in s['validation'].items():
                rows.append({'task': task, 'view': view, **{k: v for k, v in met.items() if not isinstance(v, (list, dict))}})
        for mode, met in state_utility.items():
            rows.append({'task': 'state_utility', 'view': mode, **met})
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with open(rd / 'state_probe_summary.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        latest_dir = Path(cfg['project']['exp_root']) / 'PredictiveNeedStateProbe' / cfg['experiment']['subgraph'] / a.stage
        latest_dir.mkdir(parents=True, exist_ok=True)
        write_json(latest_dir / 'LATEST.json', {
            'run_dir': str(rd), 'summary': str(rd / 'summary.json'),
            'recommendation': status, 'implementation_revision': cfg.get('implementation_revision')
        })
        write_json(rd / 'RUN_MANIFEST.json', {**lineage, 'status': 'FINISHED', 'recommendation': status})
        print('PREDICTIVE NEED STATE PROBE FINAL', status, flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f'PredictiveNeedStateProbe PASS run_dir={rd}', flush=True)
    except Exception as e:
        write_json(rd / 'RUN_MANIFEST.json', {**lineage, 'status': 'FAILED', 'error': repr(e)})
        raise


if __name__ == '__main__':
    main()
