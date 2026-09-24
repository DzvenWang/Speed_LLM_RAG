#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._common import setup, run_dir, default_tag
from srd.contract import assert_cache_ready
from srd.backbone import load_patchstg_adapter
from srd.data import CacheView
from srd.evaluate import predict_raw_batched
from srd.metrics import masked_mae, improvement_pct
from srd.patch_layout import extract_patch_layout
from srd.patch_transport import candidate_proposal, inject_patch, suffix_mask
from srd.forecast_utility import PatchAction, select_best_action_by_patch, best_single_action, build_same_episode_pool
from srd.utils import write_json


def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _gpu_status(device):
    if getattr(device, "type", "cpu") != "cuda" or not torch.cuda.is_available():
        return "device=cpu"
    try:
        alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        return f"gpu_alloc={alloc:.2f}GB gpu_reserved={reserved:.2f}GB gpu_peak={peak:.2f}GB"
    except Exception:
        return "device=cuda"


def _log(msg):
    print(f"[{_ts()}] {msg}", flush=True)


def _forecast(adapter, ck, q, variants, device, bs):
    if not variants:
        return np.empty((0,), np.float64)
    X = np.stack(variants).astype(np.float32)
    tod = np.repeat(q["tod_hist"][None, :], len(X), axis=0)
    dow = np.repeat(q["dow_hist"][None, :], len(X), axis=0)
    pred = predict_raw_batched(adapter, ck, X, tod, dow, device, batch_size=bs)
    y = np.repeat(q["y_raw"][None, :, :], len(X), axis=0)
    fv = np.repeat(q["future_valid"][None, :, :], len(X), axis=0)
    dm = np.repeat(q["direct_mask"][None, :], len(X), axis=0)
    return masked_mae(pred, y, fv, dm)


def _comp_mae(x, truth, mask):
    m = np.asarray(mask, bool) & np.isfinite(x) & np.isfinite(truth)
    if not np.any(m):
        return float("nan")
    return float(np.abs(np.asarray(x, np.float32)[m] - np.asarray(truth, np.float32)[m]).mean())


def _action_trial(q, candidate_cache, nodes, L, method, cid, cfg):
    p, f = candidate_proposal(q, candidate_cache[int(cid)], nodes, L, method, cfg)
    return inject_patch(q["x_obs"], p, f), p, f


def _compose_actions(q, candidate_cache, layout, actions_by_patch, L, method, cfg, ordered_patch_ids=None):
    x = q["x_obs"].copy()
    pids = list(actions_by_patch.keys()) if ordered_patch_ids is None else list(ordered_patch_ids)
    used = 0
    for pid in pids:
        a = actions_by_patch[int(pid)]
        p, f = candidate_proposal(q, candidate_cache[int(a.candidate_id)], layout.groups[int(pid)], L, method, cfg)
        if np.any(f):
            x = inject_patch(x, p, f)
            used += 1
    return x, used


def _compose_same_episode(q, candidate_cache, layout, eligible_row, cid, L, method, cfg):
    x = q["x_obs"].copy(); used = 0
    cand = candidate_cache[int(cid)]
    for pid, nodes in enumerate(layout.groups):
        if not bool(eligible_row[pid]):
            continue
        p, f = candidate_proposal(q, cand, nodes, L, method, cfg)
        if np.any(f):
            x = inject_patch(x, p, f)
            used += 1
    return x, used


def main():
    ap = argparse.ArgumentParser(description="Patch-Level Forecast-Utility Diagnostic v0.6")
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--patch-size", type=int, default=None)
    ap.add_argument("--patch-count", type=int, default=None)
    ap.add_argument("--transport", default=None)
    ap.add_argument("--scopes", default=None)
    ap.add_argument("--k-values", default=None)
    ap.add_argument("--same-episode-pool-cap", type=int, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    a = ap.parse_args()

    cfg = setup(a.stage); assert_cache_ready(cfg, a.stage)
    fcfg = cfg["forecast_utility_diagnostic"]
    transport = a.transport or str(fcfg["primary_transport"])
    scopes = [int(x) for x in (a.scopes.split(",") if a.scopes else fcfg["scopes"])]
    k_values = [int(x) for x in (a.k_values.split(",") if a.k_values else fcfg["k_values"])]
    pool_cap = int(a.same_episode_pool_cap or fcfg.get("same_episode_pool_cap", 512))
    allow_abstain = bool(fcfg.get("allow_abstain", True))

    tag = a.run_tag or default_tag("patchdiag_v060")
    out = run_dir(cfg, a.stage, tag) / "forecast_utility_oracle"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter, cp, ck, source = load_patchstg_adapter(cfg, a.stage, device, a.checkpoint)
    layout = extract_patch_layout(adapter.model, ck, backbone_root=cfg["project"]["backbone_root"], patch_size_override=a.patch_size, patch_count_override=a.patch_count)
    train, val = CacheView(cfg, a.stage, "train"), CacheView(cfg, a.stage, "val")
    ret = np.load(a.retrieval)
    qids = ret["query_ids"].astype(np.int64)
    cand_ids = ret["candidate_ids"].astype(np.int64)
    scores = ret["score"].astype(np.float32)
    eligible = ret["eligible"].astype(bool)
    Kmax = int(cand_ids.shape[-1])
    k_values = sorted(set(max(1, min(Kmax, int(k))) for k in k_values))

    _log("=" * 78)
    _log("Patch-Level Forecast-Utility Diagnostic v0.6.1 START")
    _log(f"run_tag={tag} stage={a.stage} device={device} queries={len(qids)} patches={layout.patch_count}")
    _log(f"transport={transport} scopes={scopes} k_values={k_values} Kmax={Kmax} pool_cap={pool_cap}")
    _log(f"checkpoint={cp} checkpoint_source={source}")
    _log(f"retrieval={a.retrieval}")
    _log(f"output={out}")
    _log(_gpu_status(device))
    _log("=" * 78)
    global_started = time.time()

    action_rows = []
    query_rows = []
    same_episode_rows = []

    for qpos, qi in enumerate(qids):
        query_started = time.time()
        q = val.row(int(qi))
        base_mae = float(_forecast(adapter, ck, q, [q["x_obs"].copy()], device, a.eval_batch_size)[0])
        _log(f"QUERY {qpos + 1}/{len(qids)} id={int(qi)} START baseline_mae={base_mae:.6f} {_gpu_status(device)}")
        raw_ids = cand_ids[qpos].reshape(-1)
        unique_ids = sorted({int(x) for x in raw_ids.tolist() if int(x) >= 0})
        candidate_cache = {cid: train.row(cid) for cid in unique_ids}

        for L in scopes:
            scope = suffix_mask(q["restorable"].shape, L)
            target_all = q["restorable"] & scope
            base_comp = _comp_mae(q["x_obs"], q["x_full"], target_all)
            scope_eligible = np.zeros(layout.patch_count, dtype=bool)
            for _pid, _nodes in enumerate(layout.groups):
                scope_eligible[_pid] = bool(eligible[qpos, _pid] and np.any(target_all[:, _nodes]))
            episode_pool = build_same_episode_pool(
                cand_ids[qpos:qpos+1], scores[qpos:qpos+1], scope_eligible[None, :], 0,
                max_candidates=pool_cap,
            )
            eligible_count = int(scope_eligible.sum())
            scope_started = time.time()
            _log(
                f"QUERY {qpos + 1}/{len(qids)} id={int(qi)} SCOPE=recent{L} START "
                f"eligible_patches={eligible_count}/{layout.patch_count} episode_pool={len(episode_pool)} "
                f"base_completion_mae={base_comp:.6f}"
            )

            # --------------------------------------------------------------
            # A. Single-patch candidate forecast utilities (future-target oracle).
            # --------------------------------------------------------------
            variants = []
            action_meta = []
            for pid, nodes in enumerate(layout.groups):
                if not eligible[qpos, pid]:
                    continue
                target = np.zeros_like(q["restorable"], bool)
                target[:, nodes] = True
                target &= target_all
                if not np.any(target):
                    continue
                for r in range(Kmax):
                    cid = int(cand_ids[qpos, pid, r])
                    if cid < 0:
                        continue
                    trial, _, fill = _action_trial(q, candidate_cache, nodes, L, transport, cid, cfg)
                    if not np.any(fill):
                        continue
                    variants.append(trial)
                    action_meta.append((int(pid), int(r + 1), cid, _comp_mae(trial, q["x_full"], target)))

            _log(
                f"QUERY {qpos + 1}/{len(qids)} SCOPE=recent{L} "
                f"single_patch_variants={len(variants)} EVALUATE batch_size={a.eval_batch_size}"
            )
            maes = _forecast(adapter, ck, q, variants, device, a.eval_batch_size)
            actions = []
            for meta, mae in zip(action_meta, maes):
                pid, rank, cid, comp_mae = meta
                util = float(base_mae - float(mae))
                arow = PatchAction(pid, rank, cid, float(mae), util, float(comp_mae))
                actions.append(arow)
                action_rows.append({
                    "query_id": int(qi), "query_pos": int(qpos), "recent_steps": int(L),
                    "patch_id": pid, "rank": rank, "candidate_id": cid, "transport": transport,
                    "baseline_forecast_mae": base_mae, "forecast_mae": float(mae),
                    "forecast_utility": util, "forecast_gain_pct": improvement_pct(base_mae, float(mae)),
                    "patch_completion_mae": float(comp_mae),
                })

            # --------------------------------------------------------------
            # B. Same-episode coherent variants. One train episode fills every
            #    damaged patch, preserving cross-patch temporal coherence.
            # --------------------------------------------------------------
            coherent_variants = []
            coherent_meta = []
            for erow in episode_pool:
                cid = int(erow["candidate_id"])
                trial, used = _compose_same_episode(q, candidate_cache, layout, scope_eligible, cid, L, transport, cfg)
                if used <= 0:
                    continue
                coherent_variants.append(trial)
                coherent_meta.append((cid, used, erow))
            _log(
                f"QUERY {qpos + 1}/{len(qids)} SCOPE=recent{L} "
                f"coherent_variants={len(coherent_variants)} EVALUATE batch_size={a.eval_batch_size}"
            )
            coherent_maes = _forecast(adapter, ck, q, coherent_variants, device, a.eval_batch_size)
            coherent_eval = []
            for meta, mae in zip(coherent_meta, coherent_maes):
                cid, used, erow = meta
                rec = {
                    "query_id": int(qi), "query_pos": int(qpos), "recent_steps": int(L),
                    "candidate_id": cid, "transport": transport, "used_patches": int(used),
                    "min_rank": int(erow["min_rank"]), "mean_rank": float(erow["mean_rank"]),
                    "patch_hits": int(erow["patch_hits"]), "coverage": float(erow["coverage"]),
                    "episode_rank_score": float(erow["episode_rank_score"]),
                    "mean_local_score": float(erow["mean_local_score"]),
                    "baseline_forecast_mae": base_mae, "forecast_mae": float(mae),
                    "forecast_utility": float(base_mae - float(mae)),
                    "forecast_gain_pct": improvement_pct(base_mae, float(mae)),
                    "completion_mae": _comp_mae(coherent_variants[len(coherent_eval)], q["x_full"], target_all),
                }
                coherent_eval.append(rec); same_episode_rows.append(rec)

            # --------------------------------------------------------------
            # C. Derive Oracle@K diagnostics from the already evaluated actions.
            # --------------------------------------------------------------
            for K in k_values:
                k_started = time.time()
                by_patch = select_best_action_by_patch(actions, K, base_mae, allow_abstain=allow_abstain)
                single = best_single_action(actions, K, base_mae, allow_abstain=allow_abstain)
                _log(
                    f"QUERY {qpos + 1}/{len(qids)} SCOPE=recent{L} K={K} START "
                    f"beneficial_patch_actions={len(by_patch)}"
                )

                if single is None:
                    single_mae = base_mae; single_pid = -1; single_cid = -1; single_rank = 0
                else:
                    single_mae = float(single.forecast_mae); single_pid = int(single.patch_id)
                    single_cid = int(single.candidate_id); single_rank = int(single.rank)
                query_rows.append({
                    "query_id": int(qi), "query_pos": int(qpos), "mode": "single_patch_forecast_oracle",
                    "recent_steps": int(L), "K": int(K), "transport": transport,
                    "baseline_forecast_mae": base_mae, "forecast_mae": single_mae,
                    "forecast_utility": base_mae - single_mae, "forecast_gain_pct": improvement_pct(base_mae, single_mae),
                    "selected_patches": 0 if single is None else 1, "selected_patch_id": single_pid,
                    "selected_candidate_id": single_cid, "selected_candidate_rank": single_rank,
                    "baseline_completion_mae": base_comp, "completion_mae": float("nan"),
                })

                # Compose all individually beneficial patches. This exposes interaction harm.
                if by_patch:
                    x_all, used_all = _compose_actions(q, candidate_cache, layout, by_patch, L, transport, cfg)
                    all_mae = float(_forecast(adapter, ck, q, [x_all], device, a.eval_batch_size)[0])
                    all_comp = _comp_mae(x_all, q["x_full"], target_all)
                else:
                    x_all = q["x_obs"].copy(); used_all = 0; all_mae = base_mae; all_comp = base_comp
                query_rows.append({
                    "query_id": int(qi), "query_pos": int(qpos), "mode": "all_individually_beneficial_patches",
                    "recent_steps": int(L), "K": int(K), "transport": transport,
                    "baseline_forecast_mae": base_mae, "forecast_mae": all_mae,
                    "forecast_utility": base_mae - all_mae, "forecast_gain_pct": improvement_pct(base_mae, all_mae),
                    "selected_patches": int(used_all), "selected_patch_id": -1, "selected_candidate_id": -1,
                    "selected_candidate_rank": 0, "baseline_completion_mae": base_comp, "completion_mae": all_comp,
                })

                # Utility-ranked prefix subset Oracle: sort patches by their single-patch future utility,
                # evaluate prefixes, and choose the best prefix using the future target (diagnostic only).
                ordered = sorted(by_patch.keys(), key=lambda pid: float(by_patch[pid].forecast_utility), reverse=True)
                prefix_variants, prefix_counts = [], []
                for m in range(1, len(ordered) + 1):
                    xp, usedp = _compose_actions(q, candidate_cache, layout, by_patch, L, transport, cfg, ordered_patch_ids=ordered[:m])
                    prefix_variants.append(xp); prefix_counts.append(usedp)
                if prefix_variants:
                    pmaes = _forecast(adapter, ck, q, prefix_variants, device, a.eval_batch_size)
                    best_i = int(np.argmin(np.r_[base_mae, pmaes]))
                    if best_i == 0:
                        prefix_mae = base_mae; prefix_used = 0; prefix_comp = base_comp
                    else:
                        idx = best_i - 1
                        prefix_mae = float(pmaes[idx]); prefix_used = int(prefix_counts[idx])
                        prefix_comp = _comp_mae(prefix_variants[idx], q["x_full"], target_all)
                else:
                    prefix_mae = base_mae; prefix_used = 0; prefix_comp = base_comp
                query_rows.append({
                    "query_id": int(qi), "query_pos": int(qpos), "mode": "patch_subset_prefix_forecast_oracle",
                    "recent_steps": int(L), "K": int(K), "transport": transport,
                    "baseline_forecast_mae": base_mae, "forecast_mae": prefix_mae,
                    "forecast_utility": base_mae - prefix_mae, "forecast_gain_pct": improvement_pct(base_mae, prefix_mae),
                    "selected_patches": int(prefix_used), "selected_patch_id": -1, "selected_candidate_id": -1,
                    "selected_candidate_rank": 0, "baseline_completion_mae": base_comp, "completion_mae": prefix_comp,
                })

                # Same-episode coherent retrieval Oracle. Candidate episode pool membership for Oracle@K
                # is defined only by inference-available local retrieval rank (min_rank <= K).
                coherent_k = [r for r in coherent_eval if int(r["min_rank"]) <= int(K)]
                if coherent_k:
                    best_c = min(coherent_k, key=lambda r: float(r["forecast_mae"]))
                    if allow_abstain and float(best_c["forecast_mae"]) >= base_mae:
                        best_c = None
                else:
                    best_c = None
                if best_c is None:
                    coh_mae = base_mae; coh_used = 0; coh_cid = -1; coh_comp = base_comp; coh_rank = 0
                else:
                    coh_mae = float(best_c["forecast_mae"]); coh_used = int(best_c["used_patches"])
                    coh_cid = int(best_c["candidate_id"]); coh_comp = float(best_c["completion_mae"])
                    coh_rank = int(best_c["min_rank"])
                query_rows.append({
                    "query_id": int(qi), "query_pos": int(qpos), "mode": "same_episode_coherent_forecast_oracle",
                    "recent_steps": int(L), "K": int(K), "transport": transport,
                    "baseline_forecast_mae": base_mae, "forecast_mae": coh_mae,
                    "forecast_utility": base_mae - coh_mae, "forecast_gain_pct": improvement_pct(base_mae, coh_mae),
                    "selected_patches": int(coh_used), "selected_patch_id": -1, "selected_candidate_id": coh_cid,
                    "selected_candidate_rank": coh_rank, "baseline_completion_mae": base_comp, "completion_mae": coh_comp,
                })
                single_gain = improvement_pct(base_mae, single_mae)
                all_gain = improvement_pct(base_mae, all_mae)
                subset_gain = improvement_pct(base_mae, prefix_mae)
                coherent_gain = improvement_pct(base_mae, coh_mae)
                _log(
                    f"QUERY {qpos + 1}/{len(qids)} SCOPE=recent{L} K={K} DONE "
                    f"single={single_gain:+.3f}% all_beneficial={all_gain:+.3f}% "
                    f"subset={subset_gain:+.3f}% coherent={coherent_gain:+.3f}% "
                    f"selected_subset_patches={prefix_used} elapsed={time.time() - k_started:.1f}s"
                )

            scope_rows = [
                r for r in query_rows
                if int(r["query_pos"]) == int(qpos) and int(r["recent_steps"]) == int(L)
            ]
            def _best_scope_gain(mode):
                vals = [float(r["forecast_gain_pct"]) for r in scope_rows if r["mode"] == mode]
                return max(vals) if vals else float("nan")
            _log(
                f"QUERY {qpos + 1}/{len(qids)} SCOPE=recent{L} DONE "
                f"actions={len(actions)} coherent_eval={len(coherent_eval)} "
                f"best_single={_best_scope_gain('single_patch_forecast_oracle'):+.3f}% "
                f"best_subset={_best_scope_gain('patch_subset_prefix_forecast_oracle'):+.3f}% "
                f"best_coherent={_best_scope_gain('same_episode_coherent_forecast_oracle'):+.3f}% "
                f"elapsed={time.time() - scope_started:.1f}s {_gpu_status(device)}"
            )

        _log(
            f"QUERY {qpos + 1}/{len(qids)} id={int(qi)} DONE "
            f"elapsed={time.time() - query_started:.1f}s total_elapsed={time.time() - global_started:.1f}s"
        )
        if (qpos + 1) % 5 == 0 or qpos + 1 == len(qids):
            _log(f"PROGRESS {qpos + 1}/{len(qids)} ({100.0 * (qpos + 1) / len(qids):.1f}%)")

    adf = pd.DataFrame(action_rows)
    qdf = pd.DataFrame(query_rows)
    cdf = pd.DataFrame(same_episode_rows)
    adf.to_csv(out / "single_patch_candidate_actions.csv.gz", index=False, compression="gzip")
    qdf.to_csv(out / "per_query.csv", index=False)
    cdf.to_csv(out / "same_episode_candidates.csv.gz", index=False, compression="gzip")

    summary_rows = []
    for keys, g in qdf.groupby(["mode", "recent_steps", "K", "transport"], sort=True):
        mode, L, K, method = keys
        b = float(g.baseline_forecast_mae.mean()); m = float(g.forecast_mae.mean())
        bc = float(g.baseline_completion_mae.mean())
        cm = float(g.completion_mae.mean()) if g.completion_mae.notna().any() else float("nan")
        summary_rows.append({
            "mode": mode, "recent_steps": int(L), "K": int(K), "transport": method,
            "baseline_forecast_mae": b, "forecast_mae": m, "forecast_gain_pct": improvement_pct(b, m),
            "beneficial_query_rate": float((g.forecast_utility > 0).mean()),
            "mean_selected_patches": float(g.selected_patches.mean()),
            "baseline_completion_mae": bc, "completion_mae": cm,
            "completion_improvement_pct": improvement_pct(bc, cm) if np.isfinite(cm) else float("nan"),
        })
    sm = pd.DataFrame(summary_rows)
    sm.to_csv(out / "forecast_oracle_by_k.csv", index=False)

    def best(mode):
        z = sm[sm["mode"] == mode]
        return float(z.forecast_gain_pct.max()) if len(z) else float("nan")

    payload = {
        "version": "0.6.1",
        "queries": int(len(qids)), "patch_count": int(layout.patch_count), "transport": transport,
        "k_values": k_values, "scopes": scopes, "same_episode_pool_cap": pool_cap,
        "candidate_space_target_pct": float(fcfg.get("candidate_space_target_pct", 8.0)),
        "strong_candidate_space_target_pct": float(fcfg.get("strong_candidate_space_target_pct", 10.0)),
        "checkpoint": str(cp), "checkpoint_source": source,
        "retrieval_is_inference_available": True,
        "forecast_oracle_uses_future_target_for_selection": True,
        "forecast_oracle_is_diagnostic_only": True,
        "allow_abstain": allow_abstain,
        "best_single_patch_forecast_oracle_pct": best("single_patch_forecast_oracle"),
        "best_all_beneficial_patches_pct": best("all_individually_beneficial_patches"),
        "best_patch_subset_prefix_forecast_oracle_pct": best("patch_subset_prefix_forecast_oracle"),
        "best_same_episode_coherent_forecast_oracle_pct": best("same_episode_coherent_forecast_oracle"),
        "results": summary_rows,
    }
    write_json(out / "summary.json", payload)
    _log("=" * 78)
    _log("FORECAST UTILITY FINAL TABLE")
    print(sm.to_string(index=False), flush=True)
    _log("FORECAST UTILITY SUMMARY JSON")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2), flush=True)
    _log(f"TOTAL_ELAPSED_SEC={time.time() - global_started:.1f} {_gpu_status(device)}")
    _log(f"FORECAST_UTILITY_SUMMARY={out / 'summary.json'}")
    _log("Patch-Level Forecast-Utility Diagnostic v0.6.1 COMPLETE")
    _log("=" * 78)


if __name__ == "__main__":
    main()
