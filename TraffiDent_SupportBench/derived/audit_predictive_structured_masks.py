#!/usr/bin/env python3
"""Scientific audit for predictive structured sensing in SupportBench v1.2.4.

The authoritative Chapter-3 metric is the network mean of per-target
predictive-context loss:

    C_q^loss = (1/N) * sum_n C_{q,n}^loss.

In v1.2.2 the API field ``aggregate_context_loss`` was a configured target
quantile (q=0.75 by default), not this network mean.  This audit therefore uses
``network_mean_context_loss`` from the canonical SupportBench API and treats the
legacy quantile as a separate descriptive diagnostic.

The audit is future-free.  It reads only fixed sensing masks and the Train-only
predictive-dependency graph.  It can audit ID or OOD layouts directly from the
canonical benchmark, so OOD does not require an OOD Support-Opportunity CSV.

Outputs
-------
  paired_rows.csv        matched general/predictive base-window rows
  pair_summary.csv       context/scope comparison statistics
  family_summary.csv     canonical family-level descriptive statistics
  api_recheck.csv        optional overlap check against an existing Stage-A CSV
  api_inventory.json     API/view-spec field inventory
  audit_summary.json     final scientific diagnostic summary
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from supportbench.dataset import SupportBench


PAIR_SPECS = {
    "regional": ("regional_outage", "predictive_regional_outage"),
    "directional": ("directional_outage", "predictive_directional_outage"),
}
AUDIT_METRICS = ["predictive_context_loss", "support_dependent_fraction"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_default(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 128 else {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, Path):
        return str(x)
    if is_dataclass(x):
        return asdict(x)
    return str(x)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=json_default) + "\n", encoding="utf-8")


def safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return float("nan")
    return float(num / den)


def bootstrap_ratio_ci(general: np.ndarray, predictive: np.ndarray, seed: int, n_boot: int = 4000) -> Tuple[float, float]:
    general = np.asarray(general, dtype=float)
    predictive = np.asarray(predictive, dtype=float)
    ok = np.isfinite(general) & np.isfinite(predictive)
    general, predictive = general[ok], predictive[ok]
    if len(general) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, len(general), size=len(general))
        den = float(np.mean(general[idx]))
        vals[b] = float(np.mean(predictive[idx]) / den) if abs(den) > 1e-12 else np.nan
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def verdict_for_ratio(ratio: float, threshold: float) -> str:
    if not np.isfinite(ratio):
        return "UNDEFINED"
    if ratio >= threshold:
        return "PASS"
    if ratio >= 1.0:
        return "WEAK_BELOW_TARGET"
    return "REVERSED"


def deterministic_group_sample(groups: List[str], max_groups: int, seed: int) -> List[str]:
    groups = sorted(set(map(str, groups)))
    if max_groups <= 0 or len(groups) <= max_groups:
        return groups
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(groups), size=max_groups, replace=False))
    return [groups[i] for i in idx]


def flatten_numeric_scalars(obj: Any, prefix: str = "", depth: int = 0, max_depth: int = 5) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if depth > max_depth:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_numeric_scalars(v, key, depth + 1, max_depth))
    elif isinstance(obj, (int, float, np.integer, np.floating)) and not isinstance(obj, bool):
        val = float(obj)
        if np.isfinite(val):
            out[prefix] = val
    return out


def interesting_scalars(d: Dict[str, float]) -> Dict[str, float]:
    pat = re.compile(r"(objective|predict|context|utility|score|loss|density|severity|missing)", re.I)
    return {k: v for k, v in d.items() if pat.search(k)}


def sample_attr_inventory(sample: Any) -> List[str]:
    try:
        return sorted([x for x in dir(sample) if not x.startswith("_")])
    except Exception:
        return []


def canonical_feature_df(
    bench: SupportBench,
    *,
    split: str,
    distribution: str,
    severity: float,
    budget: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the audit table directly from canonical fixed sensing views."""
    families = sorted({x for pair in PAIR_SPECS.values() for x in pair})
    d = bench.views.copy()
    d = d[(d["split"].astype(str) == split) & d["family"].astype(str).isin(families)].copy()
    d = d[np.isclose(d["severity"].astype(float), float(severity), atol=1e-8)].copy()
    if distribution != "all":
        d = d[d["distribution"].astype(str) == distribution].copy()
    if "instance" in d.columns:
        d = d[d["instance"].astype(int) == 0].copy()

    legal = set(map(int, bench.window_starts(split, budget)))
    d = d[d["window_start"].astype(int).isin(legal)].copy()
    if not len(d):
        raise ValueError(
            f"No canonical views remain for split={split}, distribution={distribution}, "
            f"severity={severity}, budget={budget}"
        )

    rows: List[dict[str, Any]] = []
    inventory: Dict[str, Any] = {"metric_semantics": {}, "families": {}}
    seen_family: set[str] = set()
    for k, (_, row) in enumerate(d.sort_values(["family", "window_start", "view_id"]).iterrows(), start=1):
        vid = str(row["view_id"])
        sample = bench.sample(vid)
        cm = bench.predictive_context_metrics(sample)
        formal = float(cm["network_mean_context_loss"])
        per_target = np.asarray(cm["per_target_context_loss"], dtype=float)
        explicit_mean = float(np.mean(per_target)) if len(per_target) else 0.0
        legacy = float(cm["context_loss_quantile_value"])
        rec = {
            "view_id": vid,
            "view_group_id": str(sample.view_group_id),
            "split": str(sample.split),
            "window_start": int(sample.window_start),
            "family": str(sample.view_spec["family"]),
            "severity": float(sample.view_spec["severity"]),
            "distribution": str(sample.view_spec.get("distribution", "id")),
            "predictive_context_loss": formal,
            "network_mean_context_loss": formal,
            "legacy_context_loss_quantile": legacy,
            "support_dependent_fraction": float(cm["support_dependent_fraction"]),
            "mean_recompute_abs_diff": abs(formal - explicit_mean),
        }
        rows.append(rec)

        family = rec["family"]
        if family not in seen_family:
            seen_family.add(family)
            spec_scalars = flatten_numeric_scalars(sample.view_spec)
            inventory["families"][family] = {
                "metric_keys": sorted(map(str, cm.keys())),
                "view_spec_keys": sorted(map(str, sample.view_spec.keys())),
                "interesting_view_spec_scalars": interesting_scalars(spec_scalars),
                "sample_public_attributes": sample_attr_inventory(sample),
            }
        if k % 500 == 0:
            print(f"[canonical] recomputed {k}/{len(d)} views", flush=True)

    inventory["metric_semantics"] = {
        "formal": "predictive_context_loss = network_mean_context_loss = mean(per_target_context_loss)",
        "legacy": "aggregate_context_loss/context_loss_quantile_value = configured target quantile; diagnostic only",
        "chapter3_equation": "C_q^loss = (1/N) sum_n C_{q,n}^loss",
    }
    return pd.DataFrame(rows), inventory


def collapse_family_per_group(d: pd.DataFrame, family: str) -> pd.DataFrame:
    g = d[d["family"].astype(str) == family].copy()
    if not len(g):
        return g
    group_cols = ["view_group_id"]
    if "distribution" in g.columns:
        group_cols = ["distribution", "view_group_id"]
    agg = {
        "view_id": "first",
        "predictive_context_loss": "mean",
        "network_mean_context_loss": "mean",
        "legacy_context_loss_quantile": "mean",
        "support_dependent_fraction": "mean",
    }
    for c in ["window_start", "split", "severity"]:
        if c in g.columns:
            agg[c] = "first"
    return g.sort_values(group_cols + ["view_id"]).groupby(group_cols, as_index=False).agg(agg)


def build_pair_rows(d: pd.DataFrame, pair_name: str, max_groups: int, seed: int) -> pd.DataFrame:
    general_family, pred_family = PAIR_SPECS[pair_name]
    g = collapse_family_per_group(d, general_family)
    p = collapse_family_per_group(d, pred_family)
    if not len(g) or not len(p):
        raise ValueError(f"No canonical rows for comparison {pair_name}: {general_family} vs {pred_family}")
    merge_cols = ["view_group_id"]
    if "distribution" in g.columns and "distribution" in p.columns:
        merge_cols = ["distribution", "view_group_id"]
    pair = g.merge(p, on=merge_cols, suffixes=("_general", "_predictive"), how="inner")
    if not len(pair):
        raise ValueError(f"No matched base windows for {pair_name}")
    keys = [f"{r.get('distribution','')}::{r['view_group_id']}" for _, r in pair.iterrows()]
    chosen = set(deterministic_group_sample(keys, max_groups, seed))
    keep = np.asarray([k in chosen for k in keys], bool)
    pair = pair[keep].copy().sort_values(merge_cols).reset_index(drop=True)
    pair.insert(0, "pair", pair_name)
    # For permanent node outages, the Chapter-3 layout objective J(S) equals the
    # network-mean context loss induced by S.
    pair["j_effective_general"] = pair["predictive_context_loss_general"]
    pair["j_effective_predictive"] = pair["predictive_context_loss_predictive"]
    pair["context_diff"] = pair["predictive_context_loss_predictive"] - pair["predictive_context_loss_general"]
    pair["context_ratio"] = pair["predictive_context_loss_predictive"] / pair["predictive_context_loss_general"].replace(0, np.nan)
    pair["scope_diff"] = pair["support_dependent_fraction_predictive"] - pair["support_dependent_fraction_general"]
    pair["scope_ratio"] = pair["support_dependent_fraction_predictive"] / pair["support_dependent_fraction_general"].replace(0, np.nan)
    return pair


def summarize_pair(pair: pd.DataFrame, pair_name: str, threshold: float, seed: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for metric in AUDIT_METRICS:
        g = pair[f"{metric}_general"].to_numpy(float)
        p = pair[f"{metric}_predictive"].to_numpy(float)
        ok = np.isfinite(g) & np.isfinite(p)
        g, p = g[ok], p[ok]
        diff = p - g
        ratio = safe_ratio(float(np.mean(p)), float(np.mean(g)))
        lo, hi = bootstrap_ratio_ci(g, p, seed=seed + (0 if metric == "predictive_context_loss" else 1000))
        ties = np.isclose(p, g, rtol=1e-7, atol=1e-9)
        row = {
            "pair": pair_name,
            "metric": metric,
            "n_matched": int(len(g)),
            "general_mean": float(np.mean(g)) if len(g) else np.nan,
            "predictive_mean": float(np.mean(p)) if len(p) else np.nan,
            "ratio_of_means": ratio,
            "ratio_bootstrap_ci_low": lo,
            "ratio_bootstrap_ci_high": hi,
            "paired_diff_mean": float(np.mean(diff)) if len(diff) else np.nan,
            "paired_diff_median": float(np.median(diff)) if len(diff) else np.nan,
            "predictive_win_rate": float(np.mean(p > g)) if len(g) else np.nan,
            "tie_rate": float(np.mean(ties)) if len(g) else np.nan,
            "predictive_loss_rate": float(np.mean(p < g)) if len(g) else np.nan,
        }
        if metric == "predictive_context_loss":
            row["target_ratio"] = float(threshold)
            row["verdict"] = verdict_for_ratio(ratio, threshold)
        else:
            row["target_ratio"] = np.nan
            row["verdict"] = "DESCRIPTIVE_ONLY"
        out.append(row)
    return out


def family_summary(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, g in d.groupby(["distribution", "family"], sort=True):
        dist, family = fam
        c = g["predictive_context_loss"].to_numpy(float)
        l = g["legacy_context_loss_quantile"].to_numpy(float)
        s = g["support_dependent_fraction"].to_numpy(float)
        corr = float(np.corrcoef(c, s)[0, 1]) if len(g) > 1 and np.nanstd(c) > 0 and np.nanstd(s) > 0 else np.nan
        rows.append({
            "distribution": str(dist), "family": str(family), "n": int(len(g)),
            "context_mean": float(np.nanmean(c)), "context_std": float(np.nanstd(c)),
            "context_q10": float(np.nanquantile(c, .10)), "context_q50": float(np.nanquantile(c, .50)), "context_q90": float(np.nanquantile(c, .90)),
            "legacy_context_quantile_mean": float(np.nanmean(l)),
            "scope_mean": float(np.nanmean(s)), "scope_std": float(np.nanstd(s)),
            "scope_q10": float(np.nanquantile(s, .10)), "scope_q50": float(np.nanquantile(s, .50)), "scope_q90": float(np.nanquantile(s, .90)),
            "context_scope_corr": corr,
        })
    return pd.DataFrame(rows)


def feature_crosscheck(canonical: pd.DataFrame, features: Optional[Path], distribution: str, severity: float, out_dir: Path) -> Dict[str, Any]:
    if features is None or not features.exists() or features.stat().st_size == 0:
        return {"available": False, "reason": "feature CSV not supplied or missing"}
    f = pd.read_csv(features)
    if distribution != "all" and "distribution" in f.columns:
        f = f[f["distribution"].astype(str) == distribution].copy()
    if "severity" in f.columns:
        f = f[np.isclose(f["severity"].astype(float), severity, atol=1e-8)].copy()
    if not len(f):
        return {"available": True, "n_overlap": 0, "reason": "feature CSV contains no matching distribution/severity rows"}
    cols = [c for c in ["view_id", "predictive_context_loss", "network_mean_context_loss", "legacy_context_loss_quantile", "support_dependent_fraction"] if c in f.columns]
    if "view_id" not in cols:
        return {"available": True, "n_overlap": 0, "reason": "feature CSV has no view_id"}
    z = canonical.merge(f[cols], on="view_id", suffixes=("_canonical", "_csv"), how="inner")
    if not len(z):
        return {"available": True, "n_overlap": 0, "reason": "no overlapping view_id"}
    if "predictive_context_loss_csv" in z.columns:
        z["abs_formal_context_diff"] = np.abs(z["predictive_context_loss_canonical"] - z["predictive_context_loss_csv"])
    if "support_dependent_fraction_csv" in z.columns:
        z["abs_scope_diff"] = np.abs(z["support_dependent_fraction_canonical"] - z["support_dependent_fraction_csv"])
    if "legacy_context_loss_quantile_csv" in z.columns:
        z["abs_legacy_quantile_diff"] = np.abs(z["legacy_context_loss_quantile_canonical"] - z["legacy_context_loss_quantile_csv"])
    path = out_dir / "api_recheck.csv"
    z.to_csv(path, index=False)
    info = {"available": True, "n_overlap": int(len(z)), "path": str(path)}
    for c in ["abs_formal_context_diff", "abs_scope_diff", "abs_legacy_quantile_diff"]:
        if c in z.columns:
            info[f"max_{c}"] = float(z[c].max())
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None, help="Optional Stage-A feature CSV for overlap consistency check")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--subgraph", default="TraffiDent_Net_06")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--budget", default="development", choices=["development", "full"])
    ap.add_argument("--distribution", default="id", choices=["id", "ood", "all"])
    ap.add_argument("--severity", type=float, default=0.40)
    ap.add_argument("--max-groups", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--context-ratio-target", type=float, default=1.05)
    ap.add_argument("--recompute-api", action="store_true", help="Retained for CLI compatibility; canonical API is always authoritative in v1.2.3")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    bench = SupportBench(args.subgraph, config=args.config)
    d, inventory = canonical_feature_df(
        bench, split=args.split, distribution=args.distribution,
        severity=args.severity, budget=args.budget,
    )
    write_json(out_dir / "api_inventory.json", inventory)

    all_pairs: List[pd.DataFrame] = []
    summaries: List[Dict[str, Any]] = []
    for i, pair_name in enumerate(PAIR_SPECS):
        pair = build_pair_rows(d, pair_name, args.max_groups, args.seed + i * 100)
        all_pairs.append(pair)
        summaries.extend(summarize_pair(pair, pair_name, args.context_ratio_target, args.seed + i * 1000))

    paired = pd.concat(all_pairs, ignore_index=True)
    paired_path = out_dir / "paired_rows.csv"
    paired.to_csv(paired_path, index=False)
    summary_df = pd.DataFrame(summaries)
    summary_path = out_dir / "pair_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    fam_df = family_summary(d)
    family_path = out_dir / "family_summary.csv"
    fam_df.to_csv(family_path, index=False)

    context_rows = summary_df[summary_df["metric"] == "predictive_context_loss"].copy()
    verdicts = list(map(str, context_rows["verdict"]))
    if all(v == "PASS" for v in verdicts):
        overall_status = "PASS"
    elif any(v == "REVERSED" for v in verdicts):
        overall_status = "REVERSED"
    else:
        overall_status = "WEAK_BELOW_TARGET"

    scope_direction = []
    for pair_name in PAIR_SPECS:
        c = summary_df[(summary_df["pair"] == pair_name) & (summary_df["metric"] == "predictive_context_loss")].iloc[0]
        s = summary_df[(summary_df["pair"] == pair_name) & (summary_df["metric"] == "support_dependent_fraction")].iloc[0]
        scope_direction.append({
            "pair": pair_name,
            "context_ratio": float(c["ratio_of_means"]),
            "scope_ratio": float(s["ratio_of_means"]),
            "different_direction": bool((c["ratio_of_means"] < 1.0 < s["ratio_of_means"]) or (s["ratio_of_means"] < 1.0 < c["ratio_of_means"])),
            "note": "scope is thresholded and descriptive; it is not required to move monotonically with network-mean context loss",
        })

    feature_path = Path(args.features) if args.features else None
    cross = feature_crosscheck(d, feature_path, args.distribution, args.severity, out_dir)
    report = {
        "schema": "predictive_structured_mask_audit_v2",
        "supportbench_version": "1.2.4",
        "future_free": True,
        "canonical_metric": "predictive_context_loss = network_mean_context_loss = mean(per_target_context_loss)",
        "legacy_metric": "aggregate_context_loss is the v1.2.2 configured target quantile and is diagnostic only",
        "subgraph": args.subgraph,
        "split": args.split,
        "budget": args.budget,
        "distribution": args.distribution,
        "severity": args.severity,
        "max_groups_per_comparison": args.max_groups,
        "context_ratio_target": args.context_ratio_target,
        "overall_status": overall_status,
        "overall_context_efficacy_pass": bool(overall_status == "PASS"),
        "j_effective_definition": "For full-history node outages, J(S) equals the Chapter-3 network-mean predictive-context loss induced by S.",
        "comparisons": context_rows.to_dict(orient="records"),
        "scope_direction_diagnostic": scope_direction,
        "feature_csv_crosscheck": cross,
        "outputs": {
            "paired_rows": str(paired_path),
            "pair_summary": str(summary_path),
            "family_summary": str(family_path),
            "api_inventory": str(out_dir / "api_inventory.json"),
        },
    }
    write_json(out_dir / "audit_summary.json", report)

    print("\n=== Predictive Structured Mask Audit v1.2.4 ===")
    print(f"subgraph={args.subgraph} split={args.split} budget={args.budget}")
    print(f"distribution={args.distribution} severity={args.severity:.2f} max_groups={args.max_groups}")
    print("metric=network_mean_context_loss = mean(per_target_context_loss)")
    print("\n[Context efficacy: predictive/general]")
    for _, r in context_rows.iterrows():
        print(
            f"  {r['pair']:<12s} n={int(r['n_matched']):4d} "
            f"general={r['general_mean']:.6f} predictive={r['predictive_mean']:.6f} "
            f"ratio={r['ratio_of_means']:.4f} "
            f"95%CI=[{r['ratio_bootstrap_ci_low']:.4f},{r['ratio_bootstrap_ci_high']:.4f}] "
            f"win={r['predictive_win_rate']:.3f} verdict={r['verdict']}"
        )
    print("\n[Support-dependent scope: descriptive only]")
    for _, r in summary_df[summary_df["metric"] == "support_dependent_fraction"].iterrows():
        print(
            f"  {r['pair']:<12s} general={r['general_mean']:.6f} "
            f"predictive={r['predictive_mean']:.6f} ratio={r['ratio_of_means']:.4f} "
            f"win={r['predictive_win_rate']:.3f}"
        )
    print("\n[Stage-A feature overlap check]")
    print(json.dumps(cross, indent=2))
    print(f"\nRESULT={overall_status}")
    print(f"AUDIT_SUMMARY={out_dir / 'audit_summary.json'}")
    print(f"PAIR_SUMMARY={summary_path}")
    print(f"PAIRED_ROWS={paired_path}")

    if args.strict and overall_status != "PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
