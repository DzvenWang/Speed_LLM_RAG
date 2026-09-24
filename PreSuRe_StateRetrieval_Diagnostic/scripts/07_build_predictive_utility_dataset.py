#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._common import setup
from srd.contract import assert_cache_ready
from srd.data import CacheView
from srd.patch_layout import extract_patch_layout
from srd.backbone import load_patchstg_adapter
from srd.utility_selector import add_inference_features, feature_columns
from srd.utils import write_json


def log(msg): print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _stats(v):
    v = np.asarray(v, np.float64); v = v[np.isfinite(v)]
    return (float(v.mean()), float(v.std())) if v.size else (np.nan, np.nan)


def _delta_abs(x, valid):
    x = np.asarray(x, np.float64); valid = np.asarray(valid, bool)
    if x.shape[0] < 2: return np.nan
    m = valid[1:] & valid[:-1]
    return float(np.abs(x[1:] - x[:-1])[m].mean()) if np.any(m) else np.nan


def main():
    ap = argparse.ArgumentParser(description="Build v0.7 inference-safe Predictive-Utility action dataset")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stage", default="development", choices=["development","formal"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default=None)
    a = ap.parse_args()
    cfg = setup(a.stage); assert_cache_ready(cfg, a.stage)
    rd = Path(a.run_dir).resolve()
    action_path = rd / "forecast_utility_oracle" / "single_patch_candidate_actions.csv.gz"
    retrieval_path = rd / "patch_retrieval" / "patch_retrieval_topk.csv.gz"
    if not action_path.exists(): raise FileNotFoundError(action_path)
    if not retrieval_path.exists(): raise FileNotFoundError(retrieval_path)
    out = Path(a.output).resolve() if a.output else rd / "predictive_utility_selector"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter, cp, ck, source = load_patchstg_adapter(cfg, a.stage, device, a.checkpoint)
    layout = extract_patch_layout(adapter.model, ck, backbone_root=cfg["project"]["backbone_root"])
    val = CacheView(cfg, a.stage, "val")

    log("LOAD forecast-utility labels + patch-local retrieval metadata")
    act = pd.read_csv(action_path)
    ret = pd.read_csv(retrieval_path)
    join_cols = ["query_id","query_pos","patch_id","rank","candidate_id"]
    retrieval_features = ["score","level_mae","delta_mae","overlap_fraction","tod_distance","dow_match",
                          "patch_restorable_count","patch_observed_count"]
    df = act.merge(ret[join_cols + retrieval_features], on=join_cols, how="left", validate="many_to_one")
    if df["score"].isna().any():
        raise RuntimeError(f"Retrieval-feature join failed for {int(df['score'].isna().sum())} rows")
    df["beneficial"] = (df["forecast_utility"] > 0.0).astype(np.int8)
    df["harmful"] = (df["forecast_utility"] < -0.01).astype(np.int8)

    # Query-patch-scope descriptors are tiny (Q*P*scopes) and use observed input/mask only.
    keys = df[["query_id","query_pos","patch_id","recent_steps"]].drop_duplicates().sort_values(
        ["query_pos","recent_steps","patch_id"]
    )
    desc_rows=[]; started=time.time()
    qcache={}
    for i,row in enumerate(keys.itertuples(index=False),1):
        qid=int(row.query_id); pid=int(row.patch_id); L=int(row.recent_steps)
        q=qcache.setdefault(qid,val.row(qid)); nodes=np.asarray(layout.groups[pid],np.int64)
        x=np.asarray(q["x_obs"][:,nodes],np.float32); obs=np.asarray(q["history_observed"][:,nodes],bool)
        H=x.shape[0]; sl=slice(max(0,H-L),H)
        om,os=_stats(x[obs]); rm,rs=_stats(x[sl][obs[sl]])
        desc_rows.append({
            "query_id":qid,"patch_id":pid,"recent_steps":L,
            "query_patch_observed_fraction":float(obs.mean()),
            "query_patch_missing_fraction_scope":float((~obs[sl]).mean()),
            "query_patch_missing_count":int((~obs).sum()),
            "query_patch_obs_mean":om,"query_patch_obs_std":os,
            "query_patch_recent_mean":rm,"query_patch_recent_std":rs,
            "query_patch_delta_abs_mean":_delta_abs(x,obs),
        })
        if i%2000==0 or i==len(keys): log(f"QUERY_FEATURES {i}/{len(keys)} elapsed={time.time()-started:.1f}s")
    qdesc=pd.DataFrame(desc_rows)
    df=df.merge(qdesc,on=["query_id","patch_id","recent_steps"],how="left",validate="many_to_one")
    # Override the v0.6 diagnostic restorable count with a pure observed-mask missing count before feature engineering.
    # The original column is kept only as an archived diagnostic field and is never a model feature.
    df["patch_restorable_count_diagnostic"] = df["patch_restorable_count"]
    df["patch_restorable_count"] = df["query_patch_missing_count"]
    df=add_inference_features(df)
    feats=feature_columns(df)

    # Hard leakage assertion.
    forbidden={"forecast_utility","forecast_mae","baseline_forecast_mae","forecast_gain_pct","patch_completion_mae","beneficial","harmful","candidate_id","query_id"}
    leaked=forbidden.intersection(feats)
    if leaked: raise RuntimeError(f"Leakage feature(s): {sorted(leaked)}")

    out_csv=out/"utility_action_dataset.csv.gz"
    df.to_csv(out_csv,index=False,compression="gzip")
    meta={
        "version":"0.7.1","rows":int(len(df)),"queries":int(df.query_id.nunique()),"patches":int(layout.patch_count),
        "scopes":sorted(df.recent_steps.unique().astype(int).tolist()),"feature_count":len(feats),"features":feats,
        "beneficial_rate":float(df.beneficial.mean()),"harmful_rate":float(df.harmful.mean()),
        "checkpoint":str(cp),"checkpoint_source":source,
        "leakage_contract":"Model features use only observed current history/mask, patch identity, train-retrieval metadata, rank and time context. Hidden history and future targets are labels/diagnostics only.",
        "output":str(out_csv),
    }
    write_json(out/"dataset_summary.json",meta)
    log(f"DATASET={out_csv}")
    log(json.dumps({k:v for k,v in meta.items() if k!='features'},indent=2))

if __name__=="__main__": main()
