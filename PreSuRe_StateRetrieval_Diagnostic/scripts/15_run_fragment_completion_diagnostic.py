#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._common import setup, run_dir
from srd.contract import assert_cache_ready
from srd.backbone import load_patchstg_adapter
from srd.data import CacheView
from srd.evaluate import predict_raw_batched
from srd.metrics import masked_mae, improvement_pct
from srd.patch_layout import extract_patch_layout
from srd.patch_transport import suffix_mask
from srd.fragment_completion import compose_fragment_selection, compose_fragment_softmax
from srd.utils import write_json


def _comp_mae(x, truth, mask):
    m = np.asarray(mask, bool) & np.isfinite(x) & np.isfinite(truth)
    if not np.any(m): return float("nan")
    return float(np.abs(np.asarray(x, np.float32)[m] - np.asarray(truth, np.float32)[m]).mean())


def _forecast(adapter, ck, q, variants, device, bs):
    X = np.stack(variants).astype(np.float32)
    tod = np.repeat(q["tod_hist"][None, :], len(X), axis=0)
    dow = np.repeat(q["dow_hist"][None, :], len(X), axis=0)
    pred = predict_raw_batched(adapter, ck, X, tod, dow, device, batch_size=bs)
    y = np.repeat(q["y_raw"][None, :, :], len(X), axis=0)
    fv = np.repeat(q["future_valid"][None, :, :], len(X), axis=0)
    dm = np.repeat(q["direct_mask"][None, :], len(X), axis=0)
    return masked_mae(pred, y, fv, dm)


def _parse_geometries(items):
    out=[]
    for item in items:
        if isinstance(item, str): out.append((item, 3))
        else: out.append((str(item["name"]), int(item.get("block_size", 3))))
    return out


def _scheme_name(mode, geometry, block_size):
    if geometry in {"temporal", "node_block"}:
        return f"{mode}:{geometry}{int(block_size)}"
    return f"{mode}:{geometry}"


def main():
    ap=argparse.ArgumentParser(description="v0.9 fragment retrieval/composition diagnostic")
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--stage", default="development", choices=["development","formal"])
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--patch-size", type=int, default=None)
    ap.add_argument("--patch-count", type=int, default=None)
    ap.add_argument("--scopes", default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--transport", default=None)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    a=ap.parse_args()

    cfg=setup(a.stage); assert_cache_ready(cfg,a.stage)
    fcfg=cfg["fragment_diagnostic"]
    scopes=[int(x) for x in (a.scopes.split(",") if a.scopes else fcfg.get("scopes",[12]))]
    top_k=int(a.top_k or fcfg.get("top_k",50))
    transport=str(a.transport or fcfg.get("transport","affine_alignment"))
    geometries=_parse_geometries(fcfg["geometries"])
    realizable=set(_parse_geometries(fcfg["realizable_geometries"]))
    softmax=set(_parse_geometries(fcfg.get("softmax_geometries",[])))

    out=run_dir(cfg,a.stage,a.run_tag)/"fragment_completion_diagnostic"
    out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter,cp,ck,source=load_patchstg_adapter(cfg,a.stage,device,a.checkpoint)
    layout=extract_patch_layout(adapter.model,ck,backbone_root=cfg["project"]["backbone_root"],patch_size_override=a.patch_size,patch_count_override=a.patch_count)
    train,val=CacheView(cfg,a.stage,"train"),CacheView(cfg,a.stage,"val")
    ret=np.load(a.retrieval)
    qids=ret["query_ids"].astype(np.int64)
    cand_ids=ret["candidate_ids"].astype(np.int64)
    eligible=ret["eligible"].astype(bool)
    Kmax=cand_ids.shape[-1]; top_k=max(1,min(top_k,Kmax))

    per_query=[]; frag_rows=[]
    t0=time.time()
    for qpos,qi in enumerate(qids):
        q=val.row(int(qi)); variants=[q["x_obs"].copy()]; keys=[("baseline",0)]
        meta={}
        for L in scopes:
            target_all=q["restorable"] & suffix_mask(q["restorable"].shape,L)
            # One composed input per scheme. Oracle uses hidden history only, never future targets.
            schemes=[]
            for geometry,bsz in geometries:
                schemes.append(("oracle_completion",geometry,bsz))
                if (geometry,bsz) in realizable:
                    schemes.append(("local_observed",geometry,bsz))
                    schemes.append(("global_top1",geometry,bsz))
                if (geometry,bsz) in softmax:
                    schemes.append(("local_softmax",geometry,bsz))

            for mode,geometry,bsz in schemes:
                x=q["x_obs"].copy(); patch_stats=[]; detail_count=0
                for pid,nodes in enumerate(layout.groups):
                    if not eligible[qpos,pid]: continue
                    ids=cand_ids[qpos,pid,:top_k]
                    if mode=="local_softmax":
                        xp,st=compose_fragment_softmax(q,train,ids,nodes,L,geometry,bsz,transport,cfg,float(fcfg.get("softmax_temperature",0.35)))
                        details=[]
                    else:
                        xp,st,details=compose_fragment_selection(q,train,ids,nodes,L,geometry,bsz,transport,cfg,mode)
                    pm=np.zeros_like(target_all); pm[:,np.asarray(nodes,np.int64)]=True
                    tm=target_all & pm
                    if np.any(tm): x[tm]=xp[tm]
                    patch_stats.append(st); detail_count += len(details)
                    for d in details:
                        frag_rows.append({
                            "query_pos":int(qpos),"query_id":int(qi),"recent_steps":int(L),
                            "mode":mode,"geometry":geometry,"block_size":int(bsz),"patch_id":int(pid),**d
                        })
                label=_scheme_name(mode,geometry,bsz)
                variants.append(x); keys.append((label,L))
                meta[(label,L)]={
                    "completion_mae":_comp_mae(x,q["x_full"],target_all),
                    "fragments":int(sum(int(s.get("fragments",0)) for s in patch_stats)),
                    "selected_fragments":int(sum(int(s.get("selected_fragments",s.get("used_fragments",0))) for s in patch_stats)),
                    "unique_sources_sum":int(sum(int(s.get("unique_sources",s.get("effective_sources",0))) for s in patch_stats)),
                    "detail_count":int(detail_count),
                }

        maes=_forecast(adapter,ck,q,variants,device,a.eval_batch_size)
        base_f=float(maes[0])
        for j,(label,L) in enumerate(keys[1:],1):
            target=q["restorable"] & suffix_mask(q["restorable"].shape,L)
            base_c=_comp_mae(q["x_obs"],q["x_full"],target)
            m=meta[(label,L)]
            mode,geompart=label.split(":",1)
            per_query.append({
                "query_pos":int(qpos),"query_id":int(qi),"recent_steps":int(L),"scheme":label,
                "mode":mode,"geometry":geompart,
                "baseline_forecast_mae":base_f,"forecast_mae":float(maes[j]),
                "forecast_utility":base_f-float(maes[j]),"forecast_gain_pct":improvement_pct(base_f,float(maes[j])),
                "baseline_completion_mae":base_c,"completion_mae":float(m["completion_mae"]),
                "completion_improvement_pct":improvement_pct(base_c,float(m["completion_mae"])),
                "fragments":int(m["fragments"]),"selected_fragments":int(m["selected_fragments"]),
                "source_count_sum":int(m["unique_sources_sum"]),
            })
        if (qpos+1)%5==0 or qpos+1==len(qids):
            elapsed=time.time()-t0
            print(f"[v0.9] QUERY {qpos+1}/{len(qids)} elapsed={elapsed:.1f}s variants={len(variants)-1}",flush=True)

    pdf=pd.DataFrame(per_query); pdf.to_csv(out/"per_query.csv",index=False)
    fdf=pd.DataFrame(frag_rows); fdf.to_csv(out/"fragment_selections.csv.gz",index=False,compression="gzip")
    summary=[]
    for (L,scheme),g in pdf.groupby(["recent_steps","scheme"],sort=True):
        b=float(g.baseline_forecast_mae.mean()); fm=float(g.forecast_mae.mean())
        bc=float(g.baseline_completion_mae.mean()); cm=float(g.completion_mae.mean())
        summary.append({
            "recent_steps":int(L),"scheme":scheme,"mode":str(g["mode"].iloc[0]),"geometry":str(g["geometry"].iloc[0]),
            "baseline_forecast_mae":b,"forecast_mae":fm,"forecast_gain_pct":improvement_pct(b,fm),
            "baseline_completion_mae":bc,"completion_mae":cm,"completion_improvement_pct":improvement_pct(bc,cm),
            "beneficial_forecast_query_rate":float((g.forecast_utility>0).mean()),
            "mean_fragments":float(g.fragments.mean()),"mean_selected_fragments":float(g.selected_fragments.mean()),
            "mean_source_count_sum":float(g.source_count_sum.mean()),
        })
    sm=pd.DataFrame(summary).sort_values(["recent_steps","mode","forecast_gain_pct"],ascending=[True,True,False])
    sm.to_csv(out/"summary_table.csv",index=False)

    oracle=sm[sm["mode"]=="oracle_completion"]
    real=sm[sm["mode"].isin(["local_observed","local_softmax","global_top1"])]
    payload={
        "version":"0.9.0","queries":int(len(qids)),"patch_count":int(layout.patch_count),"top_k":int(top_k),
        "transport":transport,"uses_future_target_for_selection":False,
        "oracle_uses_hidden_history_truth":True,"realizable_modes_use_hidden_history_truth":False,
        "best_fragment_completion_oracle_forecast_gain_pct":float(oracle.forecast_gain_pct.max()) if len(oracle) else float("nan"),
        "best_fragment_completion_oracle_completion_improvement_pct":float(oracle.completion_improvement_pct.max()) if len(oracle) else float("nan"),
        "best_realizable_fragment_forecast_gain_pct":float(real.forecast_gain_pct.max()) if len(real) else float("nan"),
        "best_realizable_fragment_completion_improvement_pct":float(real.completion_improvement_pct.max()) if len(real) else float("nan"),
        "checkpoint":str(cp),"checkpoint_source":source,"results":summary,
    }
    write_json(out/"summary.json",payload)
    print(sm.to_string(index=False),flush=True)
    print(json.dumps({k:v for k,v in payload.items() if k!="results"},indent=2),flush=True)
    print(f"FRAGMENT_DIAGNOSTIC={out}",flush=True)

if __name__=="__main__": main()
