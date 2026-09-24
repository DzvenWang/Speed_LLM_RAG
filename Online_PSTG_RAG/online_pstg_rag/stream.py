from __future__ import annotations
from collections import deque
import time
import numpy as np
import pandas as pd
from .metrics import MetricAccumulator, improvement_pct, query_masked_mae
from .memory import RecentResidualMemory, MemoryCase
from .model_io import predict_raw_and_key
from .retriever import retrieve
from .fusion import fuse_residual, oracle_best_candidate
from .utils import log, write_json, sha256_file


def run_stream(cfg, ds, model, ck_path, ck, source, out_dir, device, max_queries=None):
    rcfg=cfg["rag"]
    mean,std=float(ck["mean"]),float(ck["std"])
    base_acc=MetricAccumulator(cfg["experiment"]["horizon"]); rag_acc=MetricAccumulator(cfg["experiment"]["horizon"])
    memory=RecentResidualMemory(int(rcfg["memory_size"]))
    pending=deque(); rows=[]; retrieval_rows=[]; memory_rows=[]
    n=len(ds) if max_queries is None else min(len(ds),int(max_queries))
    t0=time.time(); oracle_base_sum=0.0; oracle_best_sum=0.0; oracle_n=0

    for pos in range(n):
        idx=ds.cache_index(pos); item=ds.item(idx); origin=int(item["origin"])
        matured=0
        while pending and int(pending[0][0]) <= origin:
            future_end, old_idx, old_origin, old_key, old_base, old_tod, old_dow = pending.popleft()
            old=ds.item(old_idx)
            y=old["y_raw"].numpy(); valid=old["future_valid"].numpy()
            residual=(y-old_base).astype(np.float32)
            residual=np.where(valid & np.isfinite(residual), residual, 0.0).astype(np.float32)
            memory.add(MemoryCase(index=int(old_idx),origin=int(old_origin),key=old_key.astype(np.float32),residual=residual,valid=valid.astype(bool),tod=int(old_tod),dow=int(old_dow)))
            memory_rows.append({"stream_pos":pos,"case_index":int(old_idx),"case_origin":int(old_origin),"future_end":int(future_end),"memory_size":len(memory)})
            matured += 1

        base_pred, qkey = predict_raw_and_key(model,item,device,mean,std)
        marr = memory.arrays()
        retrieved = None
        if len(memory) >= int(rcfg["min_memory_cases"]):
            retrieved=retrieve(marr,qkey,origin,int(item["tod_hist"][-1]),int(item["dow_hist"][-1]),rcfg)
        rag_pred, finfo, fdetail = fuse_residual(base_pred,item["direct_mask"].numpy(),marr,retrieved,rcfg,std)

        y=item["y_raw"].numpy(); valid=item["future_valid"].numpy(); direct=item["direct_mask"].numpy(); support=item["support_mask"].numpy()
        base_acc.update(base_pred,y,valid,direct,support); rag_acc.update(rag_pred,y,valid,direct,support)
        bm=query_masked_mae(base_pred,y,valid,direct); rm=query_masked_mae(rag_pred,y,valid,direct)
        oracle_mae=float("nan"); oracle_gain=float("nan")
        if bool(rcfg.get("oracle_diagnostic",True)) and retrieved is not None:
            oracle_mae,oracle_gain=oracle_best_candidate(base_pred,y,valid,direct,marr,retrieved,rcfg,std)
            if np.isfinite(oracle_mae) and np.isfinite(bm):
                oracle_base_sum += bm; oracle_best_sum += oracle_mae; oracle_n += 1

        rows.append({
            "stream_pos":pos,"cache_index":idx,"window_start":int(item["start"]),"origin":origin,"future_end":int(item["future_end"]),
            "matured_now":matured,"memory_size":len(memory),"retrieved":retrieved is not None,"retrieved_k":0 if retrieved is None else len(retrieved["local"]),
            "base_masked_mae":bm,"rag_masked_mae":rm,"masked_mae_delta":rm-bm,
            "gate":finfo["gate"],"alpha_eff":finfo["alpha_eff"],"correction_abs_mean":finfo["correction_abs_mean"],
            "mean_latent_sim":finfo.get("mean_latent_sim",float("nan")),"max_latent_sim":finfo.get("max_latent_sim",float("nan")),
            "mean_age":finfo.get("mean_age",float("nan")),"residual_disagreement":finfo.get("residual_disagreement",float("nan")),
            "oracle_topk_masked_mae":oracle_mae,"oracle_topk_query_gain_pct":oracle_gain,
        })
        if retrieved is not None:
            w = fdetail["weights"] if fdetail is not None else np.full(len(retrieved["local"]),np.nan)
            for rank in range(len(retrieved["local"])):
                retrieval_rows.append({
                    "stream_pos":pos,"query_index":idx,"rank":rank+1,"case_index":int(retrieved["case_index"][rank]),
                    "score":float(retrieved["score"][rank]),"latent_sim":float(retrieved["latent_sim"][rank]),
                    "context":float(retrieved["context"][rank]),"recency":float(retrieved["recency"][rank]),"age":float(retrieved["age"][rank]),
                    "weight":float(w[rank]),
                })
        pending.append((int(item["future_end"]),idx,origin,qkey,base_pred,int(item["tod_hist"][-1]),int(item["dow_hist"][-1])))
        if (pos+1)%100==0 or pos==n-1:
            bs=base_acc.summary()["masked_mae"]; rs=rag_acc.summary()["masked_mae"]
            log(f"STREAM {pos+1}/{n} memory={len(memory)} base_masked={bs:.6f} rag_masked={rs:.6f} gain={improvement_pct(bs,rs):+.3f}%")

    sdf=pd.DataFrame(rows); rdf=pd.DataFrame(retrieval_rows); mdf=pd.DataFrame(memory_rows)
    sdf.to_csv(out_dir/"stream_metrics.csv",index=False); rdf.to_csv(out_dir/"retrievals.csv",index=False); mdf.to_csv(out_dir/"memory_events.csv",index=False)
    base=base_acc.summary(); rag=rag_acc.summary()
    gains={k:improvement_pct(base[k],rag[k]) for k in ["overall_mae","masked_mae","support_mae","overall_rmse","masked_rmse","support_rmse"]}
    segments=[]
    if len(sdf):
        cuts=np.linspace(0,len(sdf),6,dtype=int)
        for j in range(5):
            z=sdf.iloc[cuts[j]:cuts[j+1]]
            b=float(z["base_masked_mae"].mean()); a=float(z["rag_masked_mae"].mean())
            segments.append({"segment":j+1,"start_pos":int(cuts[j]),"end_pos":int(cuts[j+1]),"base_query_masked_mae":b,"rag_query_masked_mae":a,"gain_pct":improvement_pct(b,a)})
    oracle={"queries":oracle_n,"base_query_masked_mae":float("nan"),"oracle_topk_query_masked_mae":float("nan"),"gain_pct":float("nan")}
    if oracle_n>0:
        ob=oracle_base_sum/oracle_n; oo=oracle_best_sum/oracle_n
        oracle={"queries":oracle_n,"base_query_masked_mae":ob,"oracle_topk_query_masked_mae":oo,"gain_pct":improvement_pct(ob,oo)}
    summary={
        "version":cfg["version"],"definition":"strictly causal online recent-residual RAG on frozen PatchSTG",
        "stage":ds.stage,"split":ds.split,"queries":n,"history":ds.history,"horizon":ds.horizon,
        "causal_rule":"A case enters retrieval memory only when its complete future has matured (future_end <= current forecast origin). The stored value is its frozen-PatchSTG residual computed only after maturity.",
        "base_checkpoint":str(ck_path),"base_checkpoint_sha256":sha256_file(ck_path),"base_checkpoint_source":source,
        "backbone_trainable_parameters":0,"rag_config":rcfg,"base":base,"rag":rag,"gains_pct":gains,"segments":segments,
        "oracle_topk_diagnostic":oracle,"final_memory_size":len(memory),"elapsed_seconds":time.time()-t0,
    }
    write_json(out_dir/"summary.json",summary)
    return summary
