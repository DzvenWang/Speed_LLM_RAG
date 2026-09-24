from __future__ import annotations
from collections import deque
import copy, time
import numpy as np
import pandas as pd
import torch
from .metrics import MetricAccumulator, improvement_pct, query_masked_mae
from .model_io import forward_norm
from .updater import OnlineUpdater
from .utils import log, write_json, sha256_file


@torch.no_grad()
def predict_raw(model, item, device, mean, std):
    batch={k:item[k].unsqueeze(0) for k in ["x_obs","y_raw","y_norm","future_valid","direct_mask","support_mask","tod_hist","dow_hist"]}
    model.eval(); pn=forward_norm(model,batch,device)
    return (pn*float(std)+float(mean)).detach().cpu().numpy()[0]


def run_stream(cfg, ds, base_model, ck_path, ck, source, variant, out_dir, device, max_queries=None):
    ocfg=cfg["online"]; vcfg=cfg["variants"][variant]
    frozen=copy.deepcopy(base_model).to(device).eval()
    online=copy.deepcopy(base_model).to(device).eval()
    updater=OnlineUpdater(online,ds,device,vcfg,ocfg,seed=cfg["experiment"]["seed"])
    mean,std=float(ck["mean"]),float(ck["std"])
    static_acc=MetricAccumulator(cfg["experiment"]["horizon"]); online_acc=MetricAccumulator(cfg["experiment"]["horizon"])
    replay=deque(maxlen=int(ocfg["replay_size"])); pending=deque(); rows=[]; updates=[]
    n=len(ds) if max_queries is None else min(len(ds),int(max_queries))
    t0=time.time(); update_every=max(1,int(ocfg["update_every_queries"]))
    for pos in range(n):
        idx=ds.cache_index(pos); item=ds.item(idx); origin=int(item["origin"])
        matured_now=[]
        while pending and int(pending[0][0]) <= origin:
            _,old_idx=pending.popleft(); replay.append(int(old_idx)); matured_now.append(int(old_idx))
        update_info={"updated":False,"loss":float("nan"),"steps":0,"param_delta_l2":0.0}
        if bool(ocfg.get("update_before_predict",True)) and pos % update_every == 0:
            update_info=updater.update(list(replay))
            if update_info["updated"]:
                updates.append({"stream_pos":pos,"current_index":idx,"origin":origin,"buffer_size":len(replay),**update_info})
        p_static=predict_raw(frozen,item,device,mean,std); p_online=predict_raw(online,item,device,mean,std)
        y=item["y_raw"].numpy(); valid=item["future_valid"].numpy(); direct=item["direct_mask"].numpy(); support=item["support_mask"].numpy()
        static_acc.update(p_static,y,valid,direct,support); online_acc.update(p_online,y,valid,direct,support)
        sm=query_masked_mae(p_static,y,valid,direct); om=query_masked_mae(p_online,y,valid,direct)
        rows.append({
            "stream_pos":pos,"cache_index":idx,"window_start":int(item["start"]),"origin":origin,"future_end":int(item["future_end"]),
            "matured_now":len(matured_now),"replay_size":len(replay),"did_update":bool(update_info["updated"]),
            "update_loss_norm":update_info["loss"],"param_delta_l2":update_info["param_delta_l2"],
            "static_masked_mae":sm,"online_masked_mae":om,"masked_mae_delta":om-sm,
        })
        pending.append((int(item["future_end"]),idx))
        if (pos+1)%100==0 or pos==n-1:
            log(f"STREAM {pos+1}/{n} replay={len(replay)} updates={updater.update_count} static_masked={static_acc.summary()['masked_mae']:.6f} online_masked={online_acc.summary()['masked_mae']:.6f}")
    sdf=pd.DataFrame(rows); udf=pd.DataFrame(updates)
    sdf.to_csv(out_dir/"stream_metrics.csv",index=False); udf.to_csv(out_dir/"updates.csv",index=False)
    static=static_acc.summary(); adapted=online_acc.summary()
    gains={k:improvement_pct(static[k],adapted[k]) for k in ["overall_mae","masked_mae","support_mae","overall_rmse","masked_rmse","support_rmse"]}
    # Chronological quartiles expose whether adaptation improves later in the stream.
    segments=[]
    if len(sdf):
        cuts=np.linspace(0,len(sdf),5,dtype=int)
        for j in range(4):
            z=sdf.iloc[cuts[j]:cuts[j+1]]
            b=float(z["static_masked_mae"].mean()); a=float(z["online_masked_mae"].mean())
            segments.append({"segment":j+1,"start_pos":int(cuts[j]),"end_pos":int(cuts[j+1]),"static_query_masked_mae":b,"online_query_masked_mae":a,"gain_pct":improvement_pct(b,a)})
    summary={
        "version":cfg["version"],"definition":"strictly causal online PatchSTG fine-tuning baseline",
        "variant":variant,"stage":ds.stage,"split":ds.split,"queries":n,"history":ds.history,"horizon":ds.horizon,
        "causal_rule":"A past query can enter replay only when future_end <= current forecast origin; updates occur before current prediction using matured cases only.",
        "base_checkpoint":str(ck_path),"base_checkpoint_sha256":sha256_file(ck_path),"base_checkpoint_source":source,
        "trainable_mode":vcfg["trainable"],"trainable_parameters":int(sum(p.numel() for p in online.parameters() if p.requires_grad)),
        "online_config":ocfg,"variant_config":vcfg,"updates":updater.update_count,"optimizer_steps":updater.step_count,
        "static":static,"online":adapted,"gains_pct":gains,"segments":segments,
        "elapsed_seconds":time.time()-t0,
    }
    if bool(ocfg.get("save_final_checkpoint",True)) and variant!="frozen":
        torch.save({"format":"online_patchstg_v1","version":cfg["version"],"variant":variant,"state_dict":online.state_dict(),
                    "base_checkpoint":str(ck_path),"base_checkpoint_sha256":sha256_file(ck_path),"stream_queries":n,"updates":updater.update_count,
                    "optimizer_steps":updater.step_count,"online_config":ocfg,"variant_config":vcfg},out_dir/"final_online.pt")
    write_json(out_dir/"summary.json",summary)
    return summary
