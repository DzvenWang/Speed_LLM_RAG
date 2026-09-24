#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, json
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from srd.config import load_config
from srd.contract import assert_cache_ready
from srd.backbone import load_patchstg_adapter
from srd.data import CacheView
from srd.evaluate import predict_raw
from srd.metrics import masked_mae
from srd.utils import seed_all, write_json


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",default="development",choices=["development","formal"]); ap.add_argument("--checkpoint",default=None); ap.add_argument("--queries",type=int,default=64); a=ap.parse_args()
    cfg=load_config(ROOT); seed_all(cfg["experiment"]["seed"]); ident=assert_cache_ready(cfg,a.stage); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter,cp,ck,source=load_patchstg_adapter(cfg,a.stage,device,a.checkpoint); val=CacheView(cfg,a.stage,"val")
    ids=np.arange(min(a.queries,len(val)),dtype=np.int64); maes=[]
    for i in ids:
        q=val.row(i); p=predict_raw(adapter,ck,q["x_obs"],q["tod_hist"],q["dow_hist"],device)[0]; maes.append(masked_mae(p,q["y_raw"],q["future_valid"],q["direct_mask"])[0])
    out={"stage":a.stage,"queries":len(ids),"mean_per_query_masked_mae":float(np.nanmean(maes)),"checkpoint":str(cp),"checkpoint_source":source,"device":str(device),"cache_identity":ident}
    print(json.dumps(out,indent=2,default=str)); print("BACKBONE CHECK PASS")
if __name__=="__main__": main()
