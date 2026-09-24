#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from srd.config import load_config
from srd.contract import assert_cache_ready, cache_dir
from srd.backbone import resolve_checkpoint


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",default="development",choices=["development","formal"]); ap.add_argument("--checkpoint",default=None); a=ap.parse_args()
    cfg=load_config(ROOT); print("workspace:",cfg["local_runtime"]["workspace_root"]); print("project:",ROOT)
    required=["backbone_root","backbone_exp_root","common_root","supportbench_root","benchmark_root","cache_root"]
    for k in required:
        p=Path(cfg["project"][k]); print(f"{k}: {p} [{'PASS' if p.exists() else 'MISSING'}]")
        if not p.exists(): raise SystemExit(2)
    ident=assert_cache_ready(cfg,a.stage); c=cache_dir(cfg,a.stage)
    print("cache:",c); print("cache_identity:",json.dumps(ident,indent=2,default=str)[:2500])
    p,ck,source=resolve_checkpoint(cfg,a.stage,a.checkpoint); print("checkpoint:",p); print("checkpoint_source:",source); print("mean/std:",ck.get("mean"),ck.get("std"))
    print("PREFLIGHT PASS")
if __name__=="__main__": main()
