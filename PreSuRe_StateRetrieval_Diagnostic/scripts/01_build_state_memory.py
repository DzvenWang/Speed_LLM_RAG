#!/usr/bin/env python3
from pathlib import Path
import argparse, sys, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts._common import setup,run_dir,default_tag
from srd.contract import assert_cache_ready
from srd.backbone import load_patchstg_adapter
from srd.data import CacheView
from srd.memory import build_memory
from srd.utils import write_json,sha256_file


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--stage",default="development",choices=["development","formal"]); ap.add_argument("--run-tag",default=None); ap.add_argument("--checkpoint",default=None); ap.add_argument("--max-windows",type=int,default=None); a=ap.parse_args()
    cfg=setup(a.stage); ident=assert_cache_ready(cfg,a.stage); tag=a.run_tag or default_tag("memory"); out=run_dir(cfg,a.stage,tag)/"memory"; out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); adapter,cp,ck,source=load_patchstg_adapter(cfg,a.stage,device,a.checkpoint); train=CacheView(cfg,a.stage,"train")
    p=build_memory(cfg,adapter,ck,train,device,out/"state_memory.npz",a.max_windows)
    write_json(out/"summary.json",{"stage":a.stage,"run_tag":tag,"memory":str(p),"memory_sha256":sha256_file(p),"checkpoint":str(cp),"checkpoint_source":source,"train_rows":len(train),"cache_identity":ident})
    print(f"STATE MEMORY READY: {p}")
if __name__=="__main__": main()
