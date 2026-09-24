from __future__ import annotations
from pathlib import Path
import sys, time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from srd.config import load_config
from srd.utils import seed_all


def setup(stage):
    cfg=load_config(ROOT); cfg["experiment"]["stage"]=stage; seed_all(cfg["experiment"]["seed"]); return cfg


def run_dir(cfg,stage,run_tag):
    p=Path(cfg["project"]["exp_root"])/cfg["experiment"]["subgraph"]/stage/run_tag
    p.mkdir(parents=True,exist_ok=True); return p


def default_tag(prefix): return time.strftime(prefix+".%Y%m%d_%H%M%S")
