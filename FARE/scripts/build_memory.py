#!/usr/bin/env python3
import argparse
from pathlib import Path
import _bootstrap
from fare.config import load_config,cache_path,project_root
from fare.memory import TrainMemory

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='TraffiDent_Net_06');p.add_argument('--target',choices=['speed','flow'],default='speed');p.add_argument('--config');p.add_argument('--output');p.add_argument('--max-rows',type=int)
    a=p.parse_args();cfg=load_config(a.config,a.dataset,a.target)
    cache=cache_path(cfg,'formal');mem=TrainMemory.from_cache(cache,a.max_rows or cfg['memory']['max_rows'],sampling=cfg['memory'].get('sampling','spread'))
    path=Path(a.output) if a.output else project_root()/'FARE_artifacts'/a.target/a.dataset/'memory.npz'
    mem.save(path,{'dataset':a.dataset,'target':a.target,'source_cache':str(cache),'train_only':True})
    print(f'FARE_MEMORY_PASS file={path} windows={len(mem.x)} nodes={mem.x.shape[-1]}',flush=True)
if __name__=='__main__':main()
