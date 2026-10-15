#!/usr/bin/env python3
"""Precompute v4 memmapped historical episode store using real Base_SupportBench."""
import _bootstrap,argparse,torch
from fare.config import read_config,root
from fare.pipeline import prepare

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='FARE/configs/pilot.yaml')
    p.add_argument('--dataset',default='TraffiDent_Net_06');p.add_argument('--target',required=True,choices=['speed','flow'])
    p.add_argument('--checkpoint',required=True);p.add_argument('--oos-forecast-file')
    a=p.parse_args();cfg=read_config(a.config,a.dataset,a.target)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _,memory,_,_,_,meta=prepare(cfg,device,a.checkpoint,oos_forecast_file=a.oos_forecast_file)
    print(f"FARE4_MEMORY_PASS rows={memory.m} nodes={memory.n} path={meta['memory_path']}",flush=True)
if __name__=='__main__':main()
