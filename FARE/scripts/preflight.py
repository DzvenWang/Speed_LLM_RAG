#!/usr/bin/env python3
import argparse,json
import _bootstrap
from fare.config import load_config,cache_path,backbone_checkpoint
from fare.data import SupportSplit,graph_from_cache

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',default='TraffiDent_Net_06');p.add_argument('--target',default='speed');p.add_argument('--config');a=p.parse_args()
    cfg=load_config(a.config,a.dataset,a.target);cache=cache_path(cfg,'formal');tr=SupportSplit(cache,'train');val=SupportSplit(cache,'val')
    adj=graph_from_cache(cache);ck=backbone_checkpoint(cfg)
    assert tr.num_nodes==val.num_nodes==adj.shape[0]
    print('FARE_PREFLIGHT_PASS '+json.dumps({'dataset':a.dataset,'target':a.target,'nodes':tr.num_nodes,'train':len(tr),'val':len(val),'cache':str(cache),'checkpoint':str(ck)}),flush=True)
if __name__=='__main__':main()
