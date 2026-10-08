#!/usr/bin/env python3
import argparse
from pathlib import Path
import _bootstrap
from fare.config import load_config
from fare.runtime import setup
from fare.trainer import load_ckpt,seed_all
from fare.oof import build_oof,train_controller

def main():
    a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--memory');a.add_argument('--output');a.add_argument('--smoke',action='store_true');q=a.parse_args()
    ck=load_ckpt(q.checkpoint);ex=ck['config']['experiment'];cfg=load_config(None,ex['dataset'],ex['target'],ex['seed'],q.smoke)
    seed_all(ex['seed']);model,memory,train,val,device,meta=setup(cfg,ck['metadata']['backbone_checkpoint'],q.memory or ck.get('memory_path'))
    load_ckpt(q.checkpoint,model,device)
    out=Path(q.output) if q.output else Path(q.checkpoint).parent/'oof'
    path=build_oof(model,memory,train,cfg,meta,device,out)
    loss=train_controller(model.controller,path,cfg,device)
    from fare.trainer import save_ckpt
    save_ckpt(Path(q.checkpoint).parent/'controller_best.pt',model,None,cfg,meta,0,'controller',{'oof_val_loss':loss},q.memory or ck.get('memory_path'))
    print(f'FARE_OOF_PASS oof={path} control_val_loss={loss:.6f}',flush=True)
if __name__=='__main__':main()
