#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np,torch
import _bootstrap
from fare.config import load_config,json_write
from fare.runtime import setup
from fare.trainer import load_ckpt,save_ckpt
from fare.data import loader,to_device
from fare.evaluation import _metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--memory');p.add_argument('--output');p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args();ck=load_ckpt(a.checkpoint);ex=ck['config']['experiment'];cfg=load_config(None,ex['dataset'],ex['target'],ex['seed'])
    model,memory,train,val,device,meta=setup(cfg,ck['metadata']['backbone_checkpoint'],a.memory or ck.get('memory_path'))
    load_ckpt(a.checkpoint,model,device);model.eval();model.refresh(memory)
    bases=[];cands=[];gates=[];targets=[];valids=[];directs=[]
    with torch.no_grad():
        for b in loader(val,a.batch_size):
            x=to_device(b,device);o=model(x,memory)
            bases.append(o['base_prediction'].cpu().numpy());cands.append(o['candidate_prediction'].cpu().numpy())
            gates.append(o['final_gate'].cpu().numpy());targets.append(b['y_raw'].numpy());valids.append(b['future_valid'].numpy());directs.append(b['direct_mask'].numpy())
    base=np.concatenate(bases);cand=np.concatenate(cands);gate=np.concatenate(gates)
    y=np.concatenate(targets);v=np.concatenate(valids);d=np.concatenate(directs)
    best=None
    for multiplier in (0,.25,.5,.75,1,1.25,1.5,2):
        pred=((base+gate*multiplier*(cand-base))*meta['std']+meta['mean'])
        metrics=_metrics(pred,y,v,d)
        objective=metrics['MAE']+.05*metrics['RMSE']
        if best is None or objective<best['objective']:best={'multiplier':multiplier,'objective':objective,'metrics':metrics}
    # Scale gating externally via calibrated post-controller multiplier.
    # The model stores it in the final gate, consistent in future evaluation.
    model.calibration_multiplier.fill_(float(best['multiplier']))
    dest=Path(a.output) if a.output else Path(a.checkpoint).parent/'best.pt'
    save_ckpt(dest,model,None,cfg,meta,0,'calibrated',best,a.memory or ck.get('memory_path'))
    json_write(dest.parent/'calibration.json',best)
    print('FARE_CALIBRATION_PASS '+json.dumps({'checkpoint':str(dest),**best}),flush=True)
if __name__=='__main__':main()
