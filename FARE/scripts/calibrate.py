#!/usr/bin/env python3
"""Fit post-OOF correction multiplier strictly on VALIDATION, save final best.pt."""
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
import _bootstrap
from fare.config import config_from_checkpoint,json_write
from fare.runtime import setup
from fare.trainer import load_ckpt,save_ckpt
from fare.data import loader,to_device
from fare.evaluation import _metrics
from fare.alpha import DEFAULT_MULTIPLIERS

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--memory');p.add_argument('--output')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--multipliers',type=float,nargs='+',default=list(DEFAULT_MULTIPLIERS))
    a=p.parse_args();ck=load_ckpt(a.checkpoint)
    cfg=config_from_checkpoint(ck)
    model,memory,train,val,device,meta=setup(cfg,ck['metadata']['backbone_checkpoint'],a.memory or ck.get('memory_path'))
    load_ckpt(a.checkpoint,model,device);model.eval();model.refresh(memory)
    if abs(float(model.calibration_multiplier)-1.)>1e-5:
        raise ValueError('Expected uncalibrated checkpoint; do not recursively recalibrate best.pt')
    bases=[];cands=[];gates=[];targets=[];valids=[];directs=[]
    with torch.no_grad():
        for b in loader(val,a.batch_size):
            x=to_device(b,device);o=model(x,memory)
            bases.append(o['base_prediction'].cpu().numpy());cands.append(o['candidate_prediction'].cpu().numpy())
            gates.append(o['final_gate'].cpu().numpy());targets.append(b['y_raw'].numpy())
            valids.append(b['future_valid'].numpy());directs.append(b['direct_mask'].numpy())
    base=np.concatenate(bases);cand=np.concatenate(cands);gate=np.concatenate(gates)
    y=np.concatenate(targets);v=np.concatenate(valids);d=np.concatenate(directs)
    rows=[]
    for multiplier in a.multipliers:
        if multiplier<0:raise ValueError('Multiplier must be non-negative')
        # Exact same clipped formula as the inference-time model.forward.
        effective_gate=np.clip(gate*multiplier,0.,model.controller.max_alpha)
        pred=((base+effective_gate*(cand-base))*meta['std']+meta['mean'])
        metrics=_metrics(pred,y,v,d)
        objective=metrics['MAE']+.05*metrics['RMSE']
        rows.append({'multiplier':float(multiplier),'objective':float(objective),
                     'MAE':float(metrics['MAE']),'RMSE':float(metrics['RMSE']),
                     'directly_masked_MAE':float(metrics['directly_masked_MAE']),
                     'gate_mean':float(effective_gate.mean())})
    best=min(rows,key=lambda x:(x['objective'],x['multiplier']))
    model.calibration_multiplier.fill_(best['multiplier'])
    dest=Path(a.output) if a.output else Path(a.checkpoint).parent/'best.pt'
    save_ckpt(dest,model,None,cfg,meta,0,'calibrated',best,a.memory or ck.get('memory_path'))
    report={'selection_split':'val','objective':'MAE+0.05*RMSE',
            'source_checkpoint':str(Path(a.checkpoint).resolve()),
            'max_alpha':model.controller.max_alpha,'best':best,'grid':rows}
    json_write(dest.parent/'calibration.json',report)
    with (dest.parent/'calibration_grid.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=tuple(rows[0]));writer.writeheader();writer.writerows(rows)
    print('FARE_CALIBRATION_PASS '+json.dumps({'checkpoint':str(dest),'best':best}),flush=True)
if __name__=='__main__':main()
