#!/usr/bin/env python3
"""OFFLINE Validation-only ranking diagnostics on the SAME legal Top-K pool.

Never export the oracle weights as model parameters or use Validation oracle to
train or tune a production retriever. The optional expensive forecast probe
re-runs supported prediction under alternative fixed-pool weights.
"""
from __future__ import annotations
import _bootstrap
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from fare.config import config_from_checkpoint,json_write
from fare.runtime import setup
from fare.trainer import load_ckpt
from fare.data import loader,to_device
from fare.ranking import target_candidate_errors

INPUT_KEYS=('x_obs','obs_mask','tod','dow','window_start')

def normalized_weights(weights,valid):
    p=weights.float()*valid.float()
    return p/p.sum(-1,keepdim=True).clamp_min(1e-8)

def fixed_pool_weights(retrieved,errors,label_valid,variant):
    valid=retrieved['valid']
    if variant=='learned':
        p=retrieved['weights']
    elif variant=='uniform':
        p=valid.float()
    elif variant=='state_oracle':
        # PRIVILEGED DIAGNOSTIC ONLY; depends on x_full withheld at inference.
        scores=errors.masked_fill(~label_valid,float('inf'))
        idx=scores.argmin(-1)
        p=torch.zeros_like(retrieved['weights']).scatter(-1,idx[...,None],1.)
        p=p*label_valid.any(-1,keepdim=True)
    elif variant.startswith('onehot_'):
        k=int(variant.split('_')[1]);p=torch.zeros_like(retrieved['weights'])
        if k<p.shape[-1]:p[...,k]=1.
    else: raise ValueError(variant)
    # If a diagnostic proposal has no valid mass despite legal candidates,
    # fall back to the learned distribution rather than fabricating zero support.
    p=normalized_weights(p,valid)
    fallback=normalized_weights(retrieved['weights'],valid)
    use_fallback=(p.sum(-1,keepdim=True)<1e-8)&valid.any(-1,keepdim=True)
    return torch.where(use_fallback,fallback,p)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--max-batches',type=int,default=2,help='Default 2 for affordable forecast probing; 0 means all Validation batches')
    p.add_argument('--batch-size',type=int,default=None)
    p.add_argument('--forecast-probe',action='store_true',help='Also evaluate each Top-K candidate under a fixed pool (costly)')
    p.add_argument('--output')
    args=p.parse_args()
    ck=load_ckpt(args.checkpoint)
    cfg=config_from_checkpoint(ck)
    model,memory,train,val,device,meta=setup(cfg,ck['metadata']['backbone_checkpoint'],ck.get('memory_path'))
    load_ckpt(args.checkpoint,model,device)
    model.eval();model.refresh(memory)
    output=Path(args.output) if args.output else Path(args.checkpoint).resolve().parent/'fixed_pool_diagnostic'
    output.mkdir(parents=True,exist_ok=True)
    metrics=[];probe=[]
    variants=['learned','uniform','state_oracle']
    with torch.no_grad():
        for batch_no,batch in enumerate(loader(val,args.batch_size or cfg['training']['batch_size'],workers=0)):
            if args.max_batches and batch_no>=args.max_batches:break
            bt=to_device(batch,device)
            inp={key:bt[key] for key in INPUT_KEYS}
            ctx=model.regions(inp['x_obs'],inp['obs_mask'])
            obs=model.observability(ctx)
            retrieved=model.retrieval(ctx,model.regions,memory,inp['window_start'],observability=obs)
            errors,labels=target_candidate_errors(bt,retrieved,memory,model.regions)
            eligible=(labels.sum(-1)>=2)
            label_mass=(retrieved['weights'].detach()*labels.float()).sum(-1)
            source=normalized_weights(retrieved['weights']*labels.float(),labels)
            uniform=normalized_weights(torch.ones_like(source),labels)
            best=errors.masked_fill(~labels,float('inf')).min(-1).values
            selected=(errors*source).sum(-1)
            avg=(errors*uniform).sum(-1)
            def summarize(a):
                return float(a[eligible].mean().item()) if eligible.any() else None
            row={'batch':batch_no,'rows':len(inp['x_obs']),
                 'eligible_group_fraction':float(eligible.float().mean()),
                 'retrieval_valid_fraction':float(retrieved['group_valid'].float().mean()),
                 'target_valid_weight_mass':summarize(label_mass),
                 'selected_state_error':summarize(selected),
                 'uniform_state_error':summarize(avg),
                 'oracle_state_error':summarize(best)}
            if args.forecast_probe:
                # Reweight precisely the same retrieved candidate IDs for all variants;
                # the label-based oracle here is a non-deployable diagnostic ceiling.
                for kind in variants+[f'onehot_{j}' for j in range(retrieved['ids'].shape[-1])]:
                    weights=fixed_pool_weights(retrieved,errors,labels,kind)
                    override={**retrieved,'weights':weights}
                    out=model(inp,memory,retrieval_override=override)
                    y=bt['y_raw'];mean=float(meta['mean']);std=float(meta['std'])
                    base=out['base_prediction']*std+mean
                    prediction=out['prediction']*std+mean
                    candidate=out['candidate_prediction']*std+mean
                    valid=bt['future_valid'].bool()
                    direct=valid&bt['direct_mask'].bool()[:,None,:]
                    if not direct.any():direct=valid
                    base_err=(base-y).abs()[direct].mean()
                    final_err=(prediction-y).abs()[direct].mean()
                    candidate_err=(candidate-y).abs()[direct].mean()
                    rank_state_error=None
                    if kind.startswith('onehot_'):
                        kth=int(kind.split('_')[1])
                        if labels[...,kth].any():
                            rank_state_error=float(errors[...,kth][labels[...,kth]].mean())
                    probe.append({'batch':batch_no,'strategy':kind,
                                  'candidate_state_error':rank_state_error,
                                  'MAE_base_direct':float(base_err),
                                  'MAE_final_direct':float(final_err),
                                  'MAE_candidate_direct':float(candidate_err),
                                  'final_gain_direct':float(base_err-final_err),
                                  'candidate_gain_direct':float(base_err-candidate_err)})
            metrics.append(row)
            print('FARE_FIXED_POOL_BATCH '+json.dumps(row),flush=True)
    if not metrics:raise RuntimeError('Empty validation diagnostics')
    def csv_write(name,data):
        if not data:return
        keys=list(dict.fromkeys(k for d in data for k in d.keys()))
        with (output/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
    csv_write('fixed_pool_ranking.csv',metrics);csv_write('fixed_pool_forecast.csv',probe)
    onehot=[q for q in probe if q['strategy'].startswith('onehot_') and q['candidate_state_error'] is not None]
    correlation=None
    if len(onehot)>=3:
        state=np.array([q['candidate_state_error'] for q in onehot])
        forecast=np.array([q['candidate_gain_direct'] for q in onehot])
        if np.std(state)>1e-8 and np.std(forecast)>1e-8:
            correlation=float(np.corrcoef(state,forecast)[0,1])
    report={'version':'1.3.1','dataset':cfg['experiment']['dataset'],'target':cfg['experiment']['target'],
            'checkpoint':str(Path(args.checkpoint).resolve()),'memory_sha256':ck.get('memory_path'),
            'max_batches':args.max_batches,'pool_identity':'Top-K IDs constant across all strategies within each batch',
            'ranking':metrics,'forecast_probe':probe,
            'state_error_vs_candidate_forecast_gain_pearson':correlation,
            'warning':'state_oracle uses masked target labels OFFLINE ONLY, never inference or learned calibration'}
    json_write(output/'fixed_pool_summary.json',report)
    print(f'FARE_FIXED_POOL_PASS output={output} batches={len(metrics)} forecast_probes={len(probe)}',flush=True)
if __name__=='__main__':main()
