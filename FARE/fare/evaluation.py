"""Formal metrics strictly from Base_SupportBench, with aligned inference traces."""
import numpy as np,torch,json
from pathlib import Path
from .protocol import make_loader,device_batch
from .config import add_shared_paths,dump_json

def canonical_metric_bundle():
    add_shared_paths()
    try:
        from experiment_common.metrics import metric_bundle
    except Exception as exc:
        raise RuntimeError('Base_SupportBench canonical experiment_common.metrics.metric_bundle is required') from exc
    return metric_bundle

@torch.no_grad()
def evaluate(model,memory,ds,device,cfg,output,mode='full',save_traces=True):
    model.eval();p=[];y=[];v=[];dm=[];origin=[];rows=[];alphas=[];nulls=[];gains=[];fw=[];bw=[];reg=[];res_use=[];evo_use=[]
    for raw in make_loader(ds,cfg['training']['batch_size'],workers=cfg['training']['workers']):
        bt=device_batch(raw,device)
        out=model(bt,memory,intervention=mode,purge=cfg['memory']['purge_steps'],return_details=False)
        pred=(out['prediction']*model.backbone.std+model.backbone.mean).cpu().numpy()
        p.append(pred);y.append(raw['target_raw'].numpy());v.append(raw['future_valid'].numpy());dm.append(raw['direct_mask'].numpy())
        rows.extend(raw['row_id'].tolist());origin.extend(raw['origin'].tolist())
        alphas.append(out['alpha'].cpu().numpy());nulls.append(out['selection']['null'].cpu().numpy());
        gains.append(out['gain_estimate'].cpu().numpy())
        fw.append(out['support']['forward_coverage'].cpu().numpy());bw.append(out['support']['backward_coverage'].cpu().numpy())
        reg.append(out['support']['regime'].cpu().numpy())
        w=out['selection']['weights']
        res_use.append((w[:,:,:,0,:].sum((1,2,3))/max(w.shape[1]*w.shape[-1],1)).cpu().numpy())
        evo_use.append((w[:,:,:,1,:].sum((1,2,3))/max(w.shape[1]*w.shape[-1],1)).cpu().numpy())
    data=dict(prediction=np.concatenate(p),target=np.concatenate(y),valid=np.concatenate(v),direct=np.concatenate(dm),
              row_id=np.asarray(rows),origin=np.asarray(origin),alpha=np.concatenate(alphas),
              null_weight=np.concatenate(nulls),predicted_gain=np.concatenate(gains),
              forward_coverage=np.concatenate(fw),backward_coverage=np.concatenate(bw),regime=np.concatenate(reg),
              residual_weight_mean=np.concatenate(res_use),evolution_weight_mean=np.concatenate(evo_use))
    # canonical metrics are the only source of headline MAE/RMSE/MAPE
    result=dict(canonical_metric_bundle()(data['prediction'],data['target'],data['valid'],data['direct']))
    result['evaluation_mode']=mode;result['rows']=len(rows)
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    dump_json(path/f'{mode}.json',result)
    if save_traces:np.savez_compressed(path/f'{mode}.npz',**data)
    return result,data

@torch.no_grad()
def evaluate_modes(model,memory,ds,device,cfg,output,modes,save_traces=False,progress_every=0):
    """Evaluate fixed-checkpoint interventions in ONE pass over query batches.

    Full output is reused by downstream interventions; no repeated Backbone
    prefix and no repeated historical retrieval when a mode affects only risk,
    integration or candidate selection.
    """
    model.eval()
    buffer={m:{'prediction':[],'alpha':[],'null_weight':[],'predicted_gain':[],
               'forward_coverage':[],'backward_coverage':[],'regime':[],
               'residual_weight_mean':[],'evolution_weight_mean':[]} for m in modes}
    common={'target':[],'valid':[],'direct':[],'row_id':[],'origin':[]}
    for batch_i,raw in enumerate(make_loader(ds,cfg['training']['batch_size'],workers=cfg['training']['workers'])):
        batch=device_batch(raw,device)
        full=model(batch,memory,purge=cfg['memory']['purge_steps'],return_details=True)
        for m in modes:
            out=model.intervene_cached(batch,memory,full,m,cfg['memory']['purge_steps'])
            b=buffer[m]
            b['prediction'].append(((out['prediction']*model.backbone.std)+model.backbone.mean).cpu().numpy())
            b['alpha'].append(out['alpha'].cpu().numpy())
            b['null_weight'].append(out['selection']['null'].cpu().numpy())
            b['predicted_gain'].append(out['gain_estimate'].cpu().numpy())
            b['forward_coverage'].append(out['support']['forward_coverage'].cpu().numpy())
            b['backward_coverage'].append(out['support']['backward_coverage'].cpu().numpy())
            b['regime'].append(out['support']['regime'].cpu().numpy())
            w=out['selection']['weights'];den=max(1,w.shape[1]*w.shape[-1])
            b['residual_weight_mean'].append((w[:,:,:,0,:].sum((1,2,3))/den).cpu().numpy())
            b['evolution_weight_mean'].append((w[:,:,:,1,:].sum((1,2,3))/den).cpu().numpy())
        common['target'].append(raw['target_raw'].numpy())
        common['valid'].append(raw['future_valid'].numpy())
        common['direct'].append(raw['direct_mask'].numpy())
        common['row_id'].append(raw['row_id'].numpy())
        common['origin'].append(raw['origin'].numpy())
        if progress_every and (batch_i+1) % int(progress_every)==0:
            print(f'FARE6_AUDIT_MODES_PROGRESS batches={batch_i+1} queries={(batch_i+1)*cfg["training"]["batch_size"]}',flush=True)
    common={k:np.concatenate(v) for k,v in common.items()}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    results={};data={}
    for mode,items in buffer.items():
        trace={**common,**{k:np.concatenate(v) for k,v in items.items()}}
        res=dict(canonical_metric_bundle()(trace['prediction'],trace['target'],trace['valid'],trace['direct']))
        res.update({'evaluation_mode':mode,'rows':len(trace['row_id']),
                    'analysis_scope':'fixed_checkpoint_budgeted' if cfg.get('analysis',{}).get('max_intervention_queries') else 'fixed_checkpoint_full'})
        dump_json(output/f'{mode}.json',res)
        if save_traces:np.savez_compressed(output/f'{mode}.npz',**trace)
        results[mode]=res;data[mode]=trace
    return results,data
