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
