"""Fully automatic v9.3 response and forecast gain audit.

Valid/TEST labels enter only loss/metrics. No full reference is loaded for val.
"""
from __future__ import annotations
import json,csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from ..dataset import DirectViewDataset
from ..model import TCRRV93,bounded
from ..teacher import atomic_json
from ..training import to_device,_check_calibration,make_loader
from ...v5.calibration import chronological_partitions


def diagnose_run(run,cache,teacher_root,settings,ck,adapter,root,target,dataset,model=None):
    run=Path(run);folder=run/'diagnostics';folder.mkdir(parents=True,exist_ok=True)
    output_ck=run/'best_cf_joint.pt'
    if not output_ck.is_file():output_ck=run/'best_utility.pt'
    if not output_ck.is_file():output_ck=run/'best_residual.pt'
    if not output_ck.is_file():raise FileNotFoundError('No trained v9.3 model')
    from ...config import load_config
    from ..graph_encoder import functional_graph
    from ..road_graph import load_or_build_graph
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None:
        meta=json.loads((Path(teacher_root)/'train/manifest.json').read_text())
        graph_mode=settings['graph']['mode']
        road=None;func=None
        if graph_mode!='none':
            road=load_or_build_graph(root,target,dataset,cache,run/'graph',settings['graph']['road_k'])
            func=functional_graph(cache,k=settings['graph']['functional_k'],
                                  max_windows=settings['graph']['functional_max_windows'])
        model=TCRRV93(meta['context_dim'],meta['nodes'],meta['horizon'],
                   settings['model']['hidden'],settings['model']['correction_bound'],
                   settings['model']['mode'],graph_mode,road,func)
    payload=torch.load(output_ck,map_location='cpu',weights_only=False)
    model.load_state_dict(payload['model'],strict=True)
    model.to(device);model.eval()
    val=DirectViewDataset(cache,teacher_root,'val',limit=settings['preparation']['val_limit'],views=1)
    selection=chronological_partitions(val.ds.window_starts[:len(val.ds)],
                           settings['model']['history'],settings['model']['horizon'],
                           smoke=bool(json.loads((run/'summary.json').read_text()).get('smoke',False)))
    calibration=json.loads((run/'calibration.json').read_text())
    check=_check_calibration(model,val,selection['check'],device,
                            settings['training']['eval_batch_size'],0,calibration['alpha'])
    atomic_json(folder/'locked_check.json',check)
    # CF response quality on held-out TRAIN queries only.
    hold=DirectViewDataset(cache,teacher_root,'train',limit=settings['preparation']['limit'],
                           train_fraction=settings['training']['train_fraction'],heldout=True)
    loader=make_loader(hold,settings['training']['eval_batch_size'],0,False,8123)
    sums={};counts={};n_total=0;correct=0;nonzero=0;pair_total=0;pair_error=0
    with torch.inference_mode():
        for b in loader:
            b=to_device(b,device);o=model(b['student']);v=b['future_valid'].bool()
            true=b['cf_teacher'].float();pred=o['cf_response'].float();sel=(true.abs()>.05)&v
            correct+=int(((true.sign()==pred.sign())&sel).sum());nonzero+=int(sel.sum())
            key=(int(b['family'][0]),int(b['severity'][0]))
            for j in range(len(b['family'])):
                key=(int(b['family'][j]),int(b['severity'][j]));valid=v[j];count=int(valid.sum())
                sums[key]=sums.get(key,0.)+float(((pred[j]-true[j]).abs()*valid).sum())
                counts[key]=counts.get(key,0)+count
            n_total+=int(v.sum())
    rows=[dict(family_id=k[0],severity=k[1],cf_response_mae_norm=sums[k]/max(counts[k],1),
               valid_positions=counts[k]) for k in sorted(sums)]
    with open(folder/'cf_response_by_failure.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ('family_id','severity','cf_response_mae_norm','valid_positions'))
        writer.writeheader();writer.writerows(rows)
    result=dict(version='9.3.0',locked_check=check,
             cf_response_mae_norm=sum(sums.values())/max(n_total,1),
             cf_sign_agreement_active=correct/max(nonzero,1),
             active_cf_positions=nonzero,source='heldout_TRAIN_only',
             model_checkpoint=output_ck.name,
             limitations=['No retrained ablation may be inferred from same-checkpoint audit',
                          'Oracle reference forecasts never enter VAL/TEST forward'])
    atomic_json(folder/'summary.json',result)
    atomic_json(folder/'recommendations.json',dict(
        cf_gain_positive=check['cf_gain_norm']>0,
        action=('Keep CF for ablation' if check['cf_gain_norm']>0 else 'CF gain unproven: compare response accuracy and utility control'),
        graph_model=settings['graph']['mode'],
        retrieval_model='none (original v9.2 remains independent baseline)'))
    print('V93_DIAGNOSTIC_PASS '+json.dumps(result),flush=True)
    return result
