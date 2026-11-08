#!/usr/bin/env python3
"""Blind severity evaluation of Direct v9.3: no full-history/query retrieval use."""
from __future__ import annotations
import sys,os,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from torch.utils.data import DataLoader
from tcrr_stla.config import load_config,cache_path,resolve_stla_checkpoint,file_sha256
from tcrr_stla.v5.shared_adapter import initial_model
from tcrr_stla.data import SupportSplit
from tcrr_stla.v93.model import TCRRV93
from tcrr_stla.v93.graph_encoder import functional_graph
from tcrr_stla.v93.road_graph import load_or_build_graph
from tcrr_stla.v93.training import to_device
from tcrr_stla.v93.teacher import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--adapter-checkpoint',required=True)
    p.add_argument('--target',required=True,choices=('speed','flow'))
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--severity',type=int,required=True,choices=(20,40,60,90))
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run=Path(a.run_dir).resolve();stageck=Path(a.checkpoint).resolve()
    sig=json.loads((run/'run_identity.json').read_text())
    if sig['version']!='9.3.0' or sig['teacher_source']['target']!=a.target or sig['teacher_source']['dataset']!=a.dataset:
        raise RuntimeError('Checkpoint metadata does not belong to selected target/dataset')
    if file_sha256(a.backbone_checkpoint)!=sig['teacher_source']['teacher'] or file_sha256(a.adapter_checkpoint)!=sig['teacher_source']['adapter']:
        raise RuntimeError('Backbone/Adapter are not identical to TRAIN Teacher')
    settings=sig['config'];os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(ROOT/'configs/tcrr_stla_v5.yaml');cfg['experiment'].update(target=a.target,dataset=a.dataset,seed=sig['seed'])
    original=resolve_stla_checkpoint(cfg,a.backbone_checkpoint)
    cache=cache_path(cfg,'evaluation',a.severity)
    train_cache=cache_path(cfg,'formal')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not a.smoke:raise RuntimeError('Formal test requires CUDA')
    teacher,bc=initial_model(cfg,original,device,Path(a.adapter_checkpoint))
    teacher.eval()
    for p in teacher.parameters():p.requires_grad_(False)
    graph_mode=settings['graph']['mode'];road=None;func=None
    if graph_mode!='none':
        road=load_or_build_graph(cfg['project']['root'],a.target,a.dataset,train_cache,run/'graph',
                                 settings['graph']['road_k'])
        func=functional_graph(train_cache,k=settings['graph']['functional_k'],
                              max_windows=settings['graph']['functional_max_windows'])
    N=int(bc['num_nodes']);C=int(teacher.model_dim)
    model=TCRRV93(C,N,settings['model']['horizon'],settings['model']['hidden'],
                  settings['model']['correction_bound'],settings['model']['mode'],graph_mode,road,func).to(device)
    payload=torch.load(stageck,map_location='cpu',weights_only=False)
    if payload.get('signature')!=sig:raise RuntimeError('Loaded weights from a different v9.3 training identity')
    model.load_state_dict(payload['model'],strict=True);model.eval()
    calibration=json.loads((run/'calibration.json').read_text());alpha=float(calibration['alpha'])
    ds=SupportSplit(cache,'test',24 if a.smoke else None,cf_training=False)
    dl=DataLoader(ds,batch_size=a.batch_size,num_workers=0,pin_memory=device.type=='cuda')
    std=float(bc['std']);mean=float(bc['mean'])
    metric={k:dict(ae=0.,se=0.,masked=0.,h12=0.,count=0,masked_count=0,h12_count=0) for k in ('base','normal','full')}
    harms=0;benefits=0;positions=0;activations=0
    with torch.inference_mode():
        for b in dl:
            # Strictly observable TEST inputs. Teacher forward cannot access x_full or y.
            s=to_device({k:b[k] for k in ('x','mask','tod','dow')},device)
            bpred,ctx=teacher(s['x'],s['mask'],s['tod'],s['dow'])
            student=dict(x_obs=s['x'],observed=s['mask'],tod=s['tod'],dow=s['dow'],
                         base=bpred.float(),context=ctx.float())
            out=model(student)
            pred={'base':out['base'],'normal':out['normal_prediction'],
                  'full':out['normal_prediction']+alpha*out['cf_increment']}
            y=b['y_raw'].to(device).float();fv=b['future_valid'].to(device).bool()
            direct=b['direct_mask'].to(device).bool()
            if direct.ndim==2:direct=direct[:,None,:].expand_as(fv)
            if direct.shape!=fv.shape:raise RuntimeError('Wrong direct mask shape')
            direct&=fv
            errors={name:(pred[name].float()*std+mean-y) for name in metric}
            for name,e in errors.items():
                z=metric[name]
                z['ae']+=float(torch.where(fv,e.abs(),0).sum())
                z['se']+=float(torch.where(fv,e.square(),0).sum())
                z['masked']+=float(torch.where(direct,e.abs(),0).sum())
                z['h12']+=float(torch.where(fv[:,-1:],e[:,-1:].abs(),0).sum())
                z['count']+=int(fv.sum());z['masked_count']+=int(direct.sum());z['h12_count']+=int(fv[:,-1:].sum())
            normal=errors['normal'].abs();full=errors['full'].abs()
            harms+=int(((full>normal+1e-6)&fv).sum());benefits+=int(((full<normal-1e-6)&fv).sum())
            activations+=int(((alpha*out['cf_increment']).abs()>1e-6).logical_and(fv).sum())
            positions+=int(fv.sum())
    result={}
    for name,z in metric.items():
        result[name]=dict(mae=z['ae']/max(1,z['count']),rmse=(z['se']/max(1,z['count']))**.5,
                  masked_mae=z['masked']/max(1,z['masked_count']),
                  h12_mae=z['h12']/max(1,z['h12_count']))
    result.update(version='9.3.0',target=a.target,dataset=a.dataset,seed=sig['seed'],
          severity=a.severity,mode=settings['model']['mode'],graph=graph_mode,
          checkpoint=str(stageck),alpha=alpha,
          cf_gain_raw=result['normal']['mae']-result['full']['mae'],
          cf_harm_rate=harms/max(1,positions),cf_benefit_rate=benefits/max(1,positions),
          cf_activation_rate=activations/max(1,positions),
          improvement_vs_base_percent=100*(result['base']['mae']-result['full']['mae'])/max(result['base']['mae'],1e-8),
          test_labels_used_for='reporting_only')
    file=run/f'test_s{a.severity}.json'
    if file.exists():raise FileExistsError('Locked TEST outcome already exists: '+str(file))
    atomic_json(file,result)
    print('V93_TEST '+json.dumps(result),flush=True)
if __name__=='__main__':main()
