#!/usr/bin/env python3
"""Blind v9 four-severity TEST: no full test query inputs or utility labels used by model."""
from __future__ import annotations
import sys,os,json,argparse,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from torch.utils.data import DataLoader
from tcrr_stla.config import load_config,cache_path,file_sha256
from tcrr_stla.v5.shared_adapter import initial_model
from tcrr_stla.v9.graph import GraphBundle
from tcrr_stla.v9.model import TCRRV9
from tcrr_stla.v9.prepare import prepare_view,V9EvidenceDataset
from tcrr_stla.v9.training import to_device
from scripts.build_retrieval_cache import ensure_cache


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--run-identity',required=True)
    p.add_argument('--severity',type=int,choices=(20,40,60,90),required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--prep-batch',type=int,default=4)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--force-replay',action='store_true',help='Remove only this test severity evidence files')
    a=p.parse_args()
    identity=json.loads(Path(a.run_identity).read_text())
    if identity['version']!='9.1.0':raise RuntimeError('Wrong v9 identity')
    os.environ['TRAFFIDENT_TARGET']=identity['target']
    cfg=load_config(ROOT/'configs/tcrr_stla_v5.yaml')
    cfg['experiment'].update(target=identity['target'],dataset=identity['dataset'],seed=identity['seed'])
    cfg['project']['retrieval_root']=str(Path(cfg['project']['retrieval_root'])/'v9_1_isolated')
    formal=cache_path(cfg,'formal')
    eval_cache=cache_path(cfg,'evaluation',a.severity)
    ckfile=Path(identity['backbone_checkpoint']);adapter=Path(identity['adapter_checkpoint'])
    if file_sha256(ckfile)!=identity['backbone_sha256'] or file_sha256(adapter)!=identity['adapter_sha256']:
        raise RuntimeError('v9 teacher/adapter checksum mismatch')
    out=Path(identity['evidence_root']);out.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint.get('version')!='9.1.0':raise RuntimeError('Not a v9 checkpoint')
    if checkpoint['seed']!=identity['seed']:raise RuntimeError('Seed mismatch')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if a.smoke!=bool(identity['smoke']):raise ValueError('Evaluation smoke setting must match checkpoint identity')
    teacher,bc=initial_model(cfg,ckfile,device,adapter)
    teacher.eval();[p.requires_grad_(False) for p in teacher.parameters()]
    rt=ensure_cache(cfg,eval_cache,'test',severity=a.severity,smoke=a.smoke)
    split_key=f'test_s{a.severity}'
    folder=out/f'{split_key}_view0'
    if a.force_replay and folder.exists():
        import shutil;shutil.rmtree(folder)
    prepare_view(eval_cache,'test',0,teacher,device,out,rt,k=12,chunk=a.prep_batch,
                 seed=identity['seed'],teacher_sha=identity['teacher_fingerprint'],
                 history_cache=formal,split_key=split_key,limit=24 if a.smoke else None)
    del teacher
    if device.type=='cuda':torch.cuda.empty_cache()
    sd=checkpoint['model'];g=GraphBundle(sd['graph.functional'],sd['graph.upstream'],sd['graph.downstream'])
    c=checkpoint['cfg']
    model=TCRRV9(g,checkpoint['context_dim'],hidden=c['hidden'],bound=c['bound'],blocks=c['blocks']).to(device)
    model.load_state_dict(sd,strict=True);model.eval()
    ds=V9EvidenceDataset(eval_cache,'test',views=(0,),root=out,split_key=split_key)
    dl=DataLoader(ds,batch_size=a.batch_size,num_workers=0,pin_memory=device.type=='cuda')
    scores={m:{'sum_abs':0.,'sum_sq':0.,'count':0} for m in ('base','normal','full')}
    masked={m:{'sum_abs':0.,'count':0} for m in scores}
    h12={m:{'sum_abs':0.,'count':0} for m in scores}
    harm={'count':0.,'total':0.,'active':0.}
    with torch.inference_mode():
        for b in dl:
            b=to_device(b,device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                result=model(base=b['base'],context=b['context'],x=b['x'],mask=b['mask'],
                       normal=b['normal'],cf=b['cf'],score=b['score'],valid=b['valid'],
                       match=b['match'],rank_mode='learned')
            y=b['y_raw'].float();w=b['future_valid'].float()
            # Prediction units converted using single frozen teacher's train normalization.
            mm=float(bc['mean']);ss=float(bc['std'])
            error_base=(b['base'].float()*ss+mm-y)
            error_n=(result['normal_prediction'].float()*ss+mm-y)
            error_full=(result['prediction'].float()*ss+mm-y)
            direct=b['direct_mask'].bool()
            if direct.ndim==2:direct=direct[:,None,:].expand_as(w)
            if direct.shape!=w.shape:raise ValueError('direct_mask must align with future labels')
            direct=direct.float()*w
            for key,err in (('base',error_base),('normal',error_n),('full',error_full)):
                scores[key]['sum_abs']+=float((err.abs()*w).sum())
                scores[key]['sum_sq']+=float((err.square()*w).sum())
                scores[key]['count']+=float(w.sum())
                masked[key]['sum_abs']+=float((err.abs()*direct).sum())
                masked[key]['count']+=float(direct.sum())
                h12[key]['sum_abs']+=float((err[:,-1:].abs()*w[:,-1:]).sum())
                h12[key]['count']+=float(w[:,-1:].sum())
            en=error_n.abs();ef=error_full.abs()
            harm['count']+=float((((ef-en)>1e-6)*w).sum())
            harm['active']+=float(((result['cf_increment'].abs()>1e-6)*w).sum())
            harm['total']+=float(w.sum())
    res={}
    for key,v in scores.items():
        z=max(1,v['count']);res[key]={'mae':v['sum_abs']/z,'rmse':(v['sum_sq']/z)**.5,
                                 'directly_masked_mae':masked[key]['sum_abs']/max(1,masked[key]['count']),
                                 'H12_MAE':h12[key]['sum_abs']/max(1,h12[key]['count'])}
    res.update(dataset=identity['dataset'],target=identity['target'],severity=a.severity,
               seed=identity['seed'],checkpoint=str(Path(a.checkpoint).resolve()),
               cf_gain_raw=res['normal']['mae']-res['full']['mae'],
               cf_harm_rate=harm['count']/max(1,harm['total']),
               cf_activation_rate=harm['active']/max(1,harm['total']),
               improvement_vs_base_pct=100*(res['base']['mae']-res['full']['mae'])/max(1e-10,res['base']['mae']))
    print('V9_TEST '+json.dumps(res),flush=True)
    run=Path(a.run_identity).parent
    path=run/f'test_s{a.severity}.json'
    if path.exists():raise FileExistsError(f'No test overwrite allowed: {path}')
    path.write_text(json.dumps(res,indent=2))

if __name__=='__main__':main()
