#!/usr/bin/env python3
"""TCRR-STLA v9.1: same-cohort, Validation-Selection-only mechanism controls.

The wrong-mask CF intervention recomputes *historical* replay on an alternative
query mask; no TEST outcomes are used. It is a synthetic negative control, not
a claim of physically realistic alternate failure.
"""
from __future__ import annotations
import os,sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from torch.utils.data import DataLoader,Subset
from tcrr_stla.config import load_config,file_sha256
from tcrr_stla.v9.prepare import V9EvidenceDataset
from tcrr_stla.v9.graph import GraphBundle
from tcrr_stla.v9.model import TCRRV9,utility_targets,V9_VERSION
from tcrr_stla.v9.training import to_device
from tcrr_stla.v5.calibration import chronological_partitions
from tcrr_stla.v5.shared_adapter import initial_model
from tcrr_stla.evidence import FutureMemory,counterfactual_residuals

@torch.inference_mode()
def run(ck,identity,max_queries=256,batch_size=8,wrong_mask_max=32):
    os.environ['TRAFFIDENT_TARGET']=identity['target']
    cfg=load_config(ROOT/'configs/tcrr_stla_v5.yaml')
    cfg['experiment'].update(target=identity['target'],dataset=identity['dataset'],seed=identity['seed'])
    cache=Path(identity['formal_cache']);root=Path(identity['evidence_root'])
    ds=V9EvidenceDataset(cache,'val',views=(0,),root=root)
    if bool(identity['smoke']):
        indices=list(range(len(ds)))
    else:
        groups=chronological_partitions(ds.ds.window_starts[:len(ds.ds)],12,12)
        indices=groups['selection']
    if max_queries<1 or batch_size<1:raise ValueError('Invalid diagnostic budget')
    indices=indices[:max_queries]
    # Every mode including wrong-mask MUST run on exactly the same Query set.
    if wrong_mask_max>0:indices=indices[:wrong_mask_max]
    if not indices:raise ValueError('No validation selection rows')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sd=ck['model'];g=GraphBundle(sd['graph.functional'],sd['graph.upstream'],sd['graph.downstream'])
    model=TCRRV9(g,ck['context_dim'],hidden=ck['cfg']['hidden'],
                 bound=ck['cfg']['bound'],blocks=ck['cfg']['blocks']).to(device)
    model.load_state_dict(sd,strict=True);model.eval()
    variants=['base','normal','correct_cf','uniform_cf','shuffled_cf',
              'no_cf_graph','no_all_graph','no_recoverability_cf']
    if wrong_mask_max>0:variants.append('wrong_mask_cf')
    metrics={v:dict(abs_sum=0.,valid=0.,harm=0.,gain=0.) for v in variants}
    teacher=None;memory=None
    if wrong_mask_max>0:
        original=Path(identity['backbone_checkpoint']);adapter=Path(identity['adapter_checkpoint'])
        if file_sha256(original)!=identity['backbone_sha256'] or file_sha256(adapter)!=identity['adapter_sha256']:
            raise RuntimeError('Teacher checkpoint mismatch')
        teacher,_=initial_model(cfg,original,device,adapter)
        teacher.eval();[p.requires_grad_(False) for p in teacher.parameters()]
        memory=FutureMemory(cache)
    count=0
    for raw in DataLoader(Subset(ds,indices),batch_size=batch_size,num_workers=0):
        b=to_device(raw,device)
        q=dict(base=b['base'],context=b['context'],x=b['x'],mask=b['mask'],
               normal=b['normal'],cf=b['cf'],score=b['score'],valid=b['valid'],match=b['match'])
        correct=model(**q)
        results=dict(base=b['base'],normal=correct['normal_prediction'],
                     correct_cf=correct['prediction'],
                     uniform_cf=model(**q,rank_mode='uniform')['prediction'],
                     shuffled_cf=model(**{**q,'cf':b['cf'].roll(1,dims=1)})['prediction'],
                     no_cf_graph=model(**q,cf_graph_enabled=False)['prediction'],
                     no_all_graph=model(**q,graph_enabled=False)['prediction'],
                     no_recoverability_cf=model(**q,recoverability_enabled=False)['prediction'])
        if teacher is not None:
            cand=memory.fetch(b['candidate_ids'].long(),device)
            wrong_mask=torch.roll(b['mask'],shifts=max(1,b['mask'].shape[-1]//4),dims=-1)
            wrong_res=counterfactual_residuals(teacher,cand,wrong_mask,chunk_size=48,
                                               strict_source_mask=True)
            wrong_cf=(wrong_res-b['normal'])*b['valid'][:,:,None,None]
            results['wrong_mask_cf']=model(**{**q,'cf':wrong_cf})['prediction']
        y=b['y_norm'].float(); w=b['future_valid'].float()
        anchor=(correct['normal_prediction']-y).abs()
        for variant,pred in results.items():
            err=(pred.float()-y).abs()
            m=metrics[variant]
            m['abs_sum']+=float((err*w).sum());m['valid']+=float(w.sum())
            m['harm']+=float(((err-anchor)>1e-6).float().mul(w).sum())
            m['gain']+=float(((anchor-err)*w).sum())
        count+=len(y)
    results={v:dict(MAE_norm=x['abs_sum']/max(1,x['valid']),
                    Gain_vs_Normal_norm=x['gain']/max(1,x['valid']),
                    Harm_vs_Normal_rate=x['harm']/max(1,x['valid']))
             for v,x in metrics.items()}
    assert len(set(x['valid'] for x in metrics.values()))==1,'Variant cohort mismatch'
    return dict(version=V9_VERSION,dataset=identity['dataset'],target=identity['target'],
                samples=count,stage=ck['stage'],results=results,
                cohort='identical validation rows for all variants',
                no_cf_graph='Historical CF graph messages disabled; Query graph retained',
                no_all_graph='Both Query and CF graph context disabled; Normal anchor also changes',
                wrong_mask_kind='node_roll_same_missing_count (synthetic intervention)')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--run-identity',required=True)
    p.add_argument('--max-queries',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--wrong-mask-max',type=int,default=32,
                   help='When positive all modes use at most this many identical rows')
    a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    identity=json.loads(Path(a.run_identity).read_text())
    if ck.get('version')!=V9_VERSION or identity.get('version')!=V9_VERSION:
        raise ValueError('v9.1 checkpoint/identity required')
    out=run(ck,identity,a.max_queries,a.batch_size,a.wrong_mask_max)
    dest=Path(a.run_identity).parent/f'diagnostics_{ck["stage"]}.json'
    dest.write_text(json.dumps(out,indent=2))
    print('V9_DIAG '+json.dumps(out),flush=True)
if __name__=='__main__':main()
