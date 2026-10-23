#!/usr/bin/env python3
"""Validation-only FARE v6.0.1 innovation audit of existing v6.0.0 weights.

Never trains, overwrites checkpoints, creates OOF, or evaluates Test. Supply
only trusted FARE checkpoint files because torch.load deserializes objects.
"""
import _bootstrap
import argparse
import json
from pathlib import Path
import torch

from fare.config import read_config,project_path,dump_json,hash_file
from fare.pipeline import prepare
from fare.innovation_audit import audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='FARE/configs/formal.yaml')
    parser.add_argument('--target',required=True,choices=('speed','flow'))
    parser.add_argument('--dataset',default='TraffiDent_Net_06')
    parser.add_argument('--seed',type=int,default=10)
    parser.add_argument('--checkpoint',required=True,help='Existing FARE v6 best.pt; NEVER an STLAformer checkpoint')
    parser.add_argument('--backbone-checkpoint',help='Frozen STLAformer checkpoint; defaults to saved v6 provenance')
    parser.add_argument('--memory',help='Original OOF Memory folder; defaults to saved v6 provenance')
    parser.add_argument('--output',help='NEW audit output directory (default: sibling innovation_audit_v601)')
    parser.add_argument('--validation-queries',type=int)
    parser.add_argument('--candidate-queries',type=int)
    parser.add_argument('--candidate-nodes',type=int)
    parser.add_argument('--candidate-slots',type=int)
    parser.add_argument('--bootstrap-samples',type=int)
    parser.add_argument('--batch-size',type=int)
    parser.add_argument('--workers',type=int)
    parser.add_argument('--smoke',action='store_true',help='Run 4 queries and 2 candidate queries first')
    args=parser.parse_args()
    ckpt=project_path(args.checkpoint,must_exist=True)
    if not ckpt.is_file():raise FileNotFoundError(ckpt)
    saved=torch.load(ckpt,map_location='cpu',weights_only=False)
    if saved.get('version')!='6.0.0':
        raise ValueError('Only FARE v6.0.0 model checkpoints are supported')
    meta=saved.get('metadata',{})
    if meta.get('dataset')!=args.dataset or meta.get('target')!=args.target or int(meta.get('seed',-1))!=args.seed:
        raise ValueError('Checkpoint dataset/target/seed does not match requested audit')
    backbone=args.backbone_checkpoint or meta.get('backbone_checkpoint')
    memory=args.memory or meta.get('memory_path')
    if not backbone or not memory:raise ValueError('Missing saved backbone/memory provenance: pass explicit paths')
    config=read_config(args.config,args.dataset,args.target,args.seed)
    ac=config.setdefault('innovation_audit',{})
    for arg,key in (('validation_queries','validation_queries'),('candidate_queries','candidate_queries'),
                    ('candidate_nodes','candidate_nodes'),('candidate_slots','candidate_slots'),
                    ('bootstrap_samples','bootstrap_samples')):
        v=getattr(args,arg)
        if v is not None:
            if v<=0:raise ValueError(f'{arg} must be positive')
            ac[key]=int(v)
    for arg,key in (('batch_size','batch_size'),('workers','workers')):
        v=getattr(args,arg)
        if v is not None:
            if v<0 or (arg=='batch_size' and v==0):raise ValueError(f'Invalid {arg}')
            config['training'][key]=v
    if args.smoke:
        ac.update({'validation_queries':4,'candidate_queries':2,'candidate_nodes':4,
                   'candidate_slots':2,'bootstrap_samples':20})
        config['training'].update({'batch_size':2,'workers':0})
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('FARE6_AUDIT_START '+json.dumps({'checkpoint':str(ckpt),'target':args.target,
          'validation_queries':ac.get('validation_queries'),'candidate_queries':ac.get('candidate_queries'),
          'device':str(device),'train':False}),flush=True)
    model,bank,_,val,_,prepared_meta=prepare(config,device,
                        backbone_file=backbone,memory_file=memory,build=False,cuda_safety=False)
    for key in ('backbone_sha256','memory_sha256'):
        if meta.get(key)!=prepared_meta.get(key):
            raise RuntimeError(f'Checkpoint/OOF provenance mismatch for {key}')
    model.load_state_dict(saved['state_dict'],strict=True)
    model.eval()
    if config['training'].get('cache_gpu_memory',False):
        print('FARE6_AUDIT_GPU_CACHE '+json.dumps(bank.enable_gpu_cache(device,
              config['training'].get('gpu_cache_budget_gb',5.0))),flush=True)
    output=project_path(args.output) if args.output else ckpt.parent/'innovation_audit_v601'
    output.mkdir(parents=True,exist_ok=True)
    # Refuse overwrite of an earlier finished audit or original run outputs.
    if (output/'innovation_verdict.json').exists():
        raise FileExistsError(f'Audit report already exists, choose a new --output: {output}')
    dump_json(output/'audit_source.json',{'checkpoint':str(ckpt),'checkpoint_sha256':hash_file(ckpt),
       'checkpoint_version':saved['version'],'backbone_checkpoint':str(backbone),
       'memory':str(memory),'model_weights_loaded_strictly':True,
       'trained_parameters_updated':False,'config':str(args.config),
       'validation_queries':ac.get('validation_queries'), 'candidate_queries':ac.get('candidate_queries')})
    audit(model,bank,val,device,config,output)
    print('FARE6_STANDALONE_AUDIT_PASS '+str(output),flush=True)

if __name__=='__main__':main()
