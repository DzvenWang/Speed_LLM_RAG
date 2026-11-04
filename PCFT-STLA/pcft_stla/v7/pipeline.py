"""v7 reuses proven v6 SupportBench/graph and original official STLA checkpoint loader."""
from __future__ import annotations
import datetime,json,os
from pathlib import Path
import torch
from pcft_stla.v6.config import load_config,root,json_write
from pcft_stla.v6.pipeline import prepare
from pcft_stla.v6.model import load_original_stla
from pcft_stla.v6.training import seed_all,to_device
from .model import PCFTV7
from .training import train_pipeline,trainable_for,prediction_loss,autocast_for


def instantiate(cfg,repo,checkpoint,graph,identity,device):
    base,ck=load_original_stla(repo,checkpoint,device=device)
    if int(ck['num_nodes'])!=graph.num_nodes:raise RuntimeError('STLA checkpoint node count vs graph mismatch')
    target_idx=identity['channels'].index(identity['target'])
    m=cfg['model'];model=PCFTV7(base,graph,target_idx,channels=len(identity['channels']),
        hidden=int(m['hidden_dim']),horizon=int(cfg['experiment']['horizon']),
        controller_dim=int(m['controller_dim']),enable_functional=bool(m['use_functional_context']))
    return model.to(device)


def preflight(model,datasets,cfg,device,gpu_backprop=False):
    from torch.utils.data import DataLoader
    batch=to_device(next(iter(DataLoader(datasets['train'],batch_size=min(2,len(datasets['train']))))),device)
    val=to_device(next(iter(DataLoader(datasets['val'],batch_size=1))),device)
    model.anchor.set_trainable_tail(0)
    diff=model.anchor.parity(batch,atol=5e-4 if device.type=='cuda' else 1e-5)
    print('V7_ANCHOR_PARITY_PASS',diff,flush=True)
    model.eval()
    with torch.no_grad():
        for stage in model.STAGES:
            model.set_stage(stage)
            o=model(batch)
            if o['prediction'].shape!=batch['y_norm'].shape or not torch.isfinite(o['prediction']).all():
                raise RuntimeError(f'V7 bad output {stage}')
            pred=model(val)
            if pred['prediction'].shape!=val['y_norm'].shape:raise RuntimeError('VAL shape mismatch')
        model.set_stage('joint')
        model.set_ablation(use_propagation=False,use_controller=False)
        x=model(batch)
        assert torch.allclose(x['prop_delta'],torch.zeros_like(x['prop_delta']))
        model.set_ablation(use_propagation=True,use_controller=True)
        print('V7_CPU_OR_DEVICE_FORWARD_PASS',device,flush=True)
    if gpu_backprop:
        for stage in model.STAGES:
            names=trainable_for(model,stage,cfg)
            model.train(True);model.zero_grad(set_to_none=True)
            with autocast_for(device,cfg):
                out=model(batch);loss=prediction_loss(out,batch,cfg,stage)
            if not torch.isfinite(loss):raise RuntimeError(f'GPU loss nonfinite {stage}')
            loss.backward()
            grads={name for name,p in model.named_parameters() if p.requires_grad and p.grad is not None}
            if not grads:raise RuntimeError(f'GPU missing gradients stage {stage}')
            if stage=='propagation' and not any('prop_head' in n for n in grads):
                raise RuntimeError('C2 correction head has no gradient')
            print('V7_GPU_STAGE_PASS',stage,'loss',float(loss.detach()),'grad_tensors',len(grads),flush=True)
        model.zero_grad(set_to_none=True)
    model.set_stage('joint')
    model.anchor.set_trainable_tail(0)
    return True


def run(cfg,checkpoint=None,run_name=None,smoke=False,resume=None,preflight_only=False,gpu_smoke_only=False):
    seed_all(int(cfg['experiment']['seed']))
    if not smoke and not preflight_only and not torch.cuda.is_available():
        raise RuntimeError('Formal v7 requires GPU; use --smoke or --preflight-only for CPU')
    # Always reuse the complete v6 cache identity, never a short smoke cache under the same path.
    # Same tested Base_SupportBench cache and audited graphs; do not touch v6 experiment weights.
    repo,_,ck,graph,datasets,identity=prepare(cfg,checkpoint,smoke=False)
    identity=dict(identity,v7_schema='pcft_v7_anchor_selective_v1',model_config=cfg['model'])
    if cfg['experiment']['train_mask_mode']=='mixed_train_masks':
        from pcft_stla.v7.data import MixedTrainMasks
        from pcft_stla.v6.data import open_bench
        bench=open_bench(repo,cfg['experiment']['dataset'],target=cfg['experiment']['target'])
        datasets['train']=MixedTrainMasks(datasets['train'],bench,
            severities=cfg['experiment'].get('train_severities',[20,40,60,90]),
            families=cfg['experiment'].get('train_families'),seed=cfg['experiment']['seed'])
        identity['train_mask_mode']='mixed_train_masks'
    else:identity['train_mask_mode']='fixed_s40'
    device=torch.device('cuda' if torch.cuda.is_available() and not preflight_only else 'cpu')
    model=instantiate(cfg,repo,ck,graph,identity,device)
    print('V7_DATA_READY',json.dumps({'dataset':cfg['experiment']['dataset'],'target':identity['target'],
        'N':graph.num_nodes,'physical_edges':int(graph.edge_index.shape[1]),
        'functional_edges':int(graph.functional_index.shape[1]),'rows':{k:len(v) for k,v in datasets.items()},
        'device':str(device)}),flush=True)
    preflight(model,datasets,cfg,device,gpu_backprop=device.type=='cuda' and not preflight_only)
    if preflight_only or gpu_smoke_only:
        print('V7_PREFLIGHT_PASS',device,flush=True)
        return
    # GPU preflight executed backwards; reinitialize to preserve exact reproducibility.
    seed_all(int(cfg['experiment']['seed']))
    del model
    if device.type=='cuda':torch.cuda.empty_cache()
    model=instantiate(cfg,repo,ck,graph,identity,device)
    run_dir=(Path(resume).resolve().parent if resume else
        repo/'PCFT-STLA_exps'/identity['target']/identity['mode']/identity['dataset']/'v7'/
        ('smoke' if smoke else 'formal')/(run_name or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+f'_seed{identity["seed"]}'))
    run_dir.mkdir(parents=True,exist_ok=True)
    json_write(run_dir/'identity.json',identity);json_write(run_dir/'config.json',cfg)
    return train_pipeline(model,datasets,cfg,identity,run_dir,device,resume=resume,smoke=smoke)
