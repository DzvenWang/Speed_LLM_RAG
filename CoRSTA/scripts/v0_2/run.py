#!/usr/bin/env python3
"""CoRSTA v0.2.2 orchestration: CF-first + Normal-Anchored modes; no Retrieval imports.

Preflight (CPU), cf/residual/joint/control staged training, lock, locked test, report.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from corsta.observability import install_timestamps
install_timestamps()
import torch,yaml
from corsta_v2.util import atomic_json,atomic_torch,sha256,source_hash,seed_all,checkpoint_load
from corsta_v2.data.dataset import CoRSTASplit,resolve_shared_cache,SEVERITIES
from corsta_v2.stla.backbone import make_stla
from corsta_v2.model import CoRSTAv2
from corsta_v2.mechanisms import mechanism_kwargs
from corsta_v2.spatial.graph import build_road,add_cf_edges
from corsta_v2.training.evaluation import evaluate,validation_partitions
from corsta_v2.training.trainer import STAGES,stages_for,train_stage


def parser():
    p=argparse.ArgumentParser(description='CoRSTA v0.2 CF + Residual, no Retrieval')
    p.add_argument('--stage',choices=('preflight','train','cf','residual','joint','control','normal','anchor_cf','anchor_control','lock','test','report','all'),default='train')
    p.add_argument('--target',choices=('speed','flow'),required=True)
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--mask-proj-checkpoint')
    p.add_argument('--formal-cache')
    p.add_argument('--run-dir')
    p.add_argument('--config',default=str(ROOT/'configs/v0_2/corsta_v022.yaml'))
    p.add_argument('--cf-graph-path')
    p.add_argument('--prediction-mode',choices=('cf_first','normal_anchored'))
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--limit-train',type=int)
    p.add_argument('--limit-val',type=int)
    p.add_argument('--limit-test',type=int)
    p.add_argument('--batch-initial',type=int,choices=(16,32,64,128))
    p.add_argument('--workers',type=int)
    return p


def compose(a):
    cfg=yaml.safe_load(Path(a.config).read_text())
    if cfg['version'] not in ('0.2.1','0.2.2','0.2.3'):
        raise RuntimeError('Expected CoRSTA v0.2.1/v0.2.2/v0.2.3 config')
    if a.prediction_mode is not None:cfg['model']['prediction_mode']=a.prediction_mode
    cfg['model'].setdefault('prediction_mode','cf_first')
    stages=stages_for(cfg)
    allowed_cfg_stages=set(cfg['training']['stage_epochs'])
    if not set(stages)<=allowed_cfg_stages:
        raise RuntimeError(f'Missing stage_epochs for {set(stages)-allowed_cfg_stages}')
    if a.stage in ('cf','residual','joint','control','normal','anchor_cf','anchor_control') and a.stage not in stages:
        raise RuntimeError(f'--stage={a.stage} is not part of prediction_mode={cfg["model"]["prediction_mode"]}')
    cfg['experiment'].update(target=a.target,dataset=a.dataset,seed=a.seed)
    if a.smoke:
        cfg['training']['stage_epochs']={k:1 for k in stages}
        cfg['training'].update(patience=1,worker_count=0,eval_batch_size=4,
                               batch_initial=16,effective_batch=16,selection_limit=4)
    if a.batch_initial is not None:cfg['training']['batch_initial']=a.batch_initial
    if a.workers is not None:cfg['training']['worker_count']=a.workers
    if a.limit_train is not None or a.limit_val is not None or a.limit_test is not None:
        if not a.smoke:raise RuntimeError('Data limits are supported only for --smoke experiments')
    if a.smoke:
        if a.limit_train is None:a.limit_train=32
        if a.limit_val is None:a.limit_val=48
        if a.limit_test is None:a.limit_test=8
    root=Path(os.environ.get('SPEED_LLM_RAG_ROOT',ROOT.parent)).resolve()
    mode=os.environ.get('BACKBONE_MODE','incomplete')
    if mode not in ('incomplete','complete'):raise ValueError('BACKBONE_MODE')
    checkpoint=Path(a.backbone_checkpoint).expanduser().resolve()
    mask=(Path(a.mask_proj_checkpoint).expanduser().resolve() if a.mask_proj_checkpoint else
          root/'CoRSTA_exps'/a.target/mode/a.dataset/'formal'/f'seed{a.seed}'/'bootstrap_v7'/'s0.pt')
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if not mask.is_file():raise FileNotFoundError(f'Missing S0 checkpoint {mask}. Pass --mask-proj-checkpoint.')
    run=(Path(a.run_dir).resolve() if a.run_dir else root/f'CoRSTA_v{cfg["version"].replace(".", "")}_exps'/a.target/mode/a.dataset/
         ('smoke' if a.smoke else 'formal')/f'seed{a.seed}')
    cache=(Path(a.formal_cache).resolve() if a.formal_cache else
           resolve_shared_cache(root,a.target,a.dataset,'formal'))
    cf_graph=(Path(a.cf_graph_path).resolve() if a.cf_graph_path else None)
    if cfg['model']['use_cf_graph'] and (cf_graph is None or not cf_graph.is_file()):
        raise RuntimeError('use_cf_graph requires --cf-graph-path from TRAIN-only builder')
    return cfg,root,run,checkpoint,mask,cache,cf_graph


def build_model(cfg,root,checkpoint,mask,cf_graph,device):
    stla,bc=make_stla(root,checkpoint,mask,device,int(cfg['model']['stla_tail_layers']))
    nodes=int(stla.base.num_nodes)
    physical=build_road(root,cfg['experiment']['target'],cfg['experiment']['dataset'],
                        k_per_direction=int(cfg['model']['road_neighbors_per_direction']))
    if physical['ids'].shape[0]!=nodes:
        raise RuntimeError('Physical graph nodes do not match STLA')
    graph=dict(physical)
    if not cfg['model']['use_road_graph']:
        import numpy as np
        graph['valid']=np.zeros_like(physical['valid'],dtype=bool)
        graph['edge_features']=np.zeros_like(physical['edge_features'])
    if cf_graph is not None and cfg['model']['use_cf_graph']:
        graph=add_cf_edges(graph,cf_graph,max_cf_neighbors=int(cfg['model']['max_cf_neighbors']))
    model=CoRSTAv2(stla,graph,hidden=int(stla.model_dim),
                   failure_dim=int(cfg['model']['failure_dim']),cf_dim=int(cfg['model']['cf_dim']),
                   residual_dim=int(cfg['model']['residual_dim']),
                   horizon=int(cfg['experiment']['horizon']),heads=int(cfg['model']['attention_heads']),
                   latent_scale=float(cfg['model']['latent_correction_scale']),
                   cf_error_gate=bool(cfg['model'].get('use_cf_error_gate',True)),
                   prediction_mode=cfg['model'].get('prediction_mode','cf_first'),
                   **mechanism_kwargs(cfg)).to(device)
    return model,bc,physical


def run_identity(cfg,checkpoint,mask,cache,root,graph_path,smoke):
    road_source=Path(root)/'data/processed/traffident/base_supportbench'/cfg['experiment']['target']/cfg['experiment']['dataset']
    graph_files=[road_source/'core/nodes.csv',road_source/'graph/adjacency.npy',road_source/'graph/distance.npy']
    graph_digests={p.name:sha256(p) for p in graph_files if p.is_file()}
    return {'version':cfg['version'],'code_sha256':source_hash(ROOT),'backbone_sha256':sha256(checkpoint),
            'mask_proj_sha256':sha256(mask),'cache_path':str(cache),'smoke':bool(smoke),
            'cf_graph_sha256':sha256(graph_path) if graph_path else None,'road_graph_sha256':graph_digests,
            'split_manifest_sha256':{s:sha256(Path(cache)/f'{s}_manifest.csv') for s in ('train','val')},
            'configuration':cfg,'root_project':str(root)}


def check_identity(run,data):
    run.mkdir(parents=True,exist_ok=True)
    path=run/'run_identity.json'
    if path.exists():
        if json.loads(path.read_text())!=data:raise RuntimeError('v0.2 identity mismatch; use NEW run-dir')
    else:
        if any(run.iterdir()):raise RuntimeError('Nonempty v0.2 run directory without identity')
        atomic_json(data,path)
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def validation(cfg,cache,graph,seed,limit=None):
    if not cfg['training']['validation_scenarios_from_val_raw']:
        raise RuntimeError('Four-severity validation requires val raw-view mask scenario construction')
    datasets={str(s):CoRSTASplit(cache,'val',seed=seed,severity=s,limit=limit,
                         neighbors=graph['adjacency_list'],directions=graph['direction_labels']) for s in SEVERITIES}
    idx_sel,idx_lock=validation_partitions(datasets['20'],int(cfg['training']['selection_limit']))
    return datasets,idx_sel,idx_lock


def _load_final(model,run,identity,use_graph):
    lock_path=run/'model_lock.json'
    if not lock_path.is_file():raise RuntimeError('TEST requires model_lock.json')
    lock=json.loads(lock_path.read_text())
    final_stage='anchor_control' if model.prediction_mode=='normal_anchored' else 'control'
    best=run/f'{final_stage}/best.pt'
    if lock['identity']!=identity or lock['sha256']!=sha256(best):
        raise RuntimeError('Model changed since lock')
    ck=checkpoint_load(best)
    if ck['identity']!=identity:raise RuntimeError('Final checkpoint identity mismatch')
    model.load_state_dict(ck['model'],strict=True)
    model.set_stage(final_stage,use_graph=use_graph)
    model.eval()
    return lock


def main(a=None):
    a=parser().parse_args() if a is None else a
    cfg,root,run,ckpt,mask,cache,cf_graph=compose(a)
    device=torch.device('cpu' if a.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type!='cuda' and a.stage not in ('preflight',) and not a.smoke:
        raise RuntimeError('Formal v0.2 training/evaluation requires CUDA, use CPU for preflight or --smoke')
    seed_all(a.seed)
    model,bc,graph=build_model(cfg,root,ckpt,mask,cf_graph,device)
    n=model.stla.base.num_nodes
    print(f'CORSTA_V2_START stage={a.stage} target={a.target} dataset={a.dataset} nodes={n} device={device}',flush=True)
    # CPU preflight is read-only, no experiment identity side effects.
    if a.stage=='preflight':
        B,L,N=2,12,n
        x=torch.randn(B,L,N,device=device);mask_tensor=(torch.rand(B,L,N,device=device)>.4).float()
        x=torch.where(mask_tensor.bool(),x,torch.zeros_like(x))
        tod=torch.randint(0,288,(B,L),device=device);dow=torch.randint(0,7,(B,L),device=device)
        model.eval()
        with torch.no_grad():
            out=model({'x':x,'mask':mask_tensor,'tod':tod,'dow':dow},enable_controller=False,enable_graph=False)
            base=model.stla(x,mask_tensor,tod,dow)[0]
            delta=(base-out['backbone']).abs().max().item()
            assert out['prediction'].shape==(B,12,N)
            if not torch.isfinite(out['prediction']).all():raise RuntimeError('Nonfinite preflight prediction')
            if delta>1e-5:raise RuntimeError(f'Protected STLA not equivalent: {delta}')
            off=model({'x':x,'mask':mask_tensor,'tod':tod,'dow':dow},enable_cf=False,enable_residual=False)
            if model.prediction_mode=='normal_anchored':
                if not torch.allclose(off['prediction'],off['normal_prediction'],atol=1e-5):
                    raise RuntimeError('Normal-anchored no-CF fallback broken')
            elif (off['prediction']-base).abs().max().item()>1e-5:
                raise RuntimeError('Backbone fallback broken')
        # Validate input cache shapes and availability without allocating train full arrays.
        sample=CoRSTASplit(cache,'train',seed=a.seed,limit=1)[0]
        if cfg['training']['validation_scenarios_from_val_raw']:
            v=CoRSTASplit(cache,'val',seed=a.seed,severity=20,limit=1,
                          neighbors=graph['adjacency_list'],directions=graph['direction_labels'])[0]
            if v['x'].shape!=(12,n):raise RuntimeError('Validation source/Backbone nodes mismatch')
        if sample['x'].shape!=(12,n):raise RuntimeError('Shared cache / checkpoint node mismatch')
        print(f'CORSTA_V2_PREFLIGHT_PASS stla_diff={delta:.3g} nodes={N} raw_train_valid={float(sample["original_valid"].mean()):.4f}',flush=True)
        return
    identity_data=run_identity(cfg,ckpt,mask,cache,root,cf_graph,a.smoke)
    identity=check_identity(run,identity_data)
    mean=float(bc['mean']);std=float(bc['std'])
    if std<=0:raise RuntimeError('Backbone invalid target std')
    def log(s):print(s,flush=True)
    train_limit=a.limit_train if a.smoke else None
    val_limit=a.limit_val if a.smoke else None
    stages=stages_for(cfg)
    final_stage=stages[-1]
    if a.stage in ('train','all','lock') or a.stage in stages:
        val_ds,selection,locked=validation(cfg,cache,graph,a.seed,limit=val_limit)
    if a.stage in ('train','all') or a.stage in stages:
        train_ds=CoRSTASplit(cache,'train',seed=a.seed,limit=train_limit,
                             augment=bool(cfg['training']['augment_training_masks']),
                             contrast_views=bool(cfg['training'].get('multimask_error_contrast',False)),
                             num_views=int(cfg['training'].get('multimask_views',2)),
                             failure_kinds=cfg['training'].get('failure_kinds'),
                             legacy_multimask=(cfg['model'].get('prediction_mode')=='cf_first'),
                             neighbors=graph['adjacency_list'],directions=graph['direction_labels'])
        end=stages.index(a.stage) if a.stage in stages else len(stages)-1
        for st in stages[:end+1]:
            train_stage(model,train_ds,val_ds,selection,device,cfg,mean,std,run,identity,st,log)
    if a.stage in ('lock','all'):
        if not (run/f'{final_stage}/COMPLETE.json').exists():raise RuntimeError('Cannot lock before final stage complete')
        path=run/'model_lock.json'
        if path.exists():
            lock=json.loads(path.read_text());assert lock['identity']==identity
            log('CORSTA_V2_LOCK_REUSE')
        else:
            model.load_state_dict(checkpoint_load(run/f'{final_stage}/best.pt')['model'],strict=True)
            model.set_stage(final_stage,use_graph=(bool(cfg['model']['use_road_graph']) or bool(cfg['model']['use_cf_graph'])))
            metrics=evaluate(model,val_ds,device,mean,std,int(cfg['training']['eval_batch_size']),locked,mode=final_stage)
            atomic_json({'identity':identity,'sha256':sha256(run/f'{final_stage}/best.pt'),
                         'checkpoint':f'{final_stage}/best.pt','locked_validation':metrics},path)
            log('CORSTA_V2_LOCK_PASS '+json.dumps(metrics['macro_mae']))
    if a.stage in ('test','all'):
        _load_final(model,run,identity,(bool(cfg['model']['use_road_graph']) or bool(cfg['model']['use_cf_graph'])))
        results={}
        for sev in SEVERITIES:
            test_path=run/'test'/f's{sev}.json'
            if test_path.exists():
                result=json.loads(test_path.read_text())
                if result['identity']!=identity:raise RuntimeError('Test result identity mismatch')
            else:
                eval_cache=resolve_shared_cache(root,a.target,a.dataset,'evaluation',sev)
                test_ds=CoRSTASplit(eval_cache,'test',seed=a.seed,limit=a.limit_test if a.smoke else None)
                metric=evaluate(model,{str(sev):test_ds},device,mean,std,
                                int(cfg['training']['eval_batch_size']),mode=final_stage)
                result={'identity':identity,'severity':sev,'metric':metric}
                atomic_json(result,test_path)
            results[str(sev)]=result['metric']['per_severity'][str(sev)]
            log('CORSTA_V2_TEST_PASS severity=s'+str(sev)+' '+json.dumps(results[str(sev)]['final']))
    if a.stage in ('report','all'):
        _load_final(model,run,identity,(bool(cfg['model']['use_road_graph']) or bool(cfg['model']['use_cf_graph'])))
        results={}
        for sev in SEVERITIES:
            file=run/'test'/f's{sev}.json'
            if not file.exists():raise RuntimeError('Missing test: '+str(sev))
            results[str(sev)]=json.loads(file.read_text())['metric']['per_severity'][str(sev)]
        keys=(('backbone','normal','normal_cf_candidate','final') if model.prediction_mode=='normal_anchored'
              else ('backbone','cf','cf_residual','final'))
        summary={k:{m:sum(results[str(s)][k][m] for s in SEVERITIES)/len(SEVERITIES)
                     for m in ('mae','rmse')} for k in keys}
        atomic_json({'identity':identity,'macro_severities':summary,'per_severity':results},run/'summary.json')
        lines=[f'# CoRSTA {cfg["version"]} {cfg["model"]["prediction_mode"]}, locked TEST','','| Model | Macro-MAE | Macro-RMSE |','|---|---:|---:|']
        for k,v in summary.items():lines.append(f'| {k} | {v["mae"]:.4f} | {v["rmse"]:.4f} |')
        (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        log('CORSTA_V2_REPORT_PASS '+json.dumps(summary))

if __name__=='__main__':main()
