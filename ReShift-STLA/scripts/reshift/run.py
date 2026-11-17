#!/usr/bin/env python3
"""ReShift orchestration (independent of every CoRSTA / Mask-line output).

Stages
  preflight  CPU shape/identity check (no artefacts)
  s0         ReShift's own STLA mask projection                     [shared]
  bank       P_full cache + Train-slot and Val replay banks          [shared]
  m0         M0 diagnostics (S0 + Val banks only)
  normal, anchor_cf                                                  [shared]
  m1         M1 diagnostics (needs anchor_cf)
  anchor_ret, anchor_control                                         [this run dir]
  train      s0 -> bank -> all model stages
  lock, test, audit, report, all (= train -> lock -> test -> audit -> report)
"""
from __future__ import annotations
import argparse,copy,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from corsta.observability import install_timestamps
install_timestamps()
import numpy as np
import torch,yaml
from reshift.util import atomic_json,sha256,source_hash,seed_all,checkpoint_load
from reshift.data.dataset import CoRSTASplit,resolve_shared_cache,SEVERITIES
from reshift.stla.backbone import make_stla
from reshift.stla.s0 import train_s0
from reshift.model import CoRSTAv2
from reshift.mechanisms import mechanism_kwargs
from reshift.spatial.graph import build_road,add_cf_edges
from reshift.training.evaluation import evaluate,validation_partitions
from reshift.training.trainer import (stages_for,train_stage,load_model_state,
                                      prepare_train_banks,prepare_eval_banks,query_desc)
from reshift.training.layout import Layout,stage_keys,retrieval_cfg
from reshift.retrieval.memory import TrainMemory
from reshift.retrieval.provider import EvidenceProvider,AUDIT_MODES
from reshift.retrieval.head import RET_ARMS
from reshift import diagnostics

STAGE_CHOICES=('preflight','s0','bank','m0','normal','anchor_cf','m1','anchor_ret','anchor_control',
               'train','lock','test','audit','report','all')


def parser():
    p=argparse.ArgumentParser(description='ReShift: failure-replayed retrieval for missing-sensor forecasting')
    p.add_argument('--stage',choices=STAGE_CHOICES,default='train')
    p.add_argument('--target',choices=('speed','flow'),required=True)
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--formal-cache')
    p.add_argument('--run-dir')
    p.add_argument('--shared-root',help='shared s0/normal/anchor_cf/bank root (default: <seed dir>/shared)')
    p.add_argument('--config',default=str(ROOT/'configs/reshift/reshift_full.yaml'))
    p.add_argument('--cf-graph-path')
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
    if not str(cfg.get('version','')).startswith('reshift-0.1'):raise RuntimeError('Expected a ReShift 0.1.x config')
    if cfg['model'].get('prediction_mode')!='normal_anchored':raise RuntimeError('ReShift requires normal_anchored')
    cfg['experiment'].update(target=a.target,dataset=a.dataset,seed=a.seed)
    stages=stages_for(cfg)
    missing=set(stages)-set(cfg['training']['stage_epochs'])
    if missing:raise RuntimeError(f'Missing stage_epochs for {missing}')
    if a.stage in ('normal','anchor_cf','anchor_ret','anchor_control') and a.stage not in stages:
        raise RuntimeError(f'--stage={a.stage} is not part of this configuration ({stages})')
    if a.smoke:
        cfg['training']['stage_epochs']={k:1 for k in cfg['training']['stage_epochs']}
        cfg['training'].update(patience=1,worker_count=0,eval_batch_size=4,batch_initial=16,
                               effective_batch=16,selection_limit=4)
        cfg.setdefault('s0',{}).update(epochs=1,batch_size=8,eval_batch_size=8,workers=0)
        rb=cfg.setdefault('retrieval_bank',{})
        rb.update(top_k=min(int(rb.get('top_k',9)),3),train_embargo_steps=0,mask_slots=1,query_batch=4,
                  workers=0,memory_on_gpu=False,bank_on_gpu=False,pfull_batch=8,memory_chunk=64,
                  min_valid_features=2)
        if a.limit_train is None:a.limit_train=32
        if a.limit_val is None:a.limit_val=48
        if a.limit_test is None:a.limit_test=8
    elif any(v is not None for v in (a.limit_train,a.limit_val,a.limit_test)):
        raise RuntimeError('Data limits are supported only for --smoke experiments')
    if a.batch_initial is not None:cfg['training']['batch_initial']=a.batch_initial
    if a.workers is not None:cfg['training']['worker_count']=a.workers
    root=Path(os.environ.get('SPEED_LLM_RAG_ROOT',ROOT.parent)).resolve()
    mode=os.environ.get('BACKBONE_MODE','incomplete')
    if mode not in ('incomplete','complete'):raise ValueError('BACKBONE_MODE')
    checkpoint=Path(a.backbone_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    seed_dir=root/'ReShift_exps'/a.target/mode/a.dataset/('smoke' if a.smoke else 'formal')/f'seed{a.seed}'
    run=Path(a.run_dir).resolve() if a.run_dir else seed_dir/Path(a.config).stem
    shared=Path(a.shared_root).resolve() if a.shared_root else seed_dir/'shared'
    cache=(Path(a.formal_cache).resolve() if a.formal_cache else resolve_shared_cache(root,a.target,a.dataset,'formal'))
    cf_graph=Path(a.cf_graph_path).resolve() if a.cf_graph_path else None
    if cfg['model']['use_cf_graph'] and (cf_graph is None or not cf_graph.is_file()):
        raise RuntimeError('use_cf_graph requires --cf-graph-path')
    limits={'train':a.limit_train,'val':a.limit_val,'test':a.limit_test}
    return cfg,root,run,shared,checkpoint,cache,cf_graph,limits


def upstream_identity(cfg,checkpoint,cache,root,cf_graph,smoke,limits):
    road=Path(root)/'data/processed/traffident/base_supportbench'/cfg['experiment']['target']/cfg['experiment']['dataset']
    graph_files=[road/'core/nodes.csv',road/'graph/adjacency.npy',road/'graph/distance.npy']
    return {'code_sha256':source_hash(ROOT),'backbone_sha256':sha256(checkpoint),'cache_path':str(cache),
            'split_manifest_sha256':{s:sha256(Path(cache)/f'{s}_manifest.csv') for s in ('train','val')},
            'road_graph_sha256':{p.name:sha256(p) for p in graph_files if p.is_file()},
            'cf_graph_sha256':sha256(cf_graph) if cf_graph else None,'smoke':bool(smoke),'limits':limits}


def check_identity(run,data):
    run.mkdir(parents=True,exist_ok=True)
    path=run/'run_identity.json'
    data=json.loads(json.dumps(data,sort_keys=True,default=str))
    if path.exists():
        if json.loads(path.read_text())!=data:raise RuntimeError(f'ReShift run identity mismatch in {run}; use a NEW --run-dir')
    else:
        if any(run.iterdir()):raise RuntimeError('Nonempty run directory without identity')
        atomic_json(data,path)


def build_model(cfg,root,checkpoint,s0_path,cf_graph,device):
    stla,bc=make_stla(root,checkpoint,s0_path,device,int(cfg['model']['stla_tail_layers']))
    nodes=int(stla.base.num_nodes)
    physical=build_road(root,cfg['experiment']['target'],cfg['experiment']['dataset'],
                        k_per_direction=int(cfg['model']['road_neighbors_per_direction']))
    if physical['ids'].shape[0]!=nodes:raise RuntimeError('Physical graph nodes do not match STLA')
    graph=dict(physical)
    if not cfg['model']['use_road_graph']:
        graph['valid']=np.zeros_like(physical['valid'],dtype=bool)
        graph['edge_features']=np.zeros_like(physical['edge_features'])
    if cf_graph is not None and cfg['model']['use_cf_graph']:
        graph=add_cf_edges(graph,cf_graph,max_cf_neighbors=int(cfg['model']['max_cf_neighbors']))
    model=CoRSTAv2(stla,graph,hidden=int(stla.model_dim),failure_dim=int(cfg['model']['failure_dim']),
                   cf_dim=int(cfg['model']['cf_dim']),residual_dim=int(cfg['model']['residual_dim']),
                   horizon=int(cfg['experiment']['horizon']),heads=int(cfg['model']['attention_heads']),
                   latent_scale=float(cfg['model']['latent_correction_scale']),
                   cf_error_gate=bool(cfg['model'].get('use_cf_error_gate',True)),
                   prediction_mode='normal_anchored',retrieval=cfg['model'].get('retrieval'),
                   **mechanism_kwargs(cfg)).to(device)
    return model,bc,physical


def validation(cfg,cache,graph,seed,limit=None):
    datasets={str(s):CoRSTASplit(cache,'val',seed=seed,severity=s,limit=limit,
                                 neighbors=graph['adjacency_list'],directions=graph['direction_labels'])
              for s in SEVERITIES}
    sel,lock=validation_partitions(datasets['20'],int(cfg['training']['selection_limit']))
    return datasets,sel,lock


def train_sets(cfg,cache,graph,seed,limit):
    kw=dict(seed=seed,limit=limit,augment=bool(cfg['training']['augment_training_masks']),
            failure_kinds=cfg['training'].get('failure_kinds'),legacy_multimask=False,
            neighbors=graph['adjacency_list'],directions=graph['direction_labels'])
    views=CoRSTASplit(cache,'train',contrast_views=bool(cfg['training'].get('multimask_error_contrast',False)),
                      num_views=int(cfg['training'].get('multimask_views',2)),**kw)
    plain=CoRSTASplit(cache,'train',contrast_views=False,**kw)
    return views,plain


def needs_provider(cfg,stage):
    r=retrieval_cfg(cfg)
    uses=bool(r) and (bool(r.get('use_features',True)) or any(x in RET_ARMS for x in r.get('arms',())))
    return stage in ('bank','m0','m1') or (uses and stage in ('anchor_ret','anchor_control','train','lock',
                                                             'test','audit','all'))


def main(a=None):
    a=parser().parse_args() if a is None else a
    cfg,root,run,shared,ckpt,cache,cf_graph,limits=compose(a)
    device=torch.device('cpu' if a.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type!='cuda' and a.stage!='preflight' and not a.smoke:
        raise RuntimeError('Formal ReShift runs require CUDA; use --smoke for CPU checks')
    log=lambda s:print(s,flush=True)
    seed_all(a.seed)
    stages=stages_for(cfg)
    up=upstream_identity(cfg,ckpt,cache,root,cf_graph,a.smoke,limits)
    keys=stage_keys(cfg,up,stages)
    layout=Layout(run,shared,keys)
    log(f'RESHIFT_START stage={a.stage} target={a.target} dataset={a.dataset} stages={list(stages)} '
        f'run={run} shared={shared} device={device}')
    if a.stage=='preflight':
        log('RESHIFT_PREFLIGHT_KEYS '+json.dumps({k:v[:16] for k,v in keys.items()}))
        sample=CoRSTASplit(cache,'train',seed=a.seed,limit=1)[0]
        log(f'RESHIFT_PREFLIGHT_PASS train_x={tuple(sample["x"].shape)} window_start={int(sample["window_start"])}')
        return
    check_identity(run,{'version':cfg['version'],'upstream':up,'configuration':cfg,'stage_keys':keys,
                        'shared_root':str(shared)})
    # ---------------------------------------------------------------- S0
    s0_dir=layout.dir('s0')
    if a.stage not in ('lock','test','audit','report','m1') and not (s0_dir/'COMPLETE.json').exists():
        lock=layout.acquire('s0')
        try:
            train_s0(root,ckpt,cache,s0_dir,keys['s0'],cfg.get('s0',{}),int(cfg['model']['stla_tail_layers']),
                     device,log,limits={'train':limits['train'],'val':limits['val']})
        finally:lock.unlink(missing_ok=True)
    if not (s0_dir/'COMPLETE.json').exists():raise RuntimeError(f'S0 missing ({s0_dir}); run --stage s0 first')
    if json.loads((s0_dir/'COMPLETE.json').read_text()).get('stage_key')!=keys['s0']:
        raise RuntimeError('S0 stage key mismatch')
    if a.stage=='s0':return
    model,bc,graph=build_model(cfg,root,ckpt,s0_dir/'best.pt',cf_graph,device)
    mean=float(bc['mean']);std=float(bc['std'])
    if std<=0:raise RuntimeError('Backbone invalid target std')
    provider=None
    if needs_provider(cfg,a.stage):
        memory=TrainMemory(cache,limit=limits['train'])
        ident={'s0_key':keys['s0'],'backbone_sha256':up['backbone_sha256'],
               'train_manifest_sha256':up['split_manifest_sha256']['train'],'limits':limits}
        provider=EvidenceProvider(memory,layout.bank_root(),model.stla,cfg.get('retrieval_bank',{}),device,ident,log)
    val_ds,selection,locked=validation(cfg,cache,graph,a.seed,limits['val'])
    eval_bs=int(cfg['training']['eval_batch_size'])
    tau0=float((cfg['model'].get('retrieval') or {}).get('tau_init',0.1))
    diag_dir=run/'diagnostics'
    final=stages[-1];has_graph=(bool(cfg['model']['use_road_graph']) or bool(cfg['model']['use_cf_graph']))

    if a.stage in ('bank','train','all'):
        if provider is not None:
            _,plain=train_sets(cfg,cache,graph,a.seed,limits['train'])
            if 'anchor_ret' in stages or needs_provider(cfg,'anchor_control'):
                prepare_train_banks(provider,plain,cfg,log)
            prepare_eval_banks(provider,val_ds,'val',cfg)
            log('RESHIFT_BANK_STAGE_PASS')
        if a.stage=='bank':return
    if a.stage=='m0':
        prepare_eval_banks(provider,val_ds,'val',cfg)
        res=diagnostics.m0(model.stla,provider,val_ds,selection,device,mean,std,tau0,eval_bs)
        atomic_json(res,diag_dir/'m0.json');log('RESHIFT_M0 '+json.dumps(res,sort_keys=True));return
    if a.stage=='m1':
        d=layout.dir('anchor_cf')
        if not (d/'COMPLETE.json').exists():raise RuntimeError('M1 needs a completed anchor_cf stage')
        load_model_state(model,checkpoint_load(d/'best.pt')['model'],True)
        model.set_stage('anchor_cf',use_graph=has_graph)
        prepare_eval_banks(provider,val_ds,'val',cfg)
        res=diagnostics.m1(model,provider,val_ds,selection,device,mean,std,tau0,eval_bs)
        atomic_json(res,diag_dir/'m1.json');log('RESHIFT_M1 '+json.dumps(res,sort_keys=True));return

    if a.stage in ('train','all') or a.stage in stages:
        train_ds,plain=train_sets(cfg,cache,graph,a.seed,limits['train'])
        end=stages.index(a.stage) if a.stage in stages else len(stages)-1
        for st in stages[:end+1]:
            train_stage(model,train_ds,val_ds,selection,device,cfg,mean,std,layout,st,log,
                        provider=provider,retrieval_train_ds=plain)
        if a.stage in stages:return

    def load_final():
        lock_path=run/'model_lock.json'
        if not lock_path.is_file():raise RuntimeError('TEST/AUDIT/REPORT require model_lock.json')
        lk=json.loads(lock_path.read_text());best=layout.dir(final)/'best.pt'
        if lk['stage_key']!=keys[final] or lk['sha256']!=sha256(best):raise RuntimeError('Model changed since lock')
        load_model_state(model,checkpoint_load(best)['model'],layout.is_shared(final))
        model.set_stage(final,use_graph=has_graph);model.eval()

    if a.stage in ('lock','all'):
        path=run/'model_lock.json'
        if path.exists():log('RESHIFT_LOCK_REUSE')
        else:
            d=layout.dir(final)
            if not (d/'COMPLETE.json').exists():raise RuntimeError('Cannot lock before the final stage is complete')
            load_model_state(model,checkpoint_load(d/'best.pt')['model'],layout.is_shared(final))
            model.set_stage(final,use_graph=has_graph)
            if provider is not None:prepare_eval_banks(provider,val_ds,'val',cfg)
            metrics=evaluate(model,val_ds,device,mean,std,eval_bs,locked,mode='full',provider=provider,split='val')
            atomic_json({'stage_key':keys[final],'sha256':sha256(d/'best.pt'),'checkpoint':str(d/'best.pt'),
                         'locked_validation':metrics},path)
            log('RESHIFT_LOCK_PASS '+json.dumps(metrics['macro_mae']))

    def test_set(sev):
        eval_cache=resolve_shared_cache(root,a.target,a.dataset,'evaluation',sev)
        return CoRSTASplit(eval_cache,'test',seed=a.seed,limit=limits['test'])

    if a.stage in ('test','all'):
        load_final()
        for sev in SEVERITIES:
            out=run/'test'/f's{sev}.json'
            if out.exists():
                if json.loads(out.read_text())['stage_key']!=keys[final]:raise RuntimeError('Test identity mismatch')
                continue
            ds=test_set(sev)
            if provider is not None:prepare_eval_banks(provider,{str(sev):ds},'test',cfg)
            metric=evaluate(model,{str(sev):ds},device,mean,std,eval_bs,mode='full',provider=provider,split='test')
            atomic_json({'stage_key':keys[final],'severity':sev,'metric':metric},out)
            log(f'RESHIFT_TEST_PASS severity=s{sev} '+json.dumps(metric['per_severity'][str(sev)]['final']))

    if a.stage in ('audit','all'):
        load_final()
        head=model.retrieval
        if head is None or not head.needs_evidence:
            log('RESHIFT_AUDIT_SKIP no retrieval evidence in this configuration')
        else:
            allow=[k not in RET_ARMS for k in head.arm_kinds]
            summary={}
            for sev in SEVERITIES:
                out=run/'audit'/f's{sev}.json'
                if out.exists():summary[str(sev)]=json.loads(out.read_text());continue
                ds=test_set(sev)
                prepare_eval_banks(provider,{str(sev):ds},'test',cfg)
                prepare_eval_banks(provider,{str(sev):ds},'test',cfg,wrong=True)
                res={}
                for mode in AUDIT_MODES+('no_retrieval',):
                    m=evaluate(model,{str(sev):ds},device,mean,std,eval_bs,mode='full',provider=provider,
                               split='test',evidence_mode=('none' if mode=='no_retrieval' else mode),
                               arm_mask=(allow if mode=='no_retrieval' else None))
                    res[mode]=m['per_severity'][str(sev)]
                base=res['full']['final']['mae']
                res['relative_mae_change_vs_full']={k:(v['final']['mae']-base)/max(base,1e-12)
                                                     for k,v in res.items() if isinstance(v,dict) and 'final' in v}
                res['note']='in-checkpoint interventions; the independently trained no-retrieval control is a separate config'
                atomic_json({'stage_key':keys[final],**res},out);summary[str(sev)]=res
                log(f'RESHIFT_AUDIT_PASS severity=s{sev} '+json.dumps(res['relative_mae_change_vs_full']))
            atomic_json(summary,run/'audit'/'summary.json')

    if a.stage in ('report','all'):
        load_final()
        results={}
        for sev in SEVERITIES:
            f=run/'test'/f's{sev}.json'
            if not f.exists():raise RuntimeError('Missing test: '+str(sev))
            results[str(sev)]=json.loads(f.read_text())['metric']['per_severity'][str(sev)]
        names=('backbone','normal','normal_cf_candidate','final')
        macro={k:{m:sum(results[str(s)][k][m] for s in SEVERITIES)/len(SEVERITIES) for m in ('mae','rmse')}
               for k in names}
        atomic_json({'stage_key':keys[final],'config':Path(a.config).name,'macro_severities':macro,
                     'per_severity':results},run/'summary.json')
        lines=[f'# ReShift {cfg["version"]} ({Path(a.config).stem}), locked TEST','',
               '| Output | Macro-MAE | Macro-RMSE |','|---|---:|---:|']
        lines+=[f'| {k} | {v["mae"]:.4f} | {v["rmse"]:.4f} |' for k,v in macro.items()]
        lines+=['','| Severity | Backbone | Normal | Final | Arms (fraction) |','|---|---:|---:|---:|---|']
        for s in SEVERITIES:
            r=results[str(s)];arms=r['cf_diagnostics'].get('arms',{})
            frac=', '.join(f'{k}:{v["fraction"]:.2f}' for k,v in arms.items())
            lines.append(f'| s{s} | {r["backbone"]["mae"]:.4f} | {r["normal"]["mae"]:.4f} | {r["final"]["mae"]:.4f} | {frac} |')
        (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        log('RESHIFT_REPORT_PASS '+json.dumps(macro))


if __name__=='__main__':main()
