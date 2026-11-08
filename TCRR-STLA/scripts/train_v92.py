#!/usr/bin/env python3
"""v9.2 complete entrypoint. Formal CUDA, independent caches, auto CF diagnostics.

No GPU smoke is required. --smoke permits CPU synthetic/protocol checks only.
"""
from __future__ import annotations
import argparse,os,sys,json,datetime,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from tcrr_stla.config import load_config,cache_path,resolve_stla_checkpoint,file_sha256
from tcrr_stla.v5.shared_adapter import fit_shared_adapter,initial_model
from tcrr_stla.v92.prepare import prepare_normal,prepare_view,PreparedView
from tcrr_stla.v9.graph import GraphBundle
from tcrr_stla.v92.training import run_training
from tcrr_stla.v92 import VERSION
from scripts.build_retrieval_cache import ensure_cache


def parse():
    p=argparse.ArgumentParser(description='TCRR v9.2 full supervised CF trainer')
    p.add_argument('--config',default=str(ROOT/'configs/tcrr_stla_v92.yaml'))
    p.add_argument('--target',required=True,choices=['speed','flow'])
    p.add_argument('--dataset',default='TraffiDent_Net_06');p.add_argument('--seed',type=int,default=10)
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--adapter-checkpoint');p.add_argument('--physical-graph');p.add_argument('--node-ids')
    p.add_argument('--regions');p.add_argument('--stage',choices=('all','prepare','train'),default='all')
    p.add_argument('--smoke',action='store_true');p.add_argument('--output');p.add_argument('--resume')
    for name in ('batch_size','prep_batch','workers','normal_epochs','align_epochs','utility_epochs','joint_epochs','paired_epochs'):
        p.add_argument('--'+name.replace('_','-'),type=int)
    p.add_argument('--no-auto-diagnose',action='store_true')
    p.add_argument('--diagnostic-queries',type=int)
    p.add_argument('--alignment-mode',choices=('signed','positive','raw','off'))
    p.add_argument('--graph-mode',choices=('none','fixed','gnn','gnn_reliability'))
    p.add_argument('--temporal-mode',choices=('cnn','stats'))
    p.add_argument('--null-mode',choices=('conditional','fixed'))
    p.add_argument('--utility-mode',choices=('learned','uniform','similarity'))
    p.add_argument('--paired-loss-off',action='store_true',help='Same-budget paired multi-mask control without cross/compatibility losses')
    return p.parse_args()


def main():
    a=parse();c=yaml.safe_load(Path(a.config).read_text())
    if c.get('version')!=VERSION:raise ValueError('Expected v9.2.0 YAML')
    settings=c['v92'];training=settings['training'];dc=settings['diagnostics']
    for name in ('alignment_mode','graph_mode','temporal_mode','null_mode','utility_mode'):
        override=getattr(a,name)
        if override is not None:settings[name]=override
    for name in ('batch_size','prep_batch','workers','normal_epochs','align_epochs','utility_epochs','joint_epochs','paired_epochs'):
        if getattr(a,name) is None:setattr(a,name,training[name])
    if not(1<=a.batch_size<=256 and 1<=a.prep_batch<=64 and 0<=a.workers<=10):
        raise ValueError('Invalid batch/workers range')
    if a.normal_epochs<1 or any(getattr(a,name)<0 for name in ('align_epochs','utility_epochs','joint_epochs','paired_epochs')):
        raise ValueError('Invalid epoch count')
    os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(ROOT/'configs/tcrr_stla_v5.yaml')
    if cfg['version']!='5.3.0':raise RuntimeError('v9.2 requires v5.3 SupportBench protocol')
    cfg['experiment'].update(target=a.target,dataset=a.dataset,seed=a.seed)
    cfg['runtime']['replay_bank_on_gpu']=False;cfg['runtime']['normal_teacher_on_gpu']=False
    cfg['project']['retrieval_root']=str(Path(cfg['project']['retrieval_root'])/'v9_2_isolated')
    ck=resolve_stla_checkpoint(cfg,a.backbone_checkpoint)
    # v5 retrieval builder resolves its teacher through the environment. Bind it
    # to the EXACT SAME checkpoint as v9.2 base and counterfactual replay.
    os.environ['BACKBONE_CHECKPOINT']=str(ck)
    cache=cache_path(cfg,'formal')
    if not(cache/'train_x_full_norm.npy').is_file():raise FileNotFoundError('Missing complete TRAIN')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not a.smoke:raise RuntimeError('Formal v9.2 requires CUDA')
    output=(Path(a.output).resolve() if a.output else Path(a.resume).resolve().parent if a.resume else
        Path(cfg['project']['exp_root'])/a.dataset/'v9_2'/('smoke' if a.smoke else 'formal')/
        f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_seed{a.seed}')
    output.mkdir(parents=True,exist_ok=True)
    adapter=(Path(a.adapter_checkpoint).resolve() if a.adapter_checkpoint else
       Path(cfg['project']['retrieval_root'])/a.dataset/('v92_adapter_smoke' if a.smoke else 'v92_adapter')/
       f'{file_sha256(ck)[:16]}_seed{a.seed}'/'adapter.pt')
    if a.stage!='train':
        if a.adapter_checkpoint:
            if not adapter.is_file():raise FileNotFoundError(adapter)
        else:fit_shared_adapter(cfg,ck,cache,adapter,device,smoke=a.smoke)
    if not adapter.is_file():raise FileNotFoundError('Adapter missing; prepare stage is required')
    teacher_device=device if a.stage!='train' else torch.device('cpu')
    teacher,backbone_ck=initial_model(cfg,ck,teacher_device,adapter)
    teacher.eval();[p.requires_grad_(False) for p in teacher.parameters()]
    fingerprint=hashlib.sha256((file_sha256(ck)+file_sha256(adapter)).encode()).hexdigest()
    evidence=Path(cfg['project']['retrieval_root'])/a.dataset/('v92_evidence_smoke' if a.smoke else 'v92_evidence')
    evidence.mkdir(parents=True,exist_ok=True)
    if a.physical_graph and not a.node_ids:raise ValueError('Physical graph requires ordered node ids')
    n=np.load(cache/'train_x_full_norm.npy',mmap_mode='r').shape[-1]
    if a.physical_graph:
        with np.load(a.physical_graph,allow_pickle=False) as g:
            ids=np.load(a.node_ids,allow_pickle=False)
            if 'node_ids' not in g or not np.array_equal(g['node_ids'].astype(str),ids.astype(str)):
                raise ValueError('Physical graph node order mismatch')
    regions=None
    if a.regions:
        regions=np.load(a.regions,allow_pickle=False)
        if regions.shape!=(n,):raise ValueError('Region labels must match dataset nodes')
    graph=GraphBundle.from_cache(cache,n,a.physical_graph,k=8)
    source_files=['train_x_full_norm.npy','train_x_obs_norm.npy','train_y_norm.npy',
       'train_history_observed.npy','train_artificial_missing.npy','train_future_valid.npy',
       'train_tod_hist.npy','train_dow_hist.npy','train_manifest.csv',
       'val_history_observed.npy','val_artificial_missing.npy','val_future_valid.npy',
       'val_x_obs_norm.npy','val_tod_hist.npy','val_dow_hist.npy','val_manifest.csv']
    source={'schema':92,'target':a.target,'dataset':a.dataset,'seed':a.seed,'smoke':a.smoke,
       'teacher':fingerprint,'cache':str(Path(cache).resolve()),
       'source_hashes':{name:(file_sha256(cache/name) if (cache/name).is_file() else None)
                        for name in source_files},
       'physical_graph_sha':file_sha256(a.physical_graph) if a.physical_graph else None,
       'node_ids_sha':file_sha256(a.node_ids) if a.node_ids else None,
       'regions_sha':file_sha256(a.regions) if a.regions else None,
       'v92_preparation_sha':file_sha256(ROOT/'tcrr_stla/v92/prepare.py'),
       'v92_retrieval_sha':file_sha256(ROOT/'tcrr_stla/v92/retrieval.py'),
       'views_sha':file_sha256(ROOT/'tcrr_stla/v9/views.py')}
    ident=evidence/'identity.json'
    if ident.is_file():
        if json.loads(ident.read_text())!=source:raise RuntimeError('Existing evidence cache identity conflicts; do not overwrite')
    else:
        if list(evidence.rglob('*.npy')):raise RuntimeError('Unidentified evidence files; refusing overwrite')
        ident.write_text(json.dumps(source,indent=2))
    if a.stage!='train':
        prepare_normal(cache,teacher,device,evidence,chunk=32,limit=256 if a.smoke else None)
        shortlist_train=ensure_cache(cfg,cache,'train',smoke=a.smoke)
        shortlist_val=ensure_cache(cfg,cache,'val',smoke=a.smoke)
        for split,view in (('train',0),('train',1),('val',0)):
            shortlist=(shortlist_train if split=='train' else shortlist_val) if view==0 else None
            prepare_view(cache,split,view,teacher,device,evidence,shortlist,k=12,chunk=a.prep_batch,
                 limit=24 if a.smoke else None,seed=a.seed,regions=regions,teacher_sha=fingerprint)
        print('V92_PREPARATION_PASS',evidence,flush=True)
    if a.stage=='prepare':return
    del teacher
    if torch.cuda.is_available():torch.cuda.empty_cache()
    run_identity=dict(version=VERSION,target=a.target,dataset=a.dataset,seed=a.seed,
         backbone_checkpoint=str(ck),backbone_sha256=file_sha256(ck),
         adapter_checkpoint=str(adapter),adapter_sha256=file_sha256(adapter),
         graph_physical=str(a.physical_graph) if a.physical_graph else None,
         node_ids=str(a.node_ids) if a.node_ids else None,
         regions=str(a.regions) if a.regions else None,
         evidence_root=str(evidence),formal_cache=str(cache),teacher_fingerprint=fingerprint,
         smoke=bool(a.smoke),v92_evidence_schema=92,
         config_file_sha=file_sha256(a.config),
         configured_variant={name:settings[name] for name in ('alignment_mode','graph_mode','temporal_mode','null_mode','utility_mode')},
         paired_loss_off=bool(a.paired_loss_off),
         epoch_budget={name:getattr(a,name) for name in ('normal_epochs','align_epochs','utility_epochs','joint_epochs','paired_epochs')})
    run_identity_path=output/'run_identity.json'
    if run_identity_path.exists():
        if json.loads(run_identity_path.read_text())!=run_identity:
            raise RuntimeError('Run identity differs; refusing to overwrite')
    else:run_identity_path.write_text(json.dumps(run_identity,indent=2))
    train_cfg=dict(batch_size=a.batch_size,eval_batch_size=min(a.batch_size,32),workers=a.workers,
       normal_epochs=a.normal_epochs,align_epochs=a.align_epochs,utility_epochs=a.utility_epochs,
       joint_epochs=a.joint_epochs,paired_epochs=a.paired_epochs,
       mean=float(backbone_ck['mean']),std=float(backbone_ck['std']),
       lr=float(training['lr']),hidden=int(settings['hidden']),cf_hidden=int(settings['cf_hidden']),
       bound=float(settings['bound']),blocks=int(settings['horizon_blocks']),
       max_align=float(settings['max_align']),temperature=float(settings['temperature']),
       alignment_mode=settings['alignment_mode'],graph_mode=settings['graph_mode'],
       temporal_mode=settings['temporal_mode'],null_mode=settings['null_mode'],
       amp=bool(training['amp_fp16']),clip=float(training['clip']),patience=int(training['patience']),
       **{key:float(training[key]) for key in ('utility_weight','rank_weight','alignment_weight',
           'marginal_weight','full_info_weight','cross_weight','compat_weight')})
    if a.paired_loss_off:
        train_cfg['cross_weight']=0.
        train_cfg['compat_weight']=0.
    context_dim=int(PreparedView(evidence/'train_view0').arrays['context'].shape[-1])
    best=run_training(cache,evidence,graph,context_dim,float(backbone_ck['mean']),
          float(backbone_ck['std']),output,a.seed,train_cfg,smoke=a.smoke,resume=a.resume)
    print('V92_TRAIN_PASS',best,flush=True)
    if not a.no_auto_diagnose and dc['auto_after_training']:
        from tcrr_stla.v92.diagnostics import diagnose_run
        diagnose_run(run_identity_path,max_queries=a.diagnostic_queries or int(dc['max_queries']),
             bootstrap=int(dc['bootstrap_samples']),wrong_mask_queries=int(dc['wrong_mask_queries']))
        print('V92_AUTO_DIAGNOSTICS_PASS',output/'diagnostics',flush=True)
if __name__=='__main__':main()
