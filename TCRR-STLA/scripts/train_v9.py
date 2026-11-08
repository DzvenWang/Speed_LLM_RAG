#!/usr/bin/env python3
"""v9 end-to-end entrypoint. Run --smoke before formal on Gadi."""
from __future__ import annotations
import argparse,os,sys,json,datetime,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from tcrr_stla.config import load_config,cache_path,resolve_stla_checkpoint,file_sha256
from tcrr_stla.v5.shared_adapter import fit_shared_adapter,initial_model
from tcrr_stla.v9.prepare import prepare_normal,prepare_view
from tcrr_stla.v9.graph import GraphBundle
from tcrr_stla.v9.training import run_training
from scripts.build_retrieval_cache import ensure_cache


def parse():
    p=argparse.ArgumentParser(description='TCRR v9 full-mask supervised CF utility trainer')
    p.add_argument('--config',default=str(ROOT/'configs/tcrr_stla_v9.yaml'))
    p.add_argument('--target',choices=('speed','flow'),required=True)
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--adapter-checkpoint',default=None,help='v5.3 frozen S0 adapter from matching checkpoint/data')
    p.add_argument('--physical-graph',default=None,help='optional trusted physical graph NPZ (requires node order check)')
    p.add_argument('--node-ids',default=None,help='dataset-ordered node ids NPY, required with physical graph')
    p.add_argument('--regions',default=None,help='optional physical-region [N] NPY for regional/directional masks')
    p.add_argument('--stage',choices=('all','prepare','train'),default='all')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--batch-size',type=int,default=None)
    p.add_argument('--prep-batch',type=int,default=None)
    p.add_argument('--workers',type=int,default=None)
    p.add_argument('--normal-epochs',type=int,default=None)
    p.add_argument('--transfer-epochs',type=int,default=None)
    p.add_argument('--utility-epochs',type=int,default=None)
    p.add_argument('--full-info-weight',type=float,default=None)
    p.add_argument('--cross-epochs',type=int,default=None,help='optional same-candidate cross-failure utility stage')
    p.add_argument('--lr',type=float,default=None)
    p.add_argument('--resume',default=None)
    p.add_argument('--output',default=None)
    return p.parse_args()

def main():
    a=parse()
    import yaml
    v9cfg=yaml.safe_load(Path(a.config).read_text())
    if v9cfg.get('version')!='9.1.0':raise ValueError('Expected v9.1.0 config')
    training_cfg=v9cfg['v9']['training']
    defaults={'batch_size':'batch_size','prep_batch':'replay_prep_batch',
              'workers':'workers','normal_epochs':'normal_epochs',
              'transfer_epochs':'transfer_epochs','utility_epochs':'utility_epochs',
              'cross_epochs':'cross_epochs','full_info_weight':'full_info_loss_weight','lr':'lr'}
    for name,key in defaults.items():
        v=(v9cfg['v9'] if name=='prep_batch' else training_cfg)[key]
        if getattr(a,name) is None:setattr(a,name,v)
    if a.batch_size<1 or a.batch_size>256 or a.prep_batch<1 or a.prep_batch>64:raise ValueError('invalid batch')
    if a.full_info_weight<0 or a.cross_epochs<0:raise ValueError('Loss weight/epochs must be nonnegative')
    if a.workers<0 or a.workers>10:raise ValueError('workers out of range')
    os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(ROOT/'configs/tcrr_stla_v5.yaml')
    if cfg['version']!='5.3.0':raise RuntimeError('Requires original v5.3 protocol')
    cfg['experiment'].update(target=a.target,dataset=a.dataset,seed=a.seed)
    cfg['runtime']['replay_bank_on_gpu']=False
    cfg['runtime']['normal_teacher_on_gpu']=False
    cfg['project']['retrieval_root']=str(Path(cfg['project']['retrieval_root'])/'v9_1_isolated')
    ck=resolve_stla_checkpoint(cfg,a.backbone_checkpoint)
    cache=cache_path(cfg,'formal')
    if not (cache/'train_x_full_norm.npy').is_file():raise FileNotFoundError('Missing complete TRAIN observations')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not a.smoke:
        raise RuntimeError('Formal v9 preparation/training requires CUDA; use --smoke for CPU-only checks')
    output=(Path(a.output).resolve() if a.output else Path(a.resume).resolve().parent if a.resume else
            Path(cfg['project']['exp_root'])/a.dataset/'v9_1'/('smoke' if a.smoke else 'formal')/
            f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_seed{a.seed}')
    output.mkdir(parents=True,exist_ok=True)
    adapter=(Path(a.adapter_checkpoint).resolve() if a.adapter_checkpoint else
        Path(cfg['project']['retrieval_root'])/a.dataset/('v9_1_adapter_smoke' if a.smoke else 'v9_1_adapter')/
        f'{file_sha256(ck)[:16]}_seed{a.seed}'/'adapter.pt')
    if a.stage!='train':
        if a.adapter_checkpoint:
            if not adapter.is_file():raise FileNotFoundError(adapter)
        else:
            fit_shared_adapter(cfg,ck,cache,adapter,device,smoke=a.smoke)
    if not adapter.is_file():raise FileNotFoundError('S0 adapter missing; run --stage prepare first')
    teacher_device=device if a.stage!='train' else torch.device('cpu')
    teacher,backbone_ck=initial_model(cfg,ck,teacher_device,adapter)
    teacher.eval();[p.requires_grad_(False) for p in teacher.parameters()]
    fingerprint=hashlib.sha256((file_sha256(ck)+file_sha256(adapter)).encode()).hexdigest()
    evidence=Path(cfg['project']['retrieval_root'])/a.dataset/'v9_1_evidence'
    if a.smoke:evidence=Path(cfg['project']['retrieval_root'])/a.dataset/'v9_1_evidence_smoke'
    evidence.mkdir(parents=True,exist_ok=True)
    if a.physical_graph and not a.node_ids:
        raise ValueError('--node-ids mandatory with --physical-graph')
    n=np.load(cache/'train_x_full_norm.npy',mmap_mode='r').shape[-1]
    if a.physical_graph:
        with np.load(a.physical_graph,allow_pickle=False) as z:
            expected=np.load(a.node_ids,allow_pickle=False)
            if 'node_ids' not in z or not np.array_equal(z['node_ids'].astype(str),expected.astype(str)):
                raise ValueError('Physical graph node order does not match dataset node ids')
    regions=None
    if a.regions:
        regions=np.load(a.regions,allow_pickle=False)
        if regions.shape!=(n,):raise ValueError('regions size mismatch')
    graph=GraphBundle.from_cache(cache,n,a.physical_graph,k=8)
    # Inputs that affect any normalized Teacher/Replay must invalidate cache.
    source_files=['train_x_full_norm.npy','train_y_norm.npy','train_history_observed.npy',
                  'train_artificial_missing.npy','train_manifest.csv','val_history_observed.npy',
                  'val_x_obs_norm.npy','val_manifest.csv']
    source={'schema':9,'target':a.target,'dataset':a.dataset,'seed':a.seed,
            'smoke':a.smoke,'teacher':fingerprint,'cache':str(Path(cache).resolve()),
            'source_hashes':{v:file_sha256(cache/v) for v in source_files},
            'physical_graph_sha':file_sha256(a.physical_graph) if a.physical_graph else None,
            'node_ids_sha':file_sha256(a.node_ids) if a.node_ids else None,
            'regions_sha':file_sha256(a.regions) if a.regions else None,
            'view_protocol':'official_fixed_candidate_pair_plus_deterministic_failure_v9_1',
            'v9_implementation_sha':file_sha256(ROOT/'tcrr_stla/v9/model.py'),
            'v9_preparation_sha':file_sha256(ROOT/'tcrr_stla/v9/prepare.py'),
            'v9_views_sha':file_sha256(ROOT/'tcrr_stla/v9/views.py')}
    meta=evidence/'identity.json'
    if meta.exists():
        if json.loads(meta.read_text())!=source:
            raise RuntimeError('Existing v9 evidence conflicts with requested data/teacher. '
                               'Use a separate cache or explicitly remove the v9_evidence directory.')
    else:
        existing=list(evidence.glob('**/*.npy'))
        if existing:raise RuntimeError('Unidentified v9 cache exists without identity.json')
        meta.write_text(json.dumps(source,indent=2))
    if a.stage!='train':
        rt=ensure_cache(cfg,cache,'train',smoke=a.smoke)
        rv=ensure_cache(cfg,cache,'val',smoke=a.smoke)
        prepare_normal(cache,teacher,device,evidence,chunk=32,limit=256 if a.smoke else None)
        prepare_view(cache,'train',0,teacher,device,evidence,rt,k=12,chunk=a.prep_batch,
                     limit=24 if a.smoke else None,seed=a.seed,regions=regions,teacher_sha=fingerprint)
        prepare_view(cache,'train',1,teacher,device,evidence,rt,k=12,chunk=a.prep_batch,
                     limit=24 if a.smoke else None,seed=a.seed,regions=regions,teacher_sha=fingerprint)
        prepare_view(cache,'val',0,teacher,device,evidence,rv,k=12,chunk=a.prep_batch,
                     limit=24 if a.smoke else None,seed=a.seed,regions=regions,teacher_sha=fingerprint)
        print('V9_PREPARATION_PASS',evidence,flush=True)
    if a.stage=='prepare':return
    del teacher
    if torch.cuda.is_available():torch.cuda.empty_cache()
    run_identity=dict(
        version='9.1.0',target=a.target,dataset=a.dataset,seed=a.seed,
        backbone_checkpoint=str(ck),backbone_sha256=file_sha256(ck),
        adapter_checkpoint=str(adapter),adapter_sha256=file_sha256(adapter),
        graph_physical=str(a.physical_graph) if a.physical_graph else None,
        node_ids=str(a.node_ids) if a.node_ids else None,
        evidence_root=str(evidence),formal_cache=str(cache),teacher_fingerprint=fingerprint,
        smoke=a.smoke)
    identity_path=output/'run_identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text())!=run_identity:
            raise RuntimeError('Existing v9 run identity differs from this launch. Refuse overwrite.')
    else:identity_path.write_text(json.dumps(run_identity,indent=2))
    train_cfg=dict(batch_size=a.batch_size,eval_batch_size=min(a.batch_size,32),workers=a.workers,
                   normal_epochs=a.normal_epochs,transfer_epochs=a.transfer_epochs,
                   utility_epochs=a.utility_epochs,cross_epochs=a.cross_epochs,
                   cross_weight=float(training_cfg['cross_failure_loss_weight']),
                   lr=a.lr,hidden=int(v9cfg['v9']['hidden']),bound=float(v9cfg['v9']['bound']),
                   blocks=int(v9cfg['v9']['horizon_blocks']),
                   amp=bool(training_cfg['amp_fp16']),
                   utility_weight=float(training_cfg['utility_loss_weight']),
                   ranking_weight=float(training_cfg['ranking_loss_weight']),
                   full_info_weight=a.full_info_weight,
                   teacher_alignment_weight=float(training_cfg.get('teacher_alignment_loss_weight',0.)),
                   clip=float(training_cfg['grad_clip']),
                   patience=int(training_cfg['patience']))
    from tcrr_stla.v9.prepare import PreparedView
    context_dim=int(PreparedView(evidence/'train_view0').arrays['context'].shape[-1])
    best=run_training(cache,evidence,graph,context_dim,
          float(backbone_ck['mean']),float(backbone_ck['std']),output,a.seed,train_cfg,
          smoke=a.smoke,resume=a.resume)
    print('V9_TRAIN_PASS',best,flush=True)

if __name__=='__main__':main()
