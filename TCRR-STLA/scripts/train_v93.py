#!/usr/bin/env python3
"""Full v9.3 CLI: official benchmark preparation, direct model training, diagnostics.

Root is SPEED_LLM_RAG_ROOT; Base/Backbone unchanged. Formal training requires CUDA.
"""
from __future__ import annotations
import sys,os,argparse,json,datetime,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,torch,yaml
from tcrr_stla.config import load_config,resolve_stla_checkpoint,cache_path,file_sha256
from tcrr_stla.v5.shared_adapter import fit_shared_adapter,initial_model
from tcrr_stla.v93.teacher import build_identity,prepare_split,atomic_json
from tcrr_stla.v93.training import run_training
from tcrr_stla.v93.model import TCRRV93
from tcrr_stla.v93.graph_encoder import functional_graph
from tcrr_stla.v93.road_graph import load_or_build_graph


def cli():
    p=argparse.ArgumentParser(description='TCRR-STLA v9.3 full-reference multi-mask direct counterfactual model')
    p.add_argument('--config',default=str(ROOT/'configs/tcrr_stla_v93.yaml'))
    p.add_argument('--target',required=True,choices=('speed','flow'))
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--adapter-checkpoint')
    p.add_argument('--stage',choices=('all','prepare','train','diagnose'),default='all')
    p.add_argument('--mode',choices=('incremental','decomposed','single'))
    p.add_argument('--graph-mode',choices=('none','fixed','v7_edge_attention'))
    p.add_argument('--teacher-root')
    p.add_argument('--output')
    p.add_argument('--resume')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--no-auto-diagnose',action='store_true')
    for k in ('batch_size','prep_batch','workers','train_limit','val_limit'):
        p.add_argument('--'+k.replace('_','-'),type=int)
    return p.parse_args()


def build_graph(cfg,cache,settings,output,target,dataset):
    mode=settings['graph']['mode']
    if mode=='none':return None,None
    root=cfg['project']['root']
    road=load_or_build_graph(root,target,dataset,cache,Path(output)/'graph',
                             k=int(settings['graph']['road_k']))
    functional=functional_graph(cache,k=int(settings['graph']['functional_k']),
                                 max_windows=int(settings['graph']['functional_max_windows']))
    return road,functional


def main():
    a=cli()
    settings=yaml.safe_load(Path(a.config).read_text())
    if settings.get('version')!='9.3.0':raise RuntimeError('Need v9.3.0 configuration')
    if a.mode:settings['model']['mode']=a.mode
    if a.graph_mode:settings['graph']['mode']=a.graph_mode
    for arg,path in (('batch_size',('training','batch_size')),
                     ('prep_batch',('preparation','batch_size')),
                     ('workers',('training','workers')),
                     ('train_limit',('preparation','limit')),
                     ('val_limit',('preparation','val_limit'))):
        if getattr(a,arg) is not None:settings[path[0]][path[1]]=getattr(a,arg)
    if a.smoke:
        settings['preparation']['limit']=min(int(settings['preparation']['limit'] or 24),24)
        settings['preparation']['val_limit']=min(int(settings['preparation']['val_limit'] or 24),24)
        settings['training']['batch_size']=min(int(settings['training']['batch_size']),4)
        settings['training']['workers']=0
        settings['preparation']['batch_size']=min(int(settings['preparation']['batch_size']),2)
        for s in settings['training']['epochs']:
            if settings['training']['epochs'][s]>0:settings['training']['epochs'][s]=1
    os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(ROOT/'configs/tcrr_stla_v5.yaml')
    if cfg['version']!='5.3.0':raise RuntimeError('Expected v5.3 Base SupportBench protocol')
    cfg['experiment'].update(target=a.target,dataset=a.dataset,seed=a.seed)
    ck=resolve_stla_checkpoint(cfg,a.backbone_checkpoint)
    cache=cache_path(cfg,'formal')
    if not(cache/'train_x_full_norm.npy').is_file():raise FileNotFoundError('Missing TRAIN complete history')
    root=Path(cfg['project']['root'])
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not a.smoke and dev.type!='cuda' and a.stage in ('all','prepare','train'):
        raise RuntimeError('Formal v9.3 requires CUDA; --smoke allows CPU checks')
    adapter=Path(a.adapter_checkpoint).resolve() if a.adapter_checkpoint else (
        Path(cfg['project']['retrieval_root'])/a.dataset/('v93_adapter_smoke' if a.smoke else 'v93_adapter')/f'{file_sha256(ck)[:16]}_seed{a.seed}'/'adapter.pt')
    if not adapter.exists():
        if a.adapter_checkpoint or a.stage not in ('all','prepare'):
            raise FileNotFoundError('Adapter checkpoint missing; run --stage prepare or pass existing adapter')
        fit_shared_adapter(cfg,ck,cache,adapter,dev,smoke=a.smoke)
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run=Path(a.output).resolve() if a.output else Path(cfg['project']['exp_root'])/a.dataset/'v9_3'/settings['model']['mode']/('smoke' if a.smoke else 'formal')/f'{stamp}_seed{a.seed}'
    run.mkdir(parents=True,exist_ok=True)
    if a.resume and Path(a.resume).resolve().parent!=run.resolve():
        raise ValueError('--resume must point to the selected --output run directory')
    trlimit=settings['preparation']['limit'];vlimit=settings['preparation']['val_limit']
    teacher_root=Path(a.teacher_root).resolve() if a.teacher_root else (Path(cfg['project']['retrieval_root'])/a.dataset/'v93_multimask_teacher'/('smoke' if a.smoke else 'formal')/f'{file_sha256(ck)[:12]}_{file_sha256(adapter)[:12]}_seed{a.seed}')
    identity=build_identity(cache,root,ck,adapter,a.target,a.dataset,a.seed,
                       int(settings['counterfactual']['views_per_query']),trlimit,vlimit)
    if (teacher_root/'identity.json').exists():
        if json.loads((teacher_root/'identity.json').read_text())!=identity:
            raise RuntimeError('Teacher root is occupied by different source identity')
    else:
        teacher_root.mkdir(parents=True,exist_ok=True)
        if list(teacher_root.rglob('*.npy')):raise RuntimeError('Unidentified teacher data present')
        atomic_json(teacher_root/'identity.json',identity)
    if a.stage in ('prepare','all'):
        teacher,_=initial_model(cfg,ck,dev,adapter)
        teacher.eval()
        for p in teacher.parameters():p.requires_grad_(False)
        prepare_split(cache,teacher_root/'train','train',teacher,dev,root,a.target,a.dataset,a.seed,
                      int(settings['counterfactual']['views_per_query']),
                      int(settings['preparation']['batch_size']),trlimit,identity)
        prepare_split(cache,teacher_root/'val','val',teacher,dev,root,a.target,a.dataset,a.seed,
                      1,int(settings['preparation']['batch_size']),vlimit,identity)
        print('V93_PREPARATION_PASS root='+str(teacher_root),flush=True)
    if a.stage=='prepare':return
    from tcrr_stla.v93.diagnostics.audit import diagnose_run
    if a.stage=='diagnose':
        diagnose_run(run,cache,teacher_root,settings,ck,adapter,root,a.target,a.dataset)
        return
    meta=json.loads((teacher_root/'train'/'manifest.json').read_text())
    road,functional=build_graph(cfg,cache,settings,run,a.target,a.dataset)
    model=TCRRV93(context_dim=int(meta['context_dim']),num_nodes=int(meta['nodes']),
         horizon=int(settings['model']['horizon']),hidden=int(settings['model']['hidden']),
         bound=float(settings['model']['correction_bound']),mode=settings['model']['mode'],
         graph_mode=settings['graph']['mode'],road_graph=road,functional=functional)
    summary=run_training(cache,teacher_root,run,settings,model,a.seed,a.smoke,a.resume)
    if not a.no_auto_diagnose:
        diagnose_run(run,cache,teacher_root,settings,ck,adapter,root,a.target,a.dataset,model=model)
    print('V93_RUN_FINISH output='+str(run),flush=True)

if __name__=='__main__':main()
