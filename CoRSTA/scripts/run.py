#!/usr/bin/env python3
"""End-to-end CoRSTA orchestrator. Use --stage preflight/bootstrap/warmup/joint/lock/test/report/all.

CLI works from Windows and Gadi. Stage outputs are isolated and manifest-locked.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from corsta.observability import install_timestamps
install_timestamps()
import torch,yaml
from corsta.config import file_sha256
from corsta.v7.runner import atomic_json,seed_everything
from corsta.training import (train_phase,lock,test_severity,report)
from scripts.bootstrap_v7 import Params

STAGES=('preflight','bootstrap','warmup','joint','lock','test','report','all')
ORDER=('bootstrap','warmup','joint','lock','test','report')


def arguments():
    p=argparse.ArgumentParser(description='CoRSTA: end-to-end STLA+C1/C2')
    p.add_argument('--stage',choices=STAGES,default='all')
    p.add_argument('--target',choices=('speed','flow'),required=True)
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--config',default=str(ROOT/'configs/corsta_v0_1.yaml'))
    p.add_argument('--bootstrap-config',default=str(ROOT/'configs/bootstrap_v7.yaml'))
    p.add_argument('--run-dir')
    p.add_argument('--workers',type=int)
    p.add_argument('--batch-initial',type=int,choices=(16,32,64,128))
    p.add_argument('--batch-mode',choices=('auto','fixed'))
    p.add_argument('--limit-train',type=int)
    p.add_argument('--limit-val',type=int)
    p.add_argument('--limit-test',type=int)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--cpu',action='store_true',help='Force CPU (allowed for preflight or smoke)')
    p.add_argument('--require-cuda',action='store_true',help='Require CUDA for GPU preflight (fail rather than silently falling back to CPU)')
    return p.parse_args()


def file_hashes():
    paths=list((ROOT/'corsta').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'configs').glob('*.yaml'))
    h=hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def compose(a):
    with open(a.config,encoding='utf8') as fp:cfg=yaml.safe_load(fp)
    if cfg['version']!='0.1.0':raise RuntimeError('Unsupported CoRSTA version')
    cfg['experiment'].update(target=a.target,dataset=a.dataset,seed=a.seed)
    if a.batch_initial is not None:cfg['training']['batch_initial']=a.batch_initial
    if a.batch_mode is not None:cfg['training']['batch_mode']=a.batch_mode
    if a.workers is not None:cfg['training']['worker_count']=a.workers
    if a.smoke:
        cfg['training'].update(warmup_epochs=1,joint_epochs=1,patience=1,worker_count=0,
                               eval_batch_size=4,effective_batch=16)
        if a.batch_initial is None:cfg['training']['batch_initial']=16
    root=Path(os.environ.get('SPEED_LLM_RAG_ROOT',ROOT.parent)).resolve()
    mode=os.environ.get('BACKBONE_MODE','incomplete').lower()
    if mode not in ('complete','incomplete'):raise ValueError('BACKBONE_MODE')
    run=(Path(a.run_dir).resolve() if a.run_dir else
         root/'CoRSTA_preflight'/a.target/mode/a.dataset/f'seed{a.seed}' if a.stage=='preflight' else
         root/'CoRSTA_exps'/a.target/mode/a.dataset/('smoke' if a.smoke else 'formal')/f'seed{a.seed}')
    legacy=run/'bootstrap_v7'
    la=argparse.Namespace(target=a.target,dataset=a.dataset,backbone_checkpoint=a.backbone_checkpoint,
         seed=a.seed,config=a.bootstrap_config,run_dir=str(legacy),step=None,
         check_run_identity=False,smoke=a.smoke,batch_size=64,workers=a.workers,
         limit_train=a.limit_train,limit_val=a.limit_val,limit_test=a.limit_test)
    os.environ['SPEED_LLM_RAG_ROOT']=str(root)
    p=Params(la)
    return cfg,root,run,p


def identity(a,cfg,run,p):
    os.makedirs(run,exist_ok=True)
    value={'version':'0.1.0','code_sha256':file_hashes(),
           'backbone_sha256':file_sha256(p.backbone_checkpoint),
           'target':a.target,'dataset':a.dataset,'seed':a.seed,
           'smoke':a.smoke,'limits':p.limits,'backbone_mode':p.backbone_mode,
           'cfg':cfg,'bootstrap_config_sha256':file_sha256(a.bootstrap_config)}
    h=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    path=run/'run_identity.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old!=value:
            # Previous v0.1.0 preflight mistakenly wrote a formal run manifest
            # before checking CUDA. Safely refresh ONLY an unstarted run when
            # its sole identity difference is the repaired source-code digest.
            legacy=run/'bootstrap_v7'
            has_only_scaffold=all(x.name in ('run_identity.json','bootstrap_v7') for x in run.iterdir())
            legacy_empty=(not legacy.exists() or (legacy.is_dir() and not any(legacy.iterdir())))
            only_source_changed=(
                {k:v for k,v in old.items() if k!='code_sha256'} ==
                {k:v for k,v in value.items() if k!='code_sha256'}
            )
            if has_only_scaffold and legacy_empty and only_source_changed:
                atomic_json(value,path)
                print('CORSTA_RUN_IDENTITY_REFRESH reason=unstarted_preflight_hotfix '
                      f'old_code={old.get("code_sha256", "unknown")[:12]} '
                      f'new_code={value["code_sha256"][:12]}',flush=True)
            else:
                raise RuntimeError('CoRSTA run identity mismatch: use a fresh --run-dir; '
                                   'never reuse checkpoints after code/config change')
    else:
        existing=list(run.iterdir())
        if existing and not (len(existing)==1 and existing[0].name=='bootstrap_v7' and not any(existing[0].iterdir())):
            raise RuntimeError(f'Nonempty CoRSTA run without manifest: {run}')
        atomic_json(value,path)
    return h


def do_bootstrap(a,p):
    """Reuse TCRR v7.0.1 preprocessing + C1 initial training; no legacy TEST access."""
    cmd=[sys.executable,'-u',str(ROOT/'scripts/bootstrap_v7.py'),
         '--target',a.target,'--dataset',a.dataset,'--seed',str(a.seed),
         '--backbone-checkpoint',str(p.backbone_checkpoint),
         '--config',a.bootstrap_config,'--run-dir',str(p.run),
         '--batch-size','64','--workers',str(p.workers)]
    if a.smoke:cmd.append('--smoke')
    for k,flag in [('train','--limit-train'),('val','--limit-val'),('test','--limit-test')]:
        if p.limits[k] is not None:cmd.extend([flag,str(p.limits[k])])
    for stage in ('s0','cache','replay','normal','c1'):
        if (p.run/f'STAGE_{stage}.done').is_file():
            print(f'CORSTA_BOOTSTRAP_REUSE stage={stage}',flush=True);continue
        print(f'CORSTA_BOOTSTRAP_START stage={stage}',flush=True)
        subprocess.run(cmd+['--step',stage],cwd=str(p.root),check=True)
        required={'s0':p.run/'s0.pt','normal':p.run/'best_normal.pt','c1':p.run/'best_c1.pt'}
        if stage in required and not required[stage].is_file():raise RuntimeError(f'Incomplete bootstrap {stage}')
        (p.run/f'STAGE_{stage}.done').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
        print(f'CORSTA_BOOTSTRAP_PASS stage={stage}',flush=True)


def preflight(a,cfg,p,device):
    if not p.backbone_checkpoint.is_file():raise FileNotFoundError(p.backbone_checkpoint)
    from corsta.stla import load_pretrained_stla,MaskAwareSTLAformer
    from corsta.student import SplitSTLA
    base,ck=load_pretrained_stla(Path(p.root),p.backbone_checkpoint,device)
    original=MaskAwareSTLAformer(base)
    from corsta.stla import missing_age
    adapted=SplitSTLA(base,int(cfg['model']['stla_tail_layers']))
    adapted.mask_proj.load_state_dict(original.mask_proj.state_dict())
    original.eval();adapted.eval()
    B=2;H=p.history;N=p.nodes
    x=torch.randn(B,H,N,device=device);m=(torch.rand(B,H,N,device=device)>.4).float()
    t=torch.randint(0,288,(B,H),device=device)
    d=torch.randint(0,7,(B,H),device=device)
    with torch.no_grad():
        # Both paths use properly zeroed missing inputs for a fair equivalence test.
        visible=x*m
        y1,h1=original(visible,m,t,d);y2,h2=adapted(visible,m,t,d)
        diff=(y1-y2).abs().max().item()
    if diff>1e-5:raise RuntimeError(f'STLA split breaks checkpoint equivalence: max_diff={diff}')
    print(f'CORSTA_PREFLIGHT_PASS stla_equivalence_max_abs={diff:.3g} '
          f'nodes={N} dim={adapted.model_dim} layers={len(base.attn_layers_st)}',flush=True)


def main():
    a=arguments()
    if a.cpu and not (a.smoke or a.stage=='preflight'):
        raise ValueError('--cpu is allowed only with --smoke or --stage preflight')
    if a.require_cuda and a.stage!='preflight':
        raise ValueError('--require-cuda is supported only with --stage preflight')
    if a.cpu and a.require_cuda:
        raise ValueError('--cpu and --require-cuda are mutually exclusive')
    cfg,root,run,p=compose(a)
    device=torch.device('cpu' if a.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if a.require_cuda and device.type!='cuda':
        raise RuntimeError('GPU Preflight requires CUDA but torch.cuda.is_available()=False; check GPU PBS allocation / CUDA environment')
    seed_everything(a.seed)

    # Preflight is CPU-compatible and does not write formal run identities.
    # --require-cuda explicitly enforces CUDA for optional GPU preflight.
    # Normal training still strictly requires a CUDA GPU below.
    if a.stage=='preflight':
        print(f'CORSTA_PREFLIGHT_START target={a.target} dataset={a.dataset} '
              f'device={device} checkpoint={p.backbone_checkpoint}',flush=True)
        if device.type=='cpu':
            print('CORSTA_PREFLIGHT_INFO device=cpu cuda_required=false '
                  'gpu_forward_backward_not_tested=true',flush=True)
        preflight(a,cfg,p,device)
        print('CORSTA_PREFLIGHT_COMPLETE result=pass',flush=True)
        return

    if device.type!='cuda' and not a.smoke and cfg['runtime'].get('require_cuda_formal',True):
        raise RuntimeError('Formal training requires CUDA. Submit CoRSTA/pbs/train.pbs '
                           'to a GPU compute node; login-node preflight works on CPU.')
    run.mkdir(parents=True,exist_ok=True)
    hash_id=identity(a,cfg,run,p)
    print(f'CORSTA_RUN_START stage={a.stage} target={a.target} dataset={a.dataset} '
          f'device={device} run={run} initial_batch={cfg["training"]["batch_initial"]} '
          f'candidates={cfg["training"]["batch_choices"]}',flush=True)
    selected=ORDER if a.stage=='all' else (a.stage,)
    for stage in selected:
        done=run/f'STAGE_{stage}.done'
        if done.exists():
            print(f'CORSTA_RESUME_SKIP stage={stage}',flush=True);continue
        print(f'CORSTA_STAGE_START stage={stage}',flush=True)
        if stage=='bootstrap':
            do_bootstrap(a,p)
        elif stage in ('warmup','joint'):
            if not (p.run/'best_c1.pt').is_file():raise RuntimeError('Run --stage bootstrap first')
            train_phase(p,cfg,device,run,stage,hash_id)
        elif stage=='lock':lock(p,cfg,device,run,hash_id)
        elif stage=='test':
            for sev in cfg['experiment']['severities']:
                target=run/'test'/f's{sev}.json'
                if target.exists():
                    print(f'CORSTA_TEST_REUSE severity=s{sev}',flush=True);continue
                test_severity(p,cfg,device,run,hash_id,int(sev))
        elif stage=='report':report(cfg,run)
        else:raise ValueError(stage)
        mandatory={'bootstrap':p.run/'best_c1.pt','warmup':run/'best_warmup.pt',
                   'joint':run/'best_joint.pt','lock':run/'model_lock.json','report':run/'summary.json'}
        if stage in mandatory and not mandatory[stage].is_file():raise RuntimeError(f'Stage incomplete {stage}')
        if stage=='test':
            for sev in cfg['experiment']['severities']:
                if not (run/'test'/f's{sev}.json').is_file():raise RuntimeError('Incomplete severity tests')
        done.write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
        print(f'CORSTA_STAGE_PASS stage={stage}',flush=True)
    print('CORSTA_PIPELINE_PASS',flush=True)

if __name__=='__main__':main()
