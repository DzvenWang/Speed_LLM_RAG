#!/usr/bin/env python3
"""TCRR C1+C2 v7.0.1 orchestrator: identity-checked stage subprocesses."""
from __future__ import annotations
import argparse,os,sys,subprocess,json,time,datetime,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from corsta.v7.timestamp import install_timestamps
install_timestamps()
import torch
from corsta.config import load_config,cache_path,file_sha256,resolve_stla_checkpoint
from corsta.v7.runner import (STAGES,seed_everything,s0_step,train_cache,fit_stage,
                                 locked_check,test_severity,report,atomic_json)
from corsta.data import SupportSplit


def parse():
    p=argparse.ArgumentParser()
    p.add_argument('--target',choices=('speed','flow'),required=True)
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--config',default=str(ROOT/'configs/bootstrap_v7.yaml'))
    p.add_argument('--run-dir')
    p.add_argument('--step',choices=STAGES)
    p.add_argument('--check-run-identity',action='store_true',
                   help='CPU-only manifest roundtrip; use before requesting a GPU allocation')
    p.add_argument('--smoke',action='store_true',help='small, separate smoke namespace; never used as formal')
    p.add_argument('--batch-size',type=int)
    p.add_argument('--workers',type=int)
    p.add_argument('--limit-train',type=int)
    p.add_argument('--limit-val',type=int)
    p.add_argument('--limit-test',type=int)
    return p.parse_args()


class Params:
    def __init__(self,a):
        os.environ['TRAFFIDENT_TARGET']=a.target
        self.config=load_config(a.config)
        if self.config['version']!='7.0.1':raise RuntimeError('Expected v7.0.1 config')
        self.root=Path(os.environ.get('SPEED_LLM_RAG_ROOT',ROOT.parent)).resolve()
        self.target=a.target;self.dataset=a.dataset;self.seed=a.seed
        self.config['experiment'].update(target=a.target,dataset=a.dataset,seed=a.seed)
        self.backbone_checkpoint=resolve_stla_checkpoint(self.config,a.backbone_checkpoint)
        self.backbone_mode=self.config['experiment']['backbone_mode']
        ck=torch.load(self.backbone_checkpoint,map_location='cpu',weights_only=False)
        if str(ck.get('model'))!='STLAformer':raise ValueError('Not a STLAformer checkpoint')
        if 'target' in ck and str(ck['target'])!=a.target:raise RuntimeError('Backbone target mismatch')
        if 'dataset' in ck and str(ck['dataset'])!=a.dataset:raise RuntimeError('Backbone dataset mismatch')
        mc=ck['model_config'];self.model_dim=int(mc.get('input_embedding_dim',32))+int(mc.get('tow_embedding_dim',32))
        self.nodes=int(ck['num_nodes']);self.history=12;self.horizon=12
        t=self.config['training'];m=self.config['model'];self.hidden=int(m['hidden'])
        self.graph_neighbours=int(m['graph_neighbors']);self.road_k=int(m.get('road_neighbors',4));self.graph_mode=str(m.get('graph_mode','dual'));self.memory_bank_device=str(t.get('memory_bank_device','auto'));self.k=int(m['top_k']);self.temperature=float(m['retrieval_temperature'])
        self.bound=float(m['max_normalized_correction']);self.groups=2
        self.epochs={name:int(t[name+'_epochs']) for name in ('s0','normal','c1','c2','joint_c12')}
        self.batch_size=a.batch_size or int(t['batch_size']);self.eval_batch_size=int(t['eval_batch_size'])
        self.teacher_batch=int(t['teacher_batch_size']);self.workers=t['workers'] if a.workers is None else a.workers
        self.s0_lr=float(t['s0_learning_rate']);self.learning_rate=float(t['learning_rate'])
        self.grad_accum=int(t['grad_accum']);self.amp=bool(t['amp'])
        self.retrieval_periods=tuple(int(p) for p in self.config['retrieval']['periods'])
        self.retrieval_shortlist=int(self.config['retrieval']['shortlist_per_period'])
        self.retrieval_history_weight=float(self.config['retrieval']['history_similarity_weight'])
        self.retrieval_forecast_weight=float(self.config['retrieval']['forecast_similarity_weight'])
        self.support_control=str(m.get('support_control','full'))
        self.horizon_graph=bool(m.get('horizon_graph',True))
        self.reliability_graph=bool(m.get('reliability_graph',True))
        self.replay_chunk=int(t.get('replay_chunk_queries',32));self.replay_batch=int(t.get('replay_teacher_batch',16));self.response_weight=float(t['response_weight']);self.patience=int(t['patience']);self.smoke=bool(a.smoke)
        self.severities=tuple(int(s) for s in self.config['experiment']['severities'])
        self.limits={'train':a.limit_train,'val':a.limit_val,'test':a.limit_test}
        if a.smoke:
            self.limits={k:(v if v is not None else (32 if k=='train' else 24)) for k,v in self.limits.items()}
            self.epochs={key:1 for key in self.epochs};self.workers=0
            self.batch_size=min(8,self.batch_size);self.teacher_batch=min(8,self.teacher_batch)
            self.k=min(4,self.k);self.retrieval_shortlist=min(8,self.retrieval_shortlist);self.replay_chunk=8;self.replay_batch=4;self.patience=1
        self.run=Path(a.run_dir).resolve() if a.run_dir else self.root/'CoRSTA_exps'/a.target/self.backbone_mode/a.dataset/'bootstrap_v7'/('smoke' if self.smoke else 'formal')/f'seed{a.seed}'
        self.run.mkdir(parents=True,exist_ok=True)
        self.public_config={'training':dict(self.config['training']),'model':dict(self.config['model']),'experiment':dict(self.config['experiment'])}
        if not self.smoke and self.batch_size>64:raise ValueError('Train batch >64 requires separate validated GPU smoke')
        if min(self.batch_size,self.eval_batch_size,self.teacher_batch,self.grad_accum)<1:raise ValueError('Batch sizes must be positive')
        if self.workers<0 or self.workers>8:raise ValueError('Workers must be 0..8')
        if self.graph_mode not in ('dual','directional','functional'):raise ValueError('Invalid graph_mode')
        if self.road_k<1 or self.road_k>16:raise ValueError('Invalid road_neighbors')
    def limit(self,split):return self.limits[split]


def one_step(params,step):
    seed_everything(params.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not params.smoke and device.type!='cuda' and not params.config['runtime'].get('allow_cpu_formal',False):
        raise RuntimeError('Formal GPU run without CUDA; use --smoke for local CPU validation')
    print(f'V7_STAGE_START stage={step} device={device} run={params.run}',flush=True)
    if step=='s0':
        p=cache_path(params.config,'formal');train=SupportSplit(p,'train',params.limit('train'))
        val=SupportSplit(p,'val',params.limit('val'))
        if train.x_obs.shape[2]!=params.nodes or val.x_obs.shape[2]!=params.nodes:raise RuntimeError('Data/Backbone node count mismatch')
        from corsta.v7.runner import s0_step
        s0_step(params,train,val,device)
    elif step=='cache':
        for split in ('train','val'):
            train_cache(params,device,split)
    elif step=='replay':
        from corsta.v7.replay import prepare_replay
        for sp in ('train','val'):prepare_replay(params,device,sp)
    elif step in ('normal','c1','c2','joint_c12'):
        fit_stage(params,step,device)
    elif step=='lock':locked_check(params,device)
    elif step=='test':
        for s in params.severities:
            path=params.run/'test'/f's{s}.json'
            if path.exists():
                print(f'V7_TEST_REUSE severity=s{s}',flush=True);continue
            test_severity(params,device,s)
    elif step=='report':report(params)
    else:raise ValueError(step)
    print(f'V7_STAGE_PASS stage={step}',flush=True)


def code_digest():
    files=list((ROOT/'corsta/v7').glob('*.py'))
    files += [ROOT/'corsta'/name for name in ('data.py','stla.py','config.py')]
    files += [ROOT/'scripts/bootstrap_v7.py', ROOT/'configs/bootstrap_v7.yaml',
              ROOT/'pbs/runtime.sh']
    h=hashlib.sha256()
    for f in sorted(files):
        h.update(str(f.relative_to(ROOT)).encode());h.update(f.read_bytes())
    return h.hexdigest()

def canonical_json(value):
    """Round-trip via JSON before comparing in-memory identities to on-disk JSON.

    In particular, Python tuple[4,2,1] becomes JSON list[4,2,1].
    Both parent/child stage manifests and replay manifests must compare the
    same JSON values, never Python tuples to loaded JSON lists.
    """
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def manifest_difference(existing, expected):
    return [k for k in sorted(set(existing)|set(expected)) if existing.get(k)!=expected.get(k)]


def main():
    args=parse();p=Params(args)
    manifest=p.run/'run_identity.json'
    expected={'code_sha256':code_digest(),'version':'7.0.1','target':p.target,'dataset':p.dataset,'seed':p.seed,
              'smoke':p.smoke,'backbone_sha256':file_sha256(p.backbone_checkpoint),
              'configuration':p.public_config,'limits':p.limits,'groups':p.groups,
              'effective_runtime':{'support_control':p.support_control,'horizon_graph':p.horizon_graph,'reliability_graph':p.reliability_graph,'retrieval_periods':p.retrieval_periods,'retrieval_shortlist':p.retrieval_shortlist,'retrieval_history_weight':p.retrieval_history_weight,'retrieval_forecast_weight':p.retrieval_forecast_weight,'replay_chunk':p.replay_chunk,'replay_batch':p.replay_batch,'batch_size':p.batch_size,'eval_batch_size':p.eval_batch_size,
                                   'teacher_batch_size':p.teacher_batch,'workers':p.workers,
                                   'epochs':p.epochs,'grad_accum':p.grad_accum,'k':p.k,'graph_mode':p.graph_mode,'road_k':p.road_k,'memory_bank_device':p.memory_bank_device}}
    expected=canonical_json(expected)
    if manifest.exists():
        actual=canonical_json(json.loads(manifest.read_text()))
        if actual!=expected:
            differing=manifest_difference(actual,expected)
            raise RuntimeError(
                f'Run directory identity mismatch at {manifest}; differing fields: {differing}. '
                'Use a fresh run directory for different source/config/code. '
                'Do not remove a manifest from a directory containing checkpoints.')
    else:
        # A lone manifest is not sufficient to certify a half-initialized run.
        # Do not claim an orphaned directory with previous artifacts.
        if any(p.run.iterdir()):
            raise RuntimeError(f'Run directory has artifacts but no identity manifest: {p.run}')
        atomic_json(expected,manifest)
    if args.check_run_identity:
        print('V7_RUN_IDENTITY_PASS '+str(manifest),flush=True)
        return
    if args.step:
        one_step(p,args.step);return
    for stage in STAGES:
        done=p.run/f'STAGE_{stage}.done'
        if done.exists():
            required_done={'s0':[p.run/'s0.pt'],'normal':[p.run/'best_normal.pt'],
                'c1':[p.run/'best_c1.pt'],'c2':[p.run/'best_c2.pt'],'joint_c12':[p.run/'best_joint_c12.pt'],
                'lock':[p.run/'model_lock.json'],
                'test':[p.run/'test'/f's{s}.json' for s in p.severities],
                'report':[p.run/'summary.json']}.get(stage,[])
            if any(not f.exists() or f.stat().st_size==0 for f in required_done):
                raise RuntimeError(f'Stale DONE marker for {stage}: required artifact missing')
            print(f'V7_RESUME_SKIP stage={stage}',flush=True);continue
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--target',p.target,
             '--dataset',p.dataset,'--seed',str(p.seed),'--backbone-checkpoint',str(p.backbone_checkpoint),
             '--config',args.config,'--run-dir',str(p.run),'--step',stage,
             '--batch-size',str(p.batch_size),'--workers',str(p.workers)]
        if p.smoke:cmd.append('--smoke')
        for k,flag in (('train','--limit-train'),('val','--limit-val'),('test','--limit-test')):
            if p.limit(k) is not None:cmd += [flag,str(p.limit(k))]
        subprocess.run(cmd,check=True,cwd=str(p.root))
        # Critical stage output checks before declaring success.
        required={'s0':[p.run/'s0.pt'],'normal':[p.run/'best_normal.pt'],
                  'c1':[p.run/'best_c1.pt'],'c2':[p.run/'best_c2.pt'],'joint_c12':[p.run/'best_joint_c12.pt'],
                  'lock':[p.run/'model_lock.json'],
                  'test':[p.run/'test'/f's{s}.json' for s in p.severities],
                  'report':[p.run/'summary.json']}.get(stage,[])
        if any(not f.exists() or f.stat().st_size==0 for f in required):
            raise RuntimeError(f'Stage {stage} finished without complete required artifacts')
        done.write_text(datetime.datetime.now().isoformat()+'\n')
    print('V7_PIPELINE_PASS',flush=True)


if __name__=='__main__':main()
