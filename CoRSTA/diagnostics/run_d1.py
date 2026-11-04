#!/usr/bin/env python3
"""One-shot Selection-only CoRSTA diagnostic; one PBS per target.

Requires completed v0.1.0 run and immutable Teacher/Replay/Checkpoints.
Does NOT train, create Replay, evaluate TEST, or edit prior artifacts.
"""
from __future__ import annotations
import argparse,contextlib,csv,datetime,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from corsta.observability import install_timestamps
install_timestamps()
import numpy as np
import torch
from corsta.config import file_sha256
from corsta.v7.runner import seed_everything,loader,as_device,selection_indices
from corsta.training import build_model,read_dataset
from corsta.batching import sample_batch,smaller_size,is_oom
from scripts.run import compose
from diagnostics.metrics import MaskedMetric,SampleDistribution,PairwiseGain,query_mae,query_oracle,element_oracle,gain_percent
from diagnostics.candidate import CandidateAudit
from diagnostics.interventions import MODES,intervene,FusionCapture,swap_student
from diagnostics.gradients import GradientAudit,one_batch

DEFAULT_BACKBONES={
    'speed':'Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/20260930_215233_seed10/best.pt',
    'flow':'Backbone_exps/flow/STLAformer/TraffiDent_Net_06/formal/20260930_220624_seed10/best.pt',
}


def cli():
    p=argparse.ArgumentParser(description='Selection-only one-shot CoRSTA-D1; never trains or evaluates TEST')
    p.add_argument('--target',required=True,choices=('speed','flow'))
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--backbone-checkpoint',required=True)
    p.add_argument('--run-dir',help='Existing CoRSTA v0.1.0 formal run directory')
    p.add_argument('--output-dir',help='Separate diagnostic directory (must not be inside existing experiment run)')
    p.add_argument('--config',default=str(ROOT/'configs/corsta_v0_1.yaml'))
    p.add_argument('--bootstrap-config',default=str(ROOT/'configs/bootstrap_v7.yaml'))
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--batch-initial',type=int,choices=(16,32,64,128),default=64)
    p.add_argument('--grad-batch',type=int,default=8)
    p.add_argument('--grad-queries',type=int,default=128)
    p.add_argument('--min-candidate-coverage',type=float,default=0.50)
    p.add_argument('--max-selection',type=int,default=None,help='For development ONLY: marks outputs partial, not formal diagnosis')
    p.add_argument('--check-only',action='store_true',help='CPU check of artifacts, no CUDA and no model forward')
    p.add_argument('--device',choices=('cuda','cpu','auto'),default='cuda',help='Full D1 diagnostic device; cuda remains the default')
    p.add_argument('--cpu-threads',type=int,default=0,help='CPU intra-op threads; 0 uses OMP_NUM_THREADS')
    return p.parse_args()


def same_hash_manifest(path):
    raw=json.loads(Path(path).read_text())
    return hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest(),raw


def verify(cfg,run,p,a):
    # Strict file checks intentionally precede graph_and_memory(): it may create missing files.
    if not run.is_dir():raise RuntimeError(f'Previous run not found: {run}')
    expect_run=run/'run_identity.json'
    if not expect_run.is_file():raise RuntimeError('Missing original run_identity.json')
    cfg_hash,old=same_hash_manifest(expect_run)
    for k,v in (('target',a.target),('dataset',a.dataset),('seed',a.seed)):
        if old.get(k)!=v:raise RuntimeError(f'Original run identity mismatches {k}: {old.get(k)} vs {v}')
    if old.get('smoke') is not False:raise RuntimeError('Diagnostic requires formal, non-smoke experiment')
    if old.get('backbone_mode')!=p.backbone_mode:raise RuntimeError('Backbone mode mismatch')
    if old.get('backbone_sha256')!=file_sha256(p.backbone_checkpoint):raise RuntimeError('Backbone checkpoint hash mismatch')
    if old.get('cfg')!=cfg:raise RuntimeError('Diagnostic config does not match completed training config')
    needed=[run/'best_warmup.pt',run/'best_joint.pt',run/'last_joint.pt',run/'model_lock.json',
        p.run/'best_c1.pt',p.run/'s0.pt',p.run/'train_only_neighbours.pt',
        p.run/'paired_replay'/'train'/'identity.json',p.run/'paired_replay'/'val'/'identity.json',
        p.run/'teacher_cache'/'train'/'train_base.npy',p.run/'teacher_cache'/'val'/'val_base.npy',
        p.run/'teacher_cache'/'train'/'train_latent.npy',p.run/'teacher_cache'/'val'/'val_latent.npy',
        p.run/'teacher_cache'/'train'/'train_c2.npy',p.run/'teacher_cache'/'train'/'train_normal.npy',
        p.run/'teacher_cache'/'train'/'train_cf.npy',p.run/'teacher_cache'/'train'/'train_DONE',
        p.run/'teacher_cache'/'val'/'val_DONE',
        p.run/'teacher_cache'/'train'/'train_identity.json',
        p.run/'teacher_cache'/'val'/'val_identity.json',
        p.run/'graph'/'graph_identity.json',p.run/'graph'/'graph_context.npz',
        p.run/'counterfactual_target_strength.json']
    missing=[str(f) for f in needed if not f.is_file()]
    if missing:raise FileNotFoundError('Missing existing artifacts. Diagnostic WILL NOT regenerate them:\n'+'\n'.join(missing))
    lock=json.loads((run/'model_lock.json').read_text())
    if lock['model_sha256']!=file_sha256(run/'best_joint.pt') or lock['config_sha256']!=cfg_hash:
        raise RuntimeError('Locked model checksum/identity mismatch')
    checks={}
    for name in ('best_warmup.pt','best_joint.pt','last_joint.pt'):
        ck=torch.load(run/name,map_location='cpu',weights_only=False)
        if ck.get('config_sha256')!=cfg_hash:raise RuntimeError(f'Checkpoint identity mismatch: {name}')
        checks[name]={'sha256':file_sha256(run/name),'phase':ck.get('phase'),'epoch':ck.get('epoch'),
                      'best_epoch':ck.get('best_epoch'),'best_selection_mae':ck.get('best_selection_mae',ck.get('best'))}
        del ck
    ck=torch.load(p.run/'best_c1.pt',map_location='cpu',weights_only=False)
    if ck.get('backbone_sha256')!=file_sha256(p.backbone_checkpoint):raise RuntimeError('Legacy C1 checkpoint mismatch')
    checks['best_c1.pt']={'sha256':file_sha256(p.run/'best_c1.pt'),'phase':ck.get('phase'),'epoch':ck.get('epoch')}
    return {'run':str(run),'target':a.target,'dataset':a.dataset,'seed':a.seed,'source_run_identity_sha256':file_sha256(expect_run),
            'source_experiment_config_sha256':cfg_hash,'original_source_code_sha256':old.get('code_sha256'),
            'backbone_sha256':old['backbone_sha256'],'backbone_mode':p.backbone_mode,'checkpoints':checks,
            'selection_only':True,'no_test_access':True,'no_training':True}


def load_ck_model(args,cfg,device,outdir,ckname):
    model,bc=build_model(args,cfg,device)
    ck=torch.load(outdir/ckname,map_location='cpu',weights_only=False)
    model.load_state_dict(ck['model'],strict=True)
    model.eval()
    return model,bc


def choose_batch(model,dataset,device,initial):
    """Inference-only probe: 64 first, optionally 128, else 32 then 16."""
    if device.type!='cuda':
        chosen=min(16,initial)
        print(f'CORSTA_D1_BATCH_SELECTED batch={chosen} device=cpu reason=conservative_memory',flush=True)
        return chosen,[{'size':chosen,'reason':'CPU conservative fixed batch; no CUDA OOM probe'}]
    attempts=[]
    def check(size):
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
            b=sample_batch(dataset,size,device)
            with torch.inference_mode():o=model(b)
            torch.cuda.synchronize(device)
            peak=torch.cuda.max_memory_allocated(device)/1024**3
            total=torch.cuda.get_device_properties(device).total_memory/1024**3
            ok=peak<total*.80
            row={'size':size,'ok':ok,'peak_gib':round(peak,3),'capacity_gib':round(total,3)}
            attempts.append(row)
            print(f'CORSTA_D1_BATCH_PROBE batch={size} ok={ok} peak_gib={peak:.3f}',flush=True)
            del b,o
            return ok,peak,total
        except (RuntimeError,MemoryError) as exc:
            if not is_oom(exc):raise
            attempts.append({'size':size,'ok':False,'reason':'oom'})
            print(f'CORSTA_D1_BATCH_PROBE batch={size} ok=false reason=oom',flush=True)
            return False,None,None
        finally:torch.cuda.empty_cache()
    ok,peak,total=check(initial)
    if ok:
        selected=initial
        if initial==64 and peak<total*.50:
            higher,_,_=check(128)
            if higher:selected=128
        return selected,attempts
    for size in sorted((x for x in (16,32,64,128) if x<initial),reverse=True):
        fits,_,_=check(size)
        if fits:return size,attempts
    raise RuntimeError('D1 no safe inference batch in {16,32,64,128}')


def model_base_prediction(legacy,b):
    return legacy.forward_progressive(
        b['cached_base'],b['cached_latent'],b['x'],b['mask'],
        b['candidate_normal'],b['candidate_cf'],
        candidate_scores=b['candidate_scores'],candidate_coverage=b['candidate_coverage'],
        candidate_valid=b['candidate_valid'],candidate_support_raw=b['candidate_support_raw'],max_stage=1)


def selection_pass(args,model,joint,legacy,bc,dataset,indices,device,batch,outdir,coverage):
    mean=float(bc['mean']);std=float(bc['std'])
    mode_metrics={k:MaskedMetric() for k in ('teacher_backbone','bootstrap_c1','warmup_backbone',
        'warmup_c1_posthoc','warmup_refined','last_joint_backbone','last_joint_refined',
        'joint_student_warm_rest','warm_student_joint_rest',
        'oracle_query','oracle_element')}
    for mode in MODES[1:]:mode_metrics['intervention_'+mode]=MaskedMetric()
    blended={str(x):MaskedMetric() for x in (0,.25,.50,.75,1.)}
    additive={str(x):MaskedMetric() for x in (0,.25,.50,.75,1.)}
    pair=PairwiseGain();candidate=CandidateAudit(coverage)
    injection={k:SampleDistribution(seed=60+i) for i,k in enumerate(('injection_c1','injection_c2','gate_c1','gate_c2','reliability_c1','reliability_c2'))}
    graph_mixes=[SampleDistribution(seed=72+i) for i in range(3)]
    stability={'no_feedback_max_abs_norm':0.,'query_count':0,'valid_count':0}
    horizon={name:[MaskedMetric() for _ in range(12)] for name in ('teacher_backbone','bootstrap_c1','warmup_refined')}
    fname=outdir/'selection_query_metrics.csv.partial'
    f=fname.open('w',newline='',encoding='utf-8')
    writer=csv.DictWriter(f,fieldnames=['query_index','window_start','valid_elements','observation_rate',
        'teacher_backbone_mae','bootstrap_c1_mae','warmup_c1_mae','warmup_backbone_mae','warmup_refined_mae',
        'last_joint_refined_mae','query_oracle_mae','c1_refined_delta_mae'])
    writer.writeheader()
    start=time.monotonic()
    try:
        dl=loader(dataset,batch,int(args.workers),subset=indices)
        for nb,raw in enumerate(dl,1):
            b=as_device(raw,device);y=b['y_raw'].float();valid=b['future_valid'].bool()
            with torch.inference_mode():
                saved={}
                def tr_hook(module,inputs,output):saved['transport']=output
                with legacy.transport.register_forward_hook(tr_hook):legacy_out=model_base_prediction(legacy,b)
                with FusionCapture(model) as capture:
                    warm=model(b)
                    details=capture.get()
                changed=joint(b)
                with swap_student(model,joint.student):
                    joint_student_warm_rest=model(b)['refined']
                with swap_student(joint,model.student):
                    warm_student_joint_rest=joint(b)['refined']
                outputs={'teacher_backbone':legacy_out['backbone'],'bootstrap_c1':legacy_out['c1'],
                    'warmup_backbone':warm['backbone'],'warmup_c1_posthoc':warm['c1_posthoc'],
                    'warmup_refined':warm['refined'],'last_joint_backbone':changed['backbone'],
                    'last_joint_refined':changed['refined'],
                    'joint_student_warm_rest':joint_student_warm_rest,
                    'warm_student_joint_rest':warm_student_joint_rest}
                for mode in MODES[1:]:
                    with intervene(model,mode):outputs['intervention_'+mode]=model(b)['refined']
                y_n=y;pred={k:v.float()*std+mean for k,v in outputs.items()}
                oracle_q,choice=query_oracle([pred['teacher_backbone'],pred['bootstrap_c1'],pred['warmup_refined']],y_n,valid)
                oracle_e,choice_e=element_oracle([pred['teacher_backbone'],pred['bootstrap_c1'],pred['warmup_refined']],y_n)
                pred.update(oracle_query=oracle_q,oracle_element=oracle_e)
                for name,prediction in pred.items():mode_metrics[name].add(prediction,y,valid)
                pair.add(pred['teacher_backbone'],pred['bootstrap_c1'],pred['warmup_refined'],y,valid)
                for alpha in (0,.25,.50,.75,1.):
                    blend=(1-alpha)*pred['bootstrap_c1']+alpha*pred['warmup_refined']
                    inc=pred['bootstrap_c1']+alpha*(pred['warmup_refined']-pred['warmup_backbone'])
                    blended[str(alpha)].add(blend,y,valid);additive[str(alpha)].add(inc,y,valid)
                candidate.add(b,legacy_out,saved['transport'],mean,std)
                for key in injection:injection[key].add(details[key])
                for k in range(3):graph_mixes[k].add(details['graph_mix'][...,k])
                stability['no_feedback_max_abs_norm']=max(stability['no_feedback_max_abs_norm'],
                    float((warm['backbone']-outputs['intervention_no_feedback']).abs().max()))
                stability['query_count']+=len(y);stability['valid_count']+=int(valid.sum())
                for name in horizon:
                    for hp in range(min(y.shape[1],12)):
                        horizon[name][hp].add(pred[name][:,hp:hp+1],y[:,hp:hp+1],valid[:,hp:hp+1])
                qs={key:query_mae(value,y,valid)[0].cpu().numpy() for key,value in pred.items() if key in (
                    'teacher_backbone','bootstrap_c1','warmup_c1_posthoc','warmup_backbone','warmup_refined','last_joint_refined','oracle_query')}
                counts=query_mae(pred['teacher_backbone'],y,valid)[1].cpu().numpy()
                indices_b=b['index'].detach().cpu().tolist();times=b['window_start'].detach().cpu().tolist();obs=b['mask'].mean((1,2)).cpu().tolist()
                for i,idx in enumerate(indices_b):
                    writer.writerow({'query_index':int(idx),'window_start':int(times[i]),'valid_elements':int(counts[i]),
                        'observation_rate':float(obs[i]),'teacher_backbone_mae':float(qs['teacher_backbone'][i]),
                        'bootstrap_c1_mae':float(qs['bootstrap_c1'][i]),'warmup_c1_mae':float(qs['warmup_c1_posthoc'][i]),
                        'warmup_backbone_mae':float(qs['warmup_backbone'][i]),'warmup_refined_mae':float(qs['warmup_refined'][i]),
                        'last_joint_refined_mae':float(qs['last_joint_refined'][i]),'query_oracle_mae':float(qs['oracle_query'][i]),
                        'c1_refined_delta_mae':float(qs['bootstrap_c1'][i]-qs['warmup_refined'][i])})
            if nb%20==0:print(f'CORSTA_D1_SELECTION_PROGRESS batches={nb} processed={stability["query_count"]}/{len(indices)} elapsed_s={time.monotonic()-start:.1f}',flush=True)
        f.flush();os.fsync(f.fileno())
    finally:f.close()
    fname.rename(outdir/'selection_query_metrics.csv')
    result={k:v.to_dict() for k,v in mode_metrics.items()}
    baseline=result['teacher_backbone']['mae'];gains={k:gain_percent(baseline,v['mae']) for k,v in result.items()}
    bestblend=min(blended.items(),key=lambda kv:kv[1].to_dict()['mae'])
    bestinc=min(additive.items(),key=lambda kv:kv[1].to_dict()['mae'])
    return {'selection_queries':stability['query_count'],'metrics':result,'gain_vs_teacher_percent':gains,
        'target_5percent_mae':baseline*.95,'pairwise':pair.to_dict(),
        'blend_mae':{k:v.to_dict() for k,v in blended.items()},'incremental_mae':{k:v.to_dict() for k,v in additive.items()},
        'best_blend_alpha_diagnostic_only':bestblend[0],'best_increment_lambda_diagnostic_only':bestinc[0],
        'candidate':candidate.to_dict(),
        'fusion':{**{k:v.to_dict() for k,v in injection.items()},
                  'graph_mixture':{name:graph_mixes[i].to_dict() for i,name in enumerate(('functional','upstream','downstream'))}},
        'horizon_mae':{k:[x.to_dict() for x in v] for k,v in horizon.items()},
        'no_feedback_backbone_equivalence_max_abs_norm':stability['no_feedback_max_abs_norm'],
        'note':'All Oracle and grid calculations use Selection future labels for offline diagnostic only; NEVER inference routing.'}


def parameter_drift(warm,joint):
    result={}
    base=warm.state_dict();after=joint.state_dict()
    prefix_count=len(warm.student.base.attn_layers_st)-warm.student.tail_layers
    groups={
        'student_prefix':lambda n:n.startswith('student.base.attn_layers_st.') and int(n.split('.')[3])<prefix_count,
        'student_tail':lambda n:n.startswith('student.base.attn_layers_st.') and int(n.split('.')[3])>=prefix_count,
        'student_output':lambda n:n.startswith('student.base.output_proj.'),
        'student_mask_proj':lambda n:n.startswith('student.mask_proj.'),
        'c1':lambda n:n.startswith('c1.'),'c2':lambda n:n.startswith('c2.'),
        'bridges_support':lambda n:n.startswith(('bridge1.','bridge2.','support.','fusion.')),
    }
    for group,belongs in groups.items():
        diff2=ref2=0.;count=0
        for name,v in base.items():
            if name not in after or not belongs(name) or not v.is_floating_point():continue
            x=v.detach().float();y=after[name].detach().float();diff2+=float((y-x).square().sum());ref2+=float(x.square().sum());count+=x.numel()
        result[group]={'relative_l2_drift':math.sqrt(diff2)/max(math.sqrt(ref2),1e-12),'absolute_l2_drift':math.sqrt(diff2),'parameters_or_buffers':count}
    return result


def run_gradients(args,cfg,device,p,run,bc,samples,batch,grad_ck_name='best_warmup.pt'):
    out=GradientAudit()
    if samples<=0:return out.to_dict()
    model,bc=load_ck_model(p,cfg,device,run,grad_ck_name)
    model.configure_student('partial',int(cfg['model']['stla_tail_layers']))
    model.c1.freeze_stage('c1')
    model.eval() # deterministic forward, but gradients remain enabled
    ds,_=read_dataset(p,'train')
    idx=np.linspace(0,len(ds)-1,min(samples,len(ds)),dtype=np.int64).tolist()
    dl=loader(ds,batch,0,subset=idx)
    for i,raw in enumerate(dl,1):
        b=as_device(raw,device)
        with torch.enable_grad(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=(device.type=='cuda')):
            data=one_batch(model,b,float(bc['mean']),float(bc['std']))
        out.add(data);print(f'CORSTA_D1_GRAD_PROGRESS batch={i}/{len(dl)}',flush=True)
        model.zero_grad(set_to_none=True)
    return out.to_dict()


def atomic_json(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str,allow_nan=False),encoding='utf8')
    os.replace(tmp,path)


def recommend(summary,grad,drift):
    m=summary['metrics'];g=summary['gain_vs_teacher_percent']
    full=m['warmup_refined']['mae'];c1=m['bootstrap_c1']['mae']
    candidate=summary['candidate'];candidate_oracle=candidate['mae_by_mode']['candidate_oracle_query']['mae']
    base=m['teacher_backbone']['mae']
    recommendations=[]
    if g['oracle_query']>=5:
        recommendations.append({'priority':'P0','action':'preserve_bootstrap_C1_and_train_reliability_based_selector',
            'evidence':f'Existing query oracle gain {g["oracle_query"]:.2f}% reaches 5%.'})
    elif g['oracle_element']>=5:
        recommendations.append({'priority':'P0','action':'node_horizon_level_reliable_fusion',
            'evidence':f'Element oracle gain {g["oracle_element"]:.2f}% reaches 5%, query oracle does not.'})
    else:
        recommendations.append({'priority':'P0','action':'enhance_retrieval_and_failure_compatible_support_information',
            'evidence':'Existing fixed-predictor query/element oracle is below 5%; routing alone has insufficient measured headroom.'})
    if candidate_oracle is not None and gain_percent(base,candidate_oracle)>=5:
        recommendations.append({'priority':'P0','action':'candidate_aware_C1_utility_and_none_calibration',
            'evidence':'Legal TRAIN-candidate query oracle reaches 5% under the diagnostic coverage rule.'})
    elif candidate['utility_top1_hit_best_fraction']<.3 and candidate['eligible_query_fraction']>.2:
        recommendations.append({'priority':'P1','action':'fix_candidate_utility_ranking_before_expanding_architecture',
            'evidence':'Poor utility Top-1 agreement with best legal candidate.'})
    d2=m['intervention_no_c2_bridge']['mae']-full
    if d2<=0:recommendations.append({'priority':'P0','action':'rework_or_remove_C2_bridge_until_independent_gain_is_shown',
        'evidence':f'Disabling C2 bridge does not hurt Selection MAE (difference {d2:+.6f}).'})
    rho2=summary['fusion']['injection_c2'].get('mean',0)
    if rho2 is not None and rho2<.005:recommendations.append({'priority':'P1','action':'reduce_over_gating_and_review_C2_feature_scale',
        'evidence':f'C2 relative feature injection norm {rho2:.5f}, exploratory threshold 0.005.'})
    warm_grad=grad.get('warmup',{}).get('mean_metrics',{})
    cos=warm_grad.get('stla_tail',{}).get('cos_refined_base')
    if cos is not None and cos<-.1:recommendations.append({'priority':'P0','action':'separate_STLA_initial_and_refined_head_gradient_objectives',
        'evidence':f'STLA-tail refined/base gradient cosine {cos:+.3f} (negative).'} )
    elif drift.get('student_tail',{}).get('relative_l2_drift',0)>0:
        recommendations.append({'priority':'P1','action':'use_progressive_partial_unfreezing_and_stronger_student_protection',
            'evidence':'Joint checkpoint changes STLA parameters while Selection degrades; verify drift magnitude and loss scales.'})
    return {'selection_only':True,'target_gain_percent':5.,'heuristic_thresholds_not_statistical_tests':True,
            'key_comparisons':{'c1_vs_deep_mae_difference':full-c1,'c2_bridge_off_minus_full_mae':d2,
                'query_oracle_gain_percent':g['oracle_query'],'element_oracle_gain_percent':g['oracle_element'],
                'candidate_oracle_gain_percent':gain_percent(base,candidate_oracle)},
            'recommended_changes_in_priority_order':recommendations}

def markdown(a,summary,grad,metadata):
    metric=summary['metrics'];gain=summary['gain_vs_teacher_percent'];line=[
        f'# CoRSTA-D1 — {a.target.upper()} — Selection-only', '',
        f'- Dataset: {a.dataset}, seed {a.seed}',
        f'- Query count: {summary["selection_queries"]}',
        '- No TRAIN updates, no TEST reads, no source checkpoint edits.',
        '- All Oracle / grid choices use Selection labels and are **not deployable**.', '',
        '## Forecasting performance', '',
        '| Output | Selection MAE | Gain vs frozen Teacher |','|---|---:|---:|',
    ]
    for k in ('teacher_backbone','bootstrap_c1','warmup_c1_posthoc','warmup_refined','last_joint_refined','joint_student_warm_rest','warm_student_joint_rest','oracle_query','oracle_element'):
        line.append(f'| {k} | {metric[k]["mae"]:.6f} | {gain[k]:+.2f}% |')
    line += ['',f'5% target selection MAE relative to frozen Teacher: **{summary["target_5percent_mae"]:.6f}**',
        '', '## Distinguishing decisions', '',
        f'- Best diagnostic blend alpha: {summary["best_blend_alpha_diagnostic_only"]}',
        f'- Best diagnostic additive lambda: {summary["best_increment_lambda_diagnostic_only"]}',
        f'- Candidate query-level Oracle MAE: {summary["candidate"]["mae_by_mode"]["candidate_oracle_query"]["mae"]:.6f}',
        f'- Candidate top-1 hit fraction: {summary["candidate"]["utility_top1_hit_best_fraction"]:.3f}',
        f'- C1 injection relative norm: {summary["fusion"]["injection_c1"].get("mean")}',
        f'- C2 injection relative norm: {summary["fusion"]["injection_c2"].get("mean")}',
        '', '## Important caveats', '',
        '- Inference interventions are *not* retrained ablations.',
        '- Selection Oracle values are upper bounds for fixed predictors and do not measure novel information obtainable with retraining.',
        '- Candidate Oracle uses a minimum valid-candidate coverage and bound-preserving transport.',
        '- Gradients are one-point snapshots at fixed checkpoints; not historical gradients from earlier epochs.',
        '- This diagnostic alone does not prove a 5% TEST MAE improvement.', '',
        f'- Gradient batches per checkpoint: {grad.get("warmup",{}).get("num_batches",0)}',
        f'- Original code identity: {metadata["original_source_code_sha256"]}',
    ]
    return '\n'.join(line)+'\n'


def main():
    a=cli();seed_everything(a.seed)
    if a.grad_batch<1 or a.grad_batch>32:raise ValueError('grad-batch must be 1..32')
    if not 0<=a.grad_queries<=2048:raise ValueError('grad-queries must be 0..2048')
    if not 0<a.min_candidate_coverage<=1:raise ValueError('candidate coverage must be within (0,1]')
    diag_batch=a.batch_initial;diag_workers=a.workers
    a.stage='all';a.batch_mode=None;a.batch_initial=None;a.workers=None
    a.limit_train=a.limit_val=a.limit_test=None;a.smoke=False
    cfg,root,run,p=compose(a)
    a.batch_initial=diag_batch;a.workers=diag_workers
    diag=Path(a.output_dir).expanduser().resolve() if a.output_dir else root/'CoRSTA_diagnostics'/'D1_Net06_seed10'/a.target
    if diag==run or run in diag.parents:raise RuntimeError('Diagnostic output must not be under original experiment run')
    metadata=verify(cfg,run,p,a)
    print(f'CORSTA_D1_PREFLIGHT_PASS target={a.target} source={run} checkpoints=validated no_test_access=true',flush=True)
    if a.check_only:
        print('CORSTA_D1_CHECK_ONLY_PASS',flush=True);return
    if a.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('D1 requires CUDA by default; use --device cpu on a CPU PBS node')
    device=torch.device('cuda' if a.device=='cuda' or (a.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if device.type=='cpu':
        threads=a.cpu_threads or int(os.environ.get('OMP_NUM_THREADS','1'))
        if threads<1:raise ValueError('--cpu-threads must be a positive integer')
        torch.set_num_threads(threads)
        print(f'CORSTA_D1_CPU_MODE threads={torch.get_num_threads()} full_selection=true gradients=true',flush=True)
    else:
        print('CORSTA_D1_GPU_MODE device=cuda',flush=True)
    diag.mkdir(parents=True,exist_ok=True)
    if (diag/'D1_COMPLETE.json').exists():raise RuntimeError('D1 already complete; do not overwrite results; choose new --output-dir')
    atomic_json(metadata,diag/'source_manifest.json')
    model,bc=load_ck_model(p,cfg,device,run,'best_warmup.pt')
    joint,bc2=load_ck_model(p,cfg,device,run,'last_joint.pt')
    if bc2['mean']!=bc['mean'] or bc2['std']!=bc['std']:raise RuntimeError('Mismatch in raw unit scale between checkpoints')
    # Cloned from initial build_model() state before warmup ck loading, preserving
    # the original, successfully pretrained C1 stage without any re-training.
    from corsta.v7.runner import graph_and_memory,load_network
    _,graph,_=graph_and_memory(p,device)
    legacy=load_network(p,device,graph,bc,previous='c1');legacy.eval()
    val,raw=read_dataset(p,'val');selection,locked=selection_indices(raw)
    if a.max_selection is not None:
        if a.max_selection<1:raise ValueError('max-selection must be positive')
        selection=selection[:a.max_selection]
    metadata['selection_query_count']=len(selection);metadata['locked_count_excluded']=len(locked)
    metadata['partial_debug_selection']=a.max_selection is not None
    metadata['diagnostic_code_sha256']=file_sha256(Path(__file__))
    atomic_json(metadata,diag/'source_manifest.json')
    batch,attempts=choose_batch(model,val,device,a.batch_initial)
    atomic_json({'selected_batch':batch,'attempts':attempts,'initial':a.batch_initial},diag/'batch_probe.json')
    while True:
        try:
            t=time.monotonic()
            print(f'CORSTA_D1_SELECTION_START target={a.target} queries={len(selection)} batch={batch} interventions={len(MODES)-1}',flush=True)
            summary=selection_pass(a,model,joint,legacy,bc,val,selection,device,batch,diag,a.min_candidate_coverage)
            print(f'CORSTA_D1_SELECTION_PASS target={a.target} seconds={time.monotonic()-t:.2f}',flush=True)
            break
        except (RuntimeError,MemoryError) as exc:
            if not is_oom(exc):raise
            nxt=smaller_size(batch)
            if nxt is None:raise
            print(f'CORSTA_D1_RUNTIME_OOM batch={batch} fallback={nxt} replaying_selection_from_start=true',flush=True)
            batch=nxt
            if device.type=='cuda':torch.cuda.empty_cache()
    atomic_json(summary['metrics'],diag/'prediction_metrics.json')
    atomic_json({k:summary[k] for k in ('selection_queries','target_5percent_mae','gain_vs_teacher_percent','pairwise',
                                      'blend_mae','incremental_mae','best_blend_alpha_diagnostic_only','best_increment_lambda_diagnostic_only')},diag/'prediction_headroom.json')
    atomic_json(summary['candidate'],diag/'candidate_utility.json')
    atomic_json(summary['fusion'],diag/'fusion_statistics.json')
    atomic_json({k:summary['metrics'][k] for k in summary['metrics'] if k.startswith('intervention_')},diag/'c2_interventions.json')
    atomic_json(summary['horizon_mae'],diag/'horizon_mae.json')
    drift=parameter_drift(model,joint)
    atomic_json({'parameter_drift':drift,
        'selection_backbone_drift_mae':summary['metrics']['last_joint_backbone']['mae']-summary['metrics']['warmup_backbone']['mae'],
        'selection_refined_drift_mae':summary['metrics']['last_joint_refined']['mae']-summary['metrics']['warmup_refined']['mae']},diag/'student_drift.json')
    print(f'CORSTA_D1_GRAD_START target={a.target} train_queries={a.grad_queries} grad_batch={a.grad_batch}',flush=True)
    grad={'warmup':run_gradients(a,cfg,device,p,run,bc,a.grad_queries,a.grad_batch,'best_warmup.pt'),
          'last_joint':run_gradients(a,cfg,device,p,run,bc,a.grad_queries,a.grad_batch,'last_joint.pt')}
    atomic_json(grad,diag/'gradient_conflict.json')
    decision=recommend(summary,grad,drift)
    atomic_json(decision,diag/'architecture_decisions.json')
    (diag/'FINAL_DIAGNOSIS.md').write_text(markdown(a,summary,grad,metadata)+
        '\n## Proposed next-version decisions (heuristic)\n\n'+''.join('- '+x['priority']+': '+x['action']+' — '+x['evidence']+'\n' for x in decision['recommended_changes_in_priority_order']),encoding='utf8')
    atomic_json({'target':a.target,'dataset':a.dataset,'seed':a.seed,'partial_debug_selection':a.max_selection is not None,
        'selection_queries':summary['selection_queries'],'status':'PASS',
        'files':['prediction_metrics.json','prediction_headroom.json','candidate_utility.json','fusion_statistics.json',
                 'c2_interventions.json','horizon_mae.json','student_drift.json','gradient_conflict.json',
                 'source_manifest.json','selection_query_metrics.csv','architecture_decisions.json','FINAL_DIAGNOSIS.md']},diag/'D1_COMPLETE.json')
    print(f'CORSTA_D1_COMPLETE target={a.target} queries={len(selection)} output={diag} selection_only=true training=false',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        print('CORSTA_D1_FAILED',flush=True)
        traceback.print_exc()
        raise
