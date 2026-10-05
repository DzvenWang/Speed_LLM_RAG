#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from types import MethodType
import numpy as np,torch
from torch.utils.data import DataLoader, Subset

GADI_ROOT=next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
ROOT=GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baseline.utils import load_yaml
from baseline.runners.common import GinARDataset,ImputationDataset,RASTDataset,SARAFTensorDataset,build_time_marks,save_split_outputs,scale_stats,set_seed,require_gpu,simple_namespace
from baseline.runners.ginar_runner import import_model as ginar_import,official_doubletransition
from baseline.runners.imputeformer_runner import build_model as impute_build,forecast_val as impute_forecast
from baseline.runners.lrtc_tmcp_runner import forecast_split as lrtc_forecast
from baseline.runners.raft_runner import import_model as raft_import,make_protocol_retrieve
from baseline.runners.saraf_runner import import_model as saraf_import,overlap_masker,_probe_time_bonus
from baseline.runners.rast_runner import build_model as rast_build,load_store,restore_store,predict as rast_predict
from baseline.runners.sprint_runner import forecast_split as sprint_forecast
from baseline.runners.mogernn_runner import MoGERNNDataset,build_model as mogernn_build,predict as mogernn_predict
from baseline.runners.remft_runner import ReMFTDataset,build_model as remft_build,predict as remft_predict,load_trainable_state_dict as remft_load
from baseline.backbones.patchstg_bridge import FrozenPatchSTG
from baseline.backbones.frozen_backbone import build_frozen_backbone
from baseline.run_identity import load_cache_identity, require_same_identity
from baseline.shared_cache import cache_path
from baseline.backbones.frozen_registry import sha256_file, load_frozen_backbone_registry


def _validate_formal_run(cfg, m, seed, run_dir: Path):
    exp=Path(cfg['project']['exp_root'])
    expected=load_cache_identity(cache_path(cfg,'formal'),cfg['version'])
    d=Path(run_dir)
    sp,cp,mp=d/'summary.json',d/'best.pt',d/'RUN_MANIFEST.json'
    if not sp.exists(): raise FileNotFoundError(sp)
    if not cp.exists(): raise FileNotFoundError(cp)
    sm=json.loads(sp.read_text())
    # RUN_MANIFEST.json is the authoritative immutable Protocol/cache identity.
    # Older SARAF v1.1.0 summaries accidentally reused the key
    # `time_feature_source` for a SARAF-specific diagnostic string, which
    # overwrote the canonical identity field only in summary.json.  Validate
    # the manifest when present so those already-trained checkpoints can be
    # evaluated without weakening the shared identity contract.
    if mp.exists():
        manifest=json.loads(mp.read_text())
        require_same_identity(manifest,expected,label=str(mp))
        if str(manifest.get('method')) != str(m):
            raise RuntimeError(f'Formal run manifest method mismatch: expected={m} got={manifest.get("method")}')
        if int(manifest.get('seed',-1)) != int(seed):
            raise RuntimeError(f'Formal run manifest seed mismatch: expected={seed} got={manifest.get("seed")}')
        mh=manifest.get('run_identity_hash')
        sh=sm.get('run_identity_hash')
        if mh and sh and str(mh) != str(sh):
            raise RuntimeError(f'Formal run summary/manifest identity hash mismatch: summary={sh} manifest={mh}')
    else:
        require_same_identity(sm,expected,label=str(sp))
    if str(sm.get('method'))!=str(m):
        raise RuntimeError(f'Formal run method mismatch: expected={m} got={sm.get("method")}')
    if int(sm.get('seed',-1))!=int(seed):
        raise RuntimeError(f'Formal run seed mismatch: expected={seed} got={sm.get("seed")}')
    if str(sm.get('code_project_version',sm.get('project_version'))) != str(cfg['version']):
        raise RuntimeError(f'Formal run was not produced by Baseline v{cfg["version"]}')
    if str(sm.get('status',''))!='COMPLETE':
        raise RuntimeError('Formal run status is not COMPLETE')
    if not bool(sm.get('selection_valid',True)):
        raise RuntimeError('Formal run is marked invalid')
    score=float(sm.get('val',{}).get('overall_MAE',float('inf')))
    if not np.isfinite(score):
        raise RuntimeError('Formal run has non-finite validation MAE')

    if cfg.get('methods',{}).get(m,{}).get('requires_frozen_backbone_for_final_forecast',False):
        cm=json.loads((cache_path(cfg,'formal')/'cache_manifest.json').read_text())
        con=cfg.get('frozen_backbone_contract',{})
        fb=load_frozen_backbone_registry(
            cfg['project']['frozen_backbone_json'],
            expected_protocol_hash=cm['protocol_stage_hash'],
            expected_time_feature_revision=cm['time_feature_revision'],
            expected_time_feature_fingerprint=cm['time_feature_fingerprint'],
            expected_time_feature_source=cm['time_feature_source'],
            expected_dataset=cfg['experiment']['subgraph'],
            expected_target=cfg['experiment']['target'],
            expected_backbone_data_contract_revision=con.get('required_data_contract_revision','traffident_experiment_cache_v1'),
            expected_shared_contract_version=con.get('required_shared_contract_version','1.2.1'),
            expected_cache_schema_version=int(con.get('required_cache_schema_version',3)),
            expected_protocol_version=con.get('required_protocol_version','2.1.0'),
            expected_paired_group_id=con.get('required_paired_group_id'),
            expected_metrics_revision=cfg['shared_contract']['metrics_revision'],
            expected_prediction_contract_revision=cfg['shared_contract']['prediction_contract_revision'],
        )
        saved_model=sm.get('frozen_backbone_model')
        saved_checkpoint=sm.get('frozen_backbone_checkpoint')
        saved_seed=sm.get('frozen_backbone_reference_seed',-1)
        if str(saved_model)!=str(fb['selected_model']):
            raise RuntimeError(f'{m} seed={seed}: frozen Backbone model mismatch saved={saved_model} current={fb["selected_model"]}')
        if str(saved_checkpoint)!=str(fb['reference_checkpoint']):
            raise RuntimeError(f'{m} seed={seed}: frozen Backbone checkpoint mismatch')
        if int(saved_seed)!=int(fb['reference_seed']):
            raise RuntimeError(f'{m} seed={seed}: frozen Backbone reference seed mismatch')

    sel={
        'run_dir':str(d),
        'checkpoint':str(cp),
        'checkpoint_sha256':sha256_file(cp),
        'val_overall_MAE':score,
        'run_identity_hash':sm.get('run_identity_hash'),
        'selection_mode':'direct_formal_run_no_baseline_freeze',
    }
    return sel,torch.load(cp,map_location='cpu',weights_only=False)


def load_sel(cfg,m,seed,formal_run_dir=None):
    """Resolve the completed Formal run directly; no Baseline freeze artifact is required."""
    exp=Path(cfg['project']['exp_root'])
    if formal_run_dir:
        return _validate_formal_run(cfg,m,seed,Path(formal_run_dir))

    rr=exp/m/cfg['experiment']['subgraph']/'formal'
    latest=rr/'LATEST_RUN.txt'
    candidates=[]
    if latest.exists():
        raw=latest.read_text().strip()
        if raw:
            candidates.append(Path(raw))
    if rr.exists():
        candidates.extend(sorted(rr.glob(f'*_seed{seed}'),reverse=True))

    seen=set(); errors=[]
    for d in candidates:
        key=str(d)
        if key in seen: continue
        seen.add(key)
        try:
            return _validate_formal_run(cfg,m,seed,d)
        except Exception as e:
            errors.append(f'{d}: {type(e).__name__}: {e}')
    detail='\n'.join(errors[-5:])
    raise RuntimeError(f'No valid completed Formal run found for {m} seed={seed}.\n{detail}')


def ginar(cfg,fc,ec,seed,device,ck,max_rows=None):
    ds=GinARDataset(ec,'test'); ds=Subset(ds,range(min(len(ds),max_rows))) if max_rows else ds; dl=DataLoader(ds,batch_size=int(cfg['runners']['GinAR']['batch_size']),shuffle=False); adj=np.load(fc/'adjacency.npy'); M=ginar_import(); c=cfg['runners']['GinAR']; m=M(12,adj.shape[0],12,3,int(c['emb_size']),int(c['graph_size']),int(c['layers']),float(c['dropout']),official_doubletransition(adj,device)).to(device); m.load_state_dict(ck['model']); m.eval(); mean,std=scale_stats(fc); out=[]
    with torch.no_grad():
        for feat,_y,_v,_i in dl: out.append(m(feat.float().to(device)).cpu().numpy())
    return np.concatenate(out)*np.float32(std)+np.float32(mean)


def impute(cfg,fc,ec,seed,device,ck,max_rows=None):
    ds=ImputationDataset(ec,'test'); dl=DataLoader(ds,batch_size=int(cfg['runners']['ImputeFormer']['batch_size']),shuffle=False); m=impute_build(cfg,ds.x.shape[2]).to(device); m.load_state_dict(ck['model']); patch=build_frozen_backbone(cfg,device); return impute_forecast(m,dl,patch,device)


def lrtc_tmcp(cfg,fc,ec,seed,device,ck,max_rows=None):
    # LRTC-TMCP has no learned benchmark weights; best.pt is a provenance marker.
    expected=cfg.get('method_implementation_revisions',{}).get('LRTC-TMCP')
    if expected and str(ck.get('method_implementation_revision')) != str(expected):
        raise RuntimeError(
            f'LRTC-TMCP Formal run uses stale implementation revision '
            f'{ck.get("method_implementation_revision")!r}; expected {expected!r}. Rerun Formal.'
        )
    patch=build_frozen_backbone(cfg,device)
    raw,_info=lrtc_forecast(cfg,ec,'test',patch,device,smoke=False)
    return raw


def sprint(cfg,fc,ec,seed,device,ck):
    # best.pt is an immutable provenance/checkpoint marker; SPRINT itself has no learned benchmark weights.
    patch=FrozenPatchSTG(cfg,device); return sprint_forecast(cfg,ec,'test',patch,device,smoke=False)


def raft(cfg,fc,ec,seed,device,ck,max_rows=None):
    txm,tym,_=build_time_marks(cfg,fc,'train'); qxm,qym,_=build_time_marks(cfg,ec,'test'); mem=SARAFTensorDataset(fc,'train',txm,tym,'full'); q=SARAFTensorDataset(ec,'test',qxm,qym,'obs',limit=max_rows); c=cfg['runners']['RAFT']; Model=raft_import(); mc=simple_namespace(task_name='long_term_forecast',seq_len=12,pred_len=12,enc_in=388,n_period=int(c['n_period']),topm=int(c['topm']),gpu=0); m=Model(mc).to(device); rt=m.rt; rt.prepare_dataset(mem); span=24; rt.retrieve=MethodType(make_protocol_retrieve(mem.meta.window_start.astype(int).to_numpy(),mem.meta.window_start.astype(int).to_numpy(),q.meta.window_start.astype(int).to_numpy(),span),rt); va=rt.retrieve_all(q,train=False,device=device); m.retrieval_dict={'valid':va.to(device)}; del m.rt; m.load_state_dict(ck['model']); dl=DataLoader(q,batch_size=int(c['batch_size']),shuffle=False); mean,std=scale_stats(fc); out=[]; m.eval()
    with torch.no_grad():
        for idx,x,_y,_xm,_ym in dl: out.append(m(x.float().to(device),idx.long().to(device),mode='valid').cpu().numpy())
    return np.concatenate(out)*np.float32(std)+np.float32(mean)


def saraf(cfg,fc,ec,seed,device,ck,max_rows=None):
    txm,tym,_=build_time_marks(cfg,fc,'train'); qxm,qym,_=build_time_marks(cfg,ec,'test'); mem=SARAFTensorDataset(fc,'train',txm,tym,'full'); q=SARAFTensorDataset(ec,'test',qxm,qym,'obs',limit=max_rows); c=cfg['runners']['SARAF']; Model=saraf_import(); mc=simple_namespace(task_name='long_term_forecast',seq_len=12,pred_len=12,features=str(c.get('features','M')),enc_in=388,n_period=int(c.get('n_period',1)),topm=int(c.get('topm',20)),freq=str(c.get('freq','5min')),time_aware_weight=float(c.get('time_aware_weight',0.3)),fusion_method=str(c.get('fusion_method','weighted')),use_gpu=True,gpu=0); m=Model(mc).to(device); rt=m.rt; rt.steps_per_day=288; rt.steps_per_year=288*365; rt.prepare_dataset(mem); _probe_time_bonus(rt,q,device); span=24; rt.masked_sim=MethodType(overlap_masker(q.meta.window_start.astype(int).to_numpy(),mem.meta.window_start.astype(int).to_numpy(),span),rt); va=rt.retrieve_all(q,train=True,device=device); m.retrieval_temporal_dict={'valid':va.to(device)}; del m.rt; m.load_state_dict(ck['model']); dl=DataLoader(q,batch_size=int(c['batch_size']),shuffle=False); mean,std=scale_stats(fc); out=[]; m.eval()
    with torch.no_grad():
        for idx,x,_y,xm,_ym in dl: out.append(m(x.float().to(device),xm.float().to(device),idx.long().to(device),mode='valid').cpu().numpy())
    return np.concatenate(out)*np.float32(std)+np.float32(mean)


def rast(cfg,fc,ec,seed,device,ck,sel,max_rows=None):
    ds=RASTDataset(ec,'test'); ds=Subset(ds,range(min(len(ds),max_rows))) if max_rows else ds; dl=DataLoader(ds,batch_size=int(cfg['runners']['RAST']['batch_size']),shuffle=False); rd=Path(sel['run_dir']); m=rast_build(cfg,'native',seed,device,rd/'evaluation_database'); m.load_state_dict(ck['model']); restore_store(m,load_store(rd/'retrieval_store_best.npz')); mean,std=scale_stats(fc)
    raw, impact = rast_predict(m,dl,device,mean,std,int(ck['best_epoch']))
    if impact is not None:
        print('INFO RAST native evaluation returned impact diagnostics; predictions use tuple[0]', flush=True)
    return raw


def mogernn(cfg,fc,ec,seed,device,ck):
    ds=MoGERNNDataset(ec,'test'); dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0); m=mogernn_build(cfg,device); m.load_state_dict(ck['model']); adj=torch.from_numpy(np.asarray(np.load(fc/'adjacency.npy'),np.float32)).to(device); mean,std=scale_stats(fc); eps=float(cfg['runners']['MoGERNN'].get('epsilon_for_observed_zero',1e-6)); return mogernn_predict(m,dl,adj,device,eps,mean,std,int(ck.get('best_epoch',0)))


def remft(cfg,fc,ec,seed,device,ck,max_rows=None):
    c=cfg['runners']['ReMFT']
    expected=cfg.get('method_implementation_revisions',{}).get('ReMFT')
    if expected and str(ck.get('method_implementation_revision')) != str(expected):
        raise RuntimeError(
            f'ReMFT Formal run uses stale implementation revision '
            f'{ck.get("method_implementation_revision")!r}; expected {expected!r}. Rerun Formal.'
        )
    mean,std=scale_stats(fc)
    ds=ReMFTDataset(cfg,ec,'test',mean=mean,std=std)
    ds=Subset(ds,range(min(len(ds),max_rows))) if max_rows else ds
    dl=DataLoader(ds,batch_size=int(c.get('formal_eval_batch_size',c['batch_size'])),shuffle=False,num_workers=0)
    m=remft_build(cfg,device); remft_load(m,ck['model'])
    use_amp=bool(c.get('formal_use_amp',True))
    return remft_predict(m,dl,device,mean,std,use_amp=use_amp)


def main():
    methods=['ImputeFormer','LRTC-TMCP','GinAR','RAFT','RAST','SARAF','ReMFT']
    ap=argparse.ArgumentParser(); ap.add_argument('--method',choices=methods,required=True); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--severity',type=int,choices=[20,40,60,90],required=True); ap.add_argument('--formal-run-dir',default=None); ap.add_argument('--lifecycle-checkpoint',default=None,help='Smoke-only checkpoint; bypass completed-Formal selection and write only under _lifecycle_smoke.'); ap.add_argument('--lifecycle-max-batches',type=int,default=2,help='Lifecycle smoke only: cap real inference to this many evaluation batches.'); a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/baseline.yaml'); set_seed(a.seed); device=require_gpu(); exp=Path(cfg['project']['exp_root']); fc=cache_path(cfg,'formal'); ec=cache_path(cfg,'evaluation',a.severity)
    if a.lifecycle_checkpoint:
        cp=Path(a.lifecycle_checkpoint).resolve()
        if not cp.exists(): raise FileNotFoundError(cp)
        ck=torch.load(cp,map_location='cpu',weights_only=False)
        if not bool(ck.get('lifecycle_smoke',False)): raise RuntimeError('Lifecycle checkpoint marker missing')
        sel={'run_dir':str(cp.parent),'checkpoint':str(cp),'checkpoint_sha256':sha256_file(cp),'selection_mode':'lifecycle_smoke'}
        print(f'LIFECYCLE_CHECKPOINT_RELOAD_PASS method={a.method} severity=s{a.severity} checkpoint={cp}',flush=True)
    else:
        sel,ck=load_sel(cfg,a.method,a.seed,a.formal_run_dir)
    fn={'GinAR':ginar,'ImputeFormer':impute,'LRTC-TMCP':lrtc_tmcp,'RAFT':raft,'SARAF':saraf,'ReMFT':remft}
    max_rows=None
    if a.lifecycle_checkpoint:
        bs=int(cfg['runners'][a.method].get('formal_eval_batch_size',cfg['runners'][a.method].get('batch_size',1)))
        max_rows=max(1,int(a.lifecycle_max_batches))*bs
        print(f'LIFECYCLE_INFERENCE_LIMIT method={a.method} severity=s{a.severity} max_batches={a.lifecycle_max_batches} max_rows={max_rows}',flush=True)
    raw=rast(cfg,fc,ec,a.seed,device,ck,sel,max_rows=max_rows) if a.method=='RAST' else fn[a.method](cfg,fc,ec,a.seed,device,ck,max_rows=max_rows)
    rd=(Path(sel['run_dir'])/'evaluation'/f's{a.severity}') if a.lifecycle_checkpoint else (exp/'evaluation'/a.method/cfg['experiment']['subgraph']/f'seed{a.seed}'/f's{a.severity}'); rd.mkdir(parents=True,exist_ok=True)
    if a.lifecycle_checkpoint:
        np.savez_compressed(rd/'test_predictions.smoke.npz',pred=np.asarray(raw,np.float32))
        sm={'method':a.method,'seed':int(a.seed),'split':'test','smoke_rows':int(raw.shape[0]),'prediction_shape':list(raw.shape),'formal_checkpoint':sel['checkpoint'],'formal_checkpoint_sha256':sel['checkpoint_sha256'],'test_access':'lifecycle_smoke_limited_inference','severity':int(a.severity),'protocol_name':ec.parent.parent.parent.name}
        (rd/'test_summary.smoke.json').write_text(json.dumps(sm,indent=2))
    else:
        sm=save_split_outputs(cfg,ec,rd,'test',raw,a.method,a.seed,{'formal_checkpoint':sel['checkpoint'],'formal_checkpoint_sha256':sel['checkpoint_sha256'],'reporting_track':cfg.get('methods',{}).get(a.method,{}).get('table_group','unknown'),'test_access':'direct_after_completed_formal_run','severity':int(a.severity),'protocol_name':ec.parent.parent.parent.name})
    print(f'EVALUATION_RESULT_DIR={rd}')
    if a.lifecycle_checkpoint: print(f'LIFECYCLE_EVAL_PASS method={a.method} severity=s{a.severity} rows={int(raw.shape[0])} shape={tuple(raw.shape)}',flush=True)
    print(json.dumps(sm,default=str,indent=2))


if __name__=='__main__': main()
