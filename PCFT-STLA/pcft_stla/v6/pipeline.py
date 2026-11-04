from __future__ import annotations
import json,os,time,datetime,copy,hashlib,traceback
from pathlib import Path
import numpy as np,torch
from .config import root,sha256,json_write,make_paths,resolve_checkpoint
from .data import open_bench,make_multi_cache,MultiTargetSplit
from .graph import build_or_load_graph,graph_audit
from .model import PCFTV6,load_original_stla
from .training import (seed_all,to_device,make_loader,make_trainable,optimizer_for,scaler_for,epoch_loop,
                       evaluate,atomic_save,build_state,check_identity,restore_rng,batch_probe)


def prepare(cfg,checkpoint,*,smoke=False,cache_only=False):
    target=cfg['experiment']['target'];dataset=cfg['experiment']['dataset']
    repo,formal,eval_paths=make_paths(cfg,target,dataset)
    bench=open_bench(repo,dataset,target=target)
    if bench.H!=12 or bench.P!=12 or len(bench.channel_names)!=3:raise ValueError('Incompatible SupportBench H/P/channels')
    if target not in bench.channel_names:raise ValueError('Selected target not in SupportBench channels')
    ck=resolve_checkpoint(repo,cfg,checkpoint)
    exps=repo/'PCFT-STLA_exps'/target/cfg['backbone']['mode']/dataset/'v6'
    store=repo/'PCFT-STLA_cache'/'v6'/dataset/target
    target_dirs={'train':formal,'val':formal}
    for severity,ep in eval_paths.items():target_dirs[f'test_s{severity:02d}']=ep
    paths={};datasets={}
    limit=int(cfg['runtime']['smoke_rows']) if smoke else None
    for label,cache in target_dirs.items():
        split='test' if label.startswith('test_') else label
        multi=store/(label+'_'+sha256(cache/'cache_manifest.json')[:12])
        paths[label]=make_multi_cache(bench,cache,split,multi,max_rows=limit)
        datasets[label]=MultiTargetSplit(cache,split,multi,limit=limit)
    # Read target mean/std from shared cache, independent of source primary channel.
    identity=json.loads((formal/'cache_manifest.json').read_text())
    fm=identity['splits']['train']
    mean=float(fm.get('target_mean',fm.get('mean',0)))
    std=float(fm.get('target_std',fm.get('std',1)))
    if not np.isfinite(std) or std<=0:raise ValueError('Invalid target normalization stats')
    for k,p in target_dirs.items():
        m=json.loads((p/'cache_manifest.json').read_text())
        info=m['splits']['test' if k.startswith('test_') else k]
        if str(info.get('target_name',m.get('target_name',target)))!=target:raise ValueError(f'Target mismatch {p}')
        if not np.isclose(float(info.get('target_mean',info.get('mean',mean))),mean,rtol=1e-4,atol=1e-5):
            raise RuntimeError(f'{k} target mean mismatch vs train')
        if not np.isclose(float(info.get('target_std',info.get('std',std))),std,rtol=1e-4,atol=1e-5):
            raise RuntimeError(f'{k} target std mismatch vs train')
    # Independent v6 graph artifacts. No v5 output/cache/checkpoint is read.
    graph=build_or_load_graph(bench,datasets['train'],repo,cfg,paths['train'])
    ga=graph_audit(bench,graph)
    print('GRAPH_V6_AUDIT_PASS',json.dumps(ga,sort_keys=True),flush=True)
    if graph.num_nodes!=int(datasets['train'].x.shape[2]):raise RuntimeError('Graph/data node count mismatch')
    details=dict(target=target,dataset=dataset,seed=int(cfg['experiment']['seed']),mode=cfg['backbone']['mode'],
        backbone_checkpoint=str(ck),backbone_sha256=sha256(ck),
        formal_identity=sha256(formal/'cache_manifest.json'),
        eval_identities={k:sha256(p/'cache_manifest.json') for k,p in target_dirs.items() if k.startswith('test_')},
        graph_nodes=graph.num_nodes,graph_edges=int(graph.edge_index.shape[1]),functional_edges=int(graph.functional_index.shape[1]),graph_identity=graph.identity,
        graph_sha256=hashlib.sha256(graph.edge_index.cpu().numpy().tobytes()+graph.edge_attr.cpu().numpy().tobytes()).hexdigest(),
        channels=bench.channel_names,mean=mean,std=std)
    return repo,exps,ck,graph,datasets,details


def instantiate(cfg,repo,ck,graph,details,device):
    base,package=load_original_stla(repo,ck,device=device)
    if int(package['num_nodes'])!=graph.num_nodes:raise ValueError('Checkpoint num_nodes vs graph mismatch')
    target_idx=details['channels'].index(details['target'])
    model=PCFTV6(base,graph,target_idx,channels=len(details['channels']),
        hidden=int(cfg['model']['hidden_dim']),max_lag=int(cfg['model']['max_lag']),horizon=int(cfg['experiment']['horizon'])).to(device)
    adapter=os.environ.get('PCFT_SHARED_ADAPTER_CHECKPOINT') or cfg['backbone'].get('shared_adapter_checkpoint')
    if adapter:
        ap=Path(adapter);ap=ap if ap.is_absolute() else repo/ap
        model.encoder.load_mask_adapter(ap,details['dataset'],details['target'],details['seed'],details['mode'],details['backbone_sha256'])
        print(f'SHARED_MASK_ADAPTER_LOADED path={ap}',flush=True)
    return model


def report_row(stage,epoch,elapsed,train,val,lr):
    return {'stage':stage,'epoch':epoch,'elapsed_s':round(elapsed,2),'training_loss':train['loss'],
            'training_amp_skips':train['amp_skips'],'val_mae':val['mae'],'val_rmse':val['rmse'],
            'val_masked_mae':val['masked_mae'],'val_h12_mae':val['h12_mae'],'learning_rate':lr}


def run(cfg,checkpoint=None,run_name=None,smoke=False,resume=None,preflight_only=False):
    seed_all(int(cfg['experiment']['seed']))
    if not torch.cuda.is_available() and not smoke and not preflight_only:
        raise RuntimeError('Formal PCFT training requires CUDA; use --smoke for CPU testing')
    print('V6_PREFLIGHT_BEGIN',flush=True)
    repo,exps,ck,graph,ds,identity=prepare(cfg,checkpoint,smoke=smoke)
    if preflight_only:
        # Fail fast on source/weights shape as well, not only on cache existence.
        model=instantiate(cfg,repo,ck,graph,identity,'cpu')
        batch=next(iter(make_loader(ds['train'],2,0,False,torch.device('cpu'))))
        with torch.no_grad():
            for stage in PCFTV6.STAGES:
                model.set_stage(stage)
                o=model(batch)
                if o['prediction'].shape!=batch['y_norm'].shape or not torch.isfinite(o['prediction']).all():
                    raise RuntimeError(f'Preflight forward failure for {stage}')
        print('V6_PREFLIGHT_PASS',json.dumps(identity,sort_keys=True),flush=True)
        return None
    run_dir=(Path(resume).resolve().parent if resume else exps/('smoke' if smoke else 'formal')/(run_name or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+f'_seed{identity["seed"]}'))
    run_dir.mkdir(parents=True,exist_ok=True)
    json_write(run_dir/'identity.json',identity)
    json_write(run_dir/'config.json',cfg)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=instantiate(cfg,repo,ck,graph,identity,device)
    # GPU preflight for ALL stages, including cross-mask/backprop & AMP; diagnose
    # failures before long S0 training and before spending queue hours.
    if device.type=='cuda' and not resume:
        print('GPU_SMOKE_BEGIN',flush=True)
        probe_loader=make_loader(ds['train'],min(2,len(ds['train'])),0,False,device)
        probe_batch=next(iter(probe_loader))
        from .training import to_device,compute_loss,autocast_for
        probe_batch=to_device(probe_batch,device)
        for stage_name in PCFTV6.STAGES:
            model.set_stage(stage_name)
            make_trainable(model,stage_name,cfg)
            model.zero_grad(set_to_none=True)
            with autocast_for(device,cfg):
                o=model(probe_batch);loss,_=compute_loss(probe_batch,o,cfg)
            if not torch.isfinite(loss):raise RuntimeError(f'GPU_SMOKE_NONFINITE {stage_name}')
            loss.backward()
            if stage_name in ('transport','coupled') and model.dynamics.trans.msg.weight.grad is None:
                raise RuntimeError(f'GPU_SMOKE_NO_TRANSPORT_GRAD {stage_name}')
            print('GPU_SMOKE_STAGE_PASS',stage_name,'loss',float(loss.detach()),flush=True)
        model.zero_grad(set_to_none=True)
        del probe_batch,probe_loader,o,loss
        # Restore reproducible RNG / initial parameters after all smoke forwards.
        seed_all(int(cfg['experiment']['seed']))
        model=instantiate(cfg,repo,ck,graph,identity,device)
        print('GPU_SMOKE_PASS',flush=True)
    bsz=int(cfg['training']['batch_size'])
    if cfg['training'].get('auto_batch_probe',False) and device.type=='cuda' and not resume:
        # Probe uses optimizer.step; restore every weight and RNG afterwards.
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
        trial_loader=make_loader(ds['train'],bsz,0,False,device)
        bsz=batch_probe(model,trial_loader,device,cfg,cfg['training']['candidate_batch_sizes'])
        model.load_state_dict(initial,strict=True)
        torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
        print('SELECTED_BATCH',bsz,flush=True)
    nworkers=int(cfg['runtime']['workers'])
    train_loader=make_loader(ds['train'],bsz,nworkers,True,device)
    val_loader=make_loader(ds['val'],int(cfg['runtime']['eval_batch']),0,False,device)
    print('DATA_READY',json.dumps({k:len(v) for k,v in ds.items()}),'nodes',graph.num_nodes,'edges',graph.edge_index.shape[1],flush=True)
    stages=cfg['training']['stages'];names=[s['name'] for s in stages]
    if names!=list(PCFTV6.STAGES):raise RuntimeError(f'Stage order must be {PCFTV6.STAGES}')
    completed=[];resume_state=None;resume_stage=None;resume_epoch=0
    if resume:
        resume_state=torch.load(resume,map_location='cpu',weights_only=False)
        check_identity(resume_state,identity)
        if resume_state.get('config')!=cfg:raise RuntimeError('Resume training config mismatch')
        model.load_state_dict(resume_state['model'],strict=True)
        completed=list(resume_state['completed']);resume_stage=resume_state['stage'];resume_epoch=int(resume_state['epoch'])
        from .training import restore_rng
        restore_rng(resume_state['rng'])
        print('PIPELINE_RESUME',resume_stage,'next_epoch',resume_epoch+1,flush=True)
    start=time.monotonic();max_hours=float(cfg['runtime']['max_pipeline_hours'])
    history_file=run_dir/'epoch_history.jsonl'
    for sc in stages:
        stage=sc['name'];idx=names.index(stage)
        if stage in completed:continue
        if resume_stage and stage!=resume_stage and names.index(stage)<names.index(resume_stage):continue
        model.set_stage(stage);make_trainable(model,stage,cfg)
        opt=optimizer_for(model,cfg);scaler=scaler_for(device,cfg)
        stage_best=float('inf');stage_best_epoch=0
        first_epoch=1
        if resume_state is not None and resume_stage==stage:
            if resume_state['optimizer'] is not None:opt.load_state_dict(resume_state['optimizer'])
            if resume_state['scaler'] is not None:scaler.load_state_dict(resume_state['scaler'])
            stage_best=float(resume_state['stage_best']);stage_best_epoch=int(resume_state['stage_best_epoch'])
            first_epoch=resume_epoch+1
        patience=int(sc['patience']);max_epochs=(min(int(sc['max_epochs']),2) if smoke else int(sc['max_epochs']))
        for epoch in range(first_epoch,max_epochs+1):
            t=time.perf_counter()
            train=epoch_loop(model,train_loader,device,cfg,opt=opt,scaler=scaler,limit_batches=2 if smoke else None)
            val=evaluate(model,val_loader,device,cfg,identity['mean'],identity['std'],max_batches=2 if smoke else None)
            elapsed=time.perf_counter()-t
            row=report_row(stage,epoch,elapsed,train,val,opt.param_groups[-1]['lr'])
            with history_file.open('a',encoding='utf-8') as f:f.write(json.dumps({'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),**row})+'\n')
            print('V6_EPOCH',json.dumps(row),flush=True)
            if val['mae']+float(cfg['training']['min_delta']) < stage_best:
                stage_best=float(val['mae']);stage_best_epoch=epoch
                atomic_save(run_dir/f'{stage}_best.pt',{'schema':'pcft_v6_stage_best','stage':stage,'epoch':epoch,
                    'model':model.state_dict(),'val':val,'identity':identity})
                print('V6_BEST',stage,epoch,stage_best,flush=True)
            state=build_state(model,opt,scaler,stage,epoch,stage_best,stage_best_epoch,completed,cfg,identity)
            atomic_save(run_dir/'last.pt',state)
            if epoch-stage_best_epoch>=patience:
                print('STAGE_EARLY_STOP',stage,epoch,stage_best_epoch,flush=True);break
            if (time.monotonic()-start)/3600>max_hours:
                raise RuntimeError('V6_TIME_GUARD: saved last.pt; resume in another allocation to complete all stages/tests')
        best=run_dir/f'{stage}_best.pt'
        if not best.exists():raise RuntimeError(f'No valid checkpoint for stage {stage}')
        cp=torch.load(best,map_location='cpu',weights_only=False);check_identity({'schema':'pcft_v6_pipeline_v1','identity':cp['identity']},identity)
        model.load_state_dict(cp['model'],strict=True)
        completed.append(stage)
        next_stage=(names[idx+1] if idx+1<len(names) else 'finished')
        # Atomic transition marker always contains model state restored from best stage.
        state=build_state(model,None,None,next_stage,0,float('inf'),0,completed,cfg,identity)
        atomic_save(run_dir/'last.pt',state)
        json_write(run_dir/f'{stage}_validation.json',{'stage':stage,'best_epoch':cp['epoch'],'validation':cp['val']})
        print('STAGE_COMPLETE',stage,'best_epoch',cp['epoch'],flush=True)
        resume_state=None
    # No Test data has been read in training. All checkpoints selected by validation.
    if len(completed)!=len(names):raise RuntimeError('Pipeline not fully completed')
    results=[]
    for stage in names:
        cp=torch.load(run_dir/f'{stage}_best.pt',map_location=device,weights_only=False)
        model.load_state_dict(cp['model'],strict=True);model.set_stage(stage)
        for severity in cfg['experiment']['severities']:
            key=f'test_s{int(severity):02d}'
            loader=make_loader(ds[key],int(cfg['runtime']['eval_batch']),0,False,device)
            met=evaluate(model,loader,device,cfg,identity['mean'],identity['std'],max_batches=2 if smoke else None)
            result={'stage':stage,'severity':int(severity),'target':identity['target'],**met,'selection':'stage_best_validation'}
            results.append(result)
            print('V6_TEST',json.dumps(result),flush=True)
            json_write(run_dir/'test_results.json',results)
    # Mechanism interventions are NOT retrained ablations; run on val for cost/safety.
    cp=torch.load(run_dir/'coupled_best.pt',map_location=device,weights_only=False)
    model.load_state_dict(cp['model'],strict=True);model.set_stage('coupled')
    interventions={}
    for name in ('none','disable_c1','mask_only','no_transport','remove_functional','uniform_lag','reverse_lag','reverse_direction'):
        v=evaluate(model,val_loader,device,cfg,identity['mean'],identity['std'],
                   intervention=None if name=='none' else name,max_batches=2 if smoke else None)
        interventions[name]=v
        print('V6_INTERVENTION',name,json.dumps(v),flush=True)
    json_write(run_dir/'mechanism_interventions_validation.json',interventions)
    summary={}
    for stage in names:
        rows=[r for r in results if r['stage']==stage]
        summary[stage]={metric:float(np.mean([r[metric] for r in rows if r.get(metric) is not None]))
                        for metric in ('mae','rmse','masked_mae','h12_mae','evolution_mae')}
    changes={}
    for a,b in zip(names,names[1:]):
        changes[f'{a}_to_{b}_mae_pct']=100*(summary[a]['mae']-summary[b]['mae'])/summary[a]['mae']
    final={'schema':'pcft_v6_final_report_v1','identity':identity,'status':'complete',
           'stage_summary':summary,'stage_changes_pct':changes,
           'important_limitation':'Sequential stages are NOT independently retrained ablations. Fixed-checkpoint interventions are diagnostic only.',
           'test_results':results,'mechanism_interventions_on_validation':interventions}
    json_write(run_dir/'FINAL_REPORT.json',final)
    import csv
    with (run_dir/'stage_comparison.csv').open('w',newline='',encoding='utf-8') as fp:
        cols=['stage','mae','rmse','masked_mae','h12_mae','evolution_mae']
        writer=csv.DictWriter(fp,fieldnames=cols);writer.writeheader()
        for stage in names:writer.writerow({'stage':stage,**summary[stage]})
    with (run_dir/'severity_results.csv').open('w',newline='',encoding='utf-8') as fp:
        cols=['stage','severity','target','mae','rmse','masked_mae','h12_mae','evolution_mae','valid_count','masked_count','transport_norm','lag_entropy']
        writer=csv.DictWriter(fp,fieldnames=cols);writer.writeheader()
        for row in results:writer.writerow({k:row.get(k) for k in cols})
    with (run_dir/'fixed_checkpoint_interventions.csv').open('w',newline='',encoding='utf-8') as fp:
        cols=['intervention','mae','rmse','masked_mae','h12_mae','evolution_mae']
        writer=csv.DictWriter(fp,fieldnames=cols);writer.writeheader()
        for name,value in interventions.items():writer.writerow({'intervention':name,**{k:value.get(k) for k in cols[1:]}})

    print('V6_FINAL_REPORT_PASS',run_dir/'FINAL_REPORT.json',flush=True)
    return run_dir
