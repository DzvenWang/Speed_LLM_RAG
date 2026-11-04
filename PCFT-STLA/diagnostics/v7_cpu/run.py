#!/usr/bin/env python3
"""PCFT v7 read-only CPU-only behavior, paired intervention & report tool.

Never fits any parameters, updates original cache, or uses a future target as a model input.
"""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path

# Disable CUDA before importing torch; also enforced by loader/device checks.
os.environ['CUDA_VISIBLE_DEVICES']=''
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from diagnostics.v7_cpu.diagnostic.loader import load_model
from diagnostics.v7_cpu.diagnostic.collector import DiagnosticCollector
from diagnostics.v7_cpu.diagnostic.bootstrap import block_bootstrap
from diagnostics.v7_cpu.diagnostic.sampling import choose_indices,sample_manifest,manifest_sha
from diagnostics.v7_cpu.diagnostic.reporting import write_csv,json_write,generate_report
from diagnostics.v7_cpu.diagnostic.interventions import SOURCE,ALL_INTERVENTIONS
from pcft_stla.v6.config import sha256


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,help='Trained PCFT joint_best.pt (read-only)')
    p.add_argument('--backbone-checkpoint',required=True,help='Official STLA best.pt')
    p.add_argument('--target',choices=['speed','flow'],required=True)
    p.add_argument('--dataset',default='TraffiDent_Net_06')
    p.add_argument('--split',default='val',choices=['val','test_s20','test_s40','test_s60','test_s90'])
    p.add_argument('--allow-test',action='store_true',help='Explicitly permit descriptive test analysis; no tuning')
    p.add_argument('--mode',choices=['summary','behavior','intervention','deep'],default='behavior')
    p.add_argument('--max-windows',type=int,default=512,help='0 = all available windows')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--cpu-threads',type=int,default=8)
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--max-fresh-age',type=int,default=3)
    p.add_argument('--output',required=True,help='New output directory, never original experiment path')
    p.add_argument('--interventions',nargs='*',default=list(SOURCE),choices=list(ALL_INTERVENTIONS))
    p.add_argument('--max-deep-windows',type=int,default=256)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.split.startswith('test_') and not args.allow_test:
        raise SystemExit('TEST_LOCK: use --allow-test for descriptive test reports; never tune on Test')
    if args.batch_size<=0 or args.cpu_threads<1 or args.max_windows<0:raise ValueError('Invalid CPU resource limits')
    if args.mode=='deep' and (args.max_windows==0 or args.max_windows>args.max_deep_windows):
        raise ValueError('Deep edge tracing requires 1<=max-windows<=max-deep-windows')
    checkpoint=Path(args.checkpoint).resolve();base=Path(args.backbone_checkpoint).resolve()
    output=Path(args.output).resolve()
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if not base.is_file():raise FileNotFoundError(base)
    # Avoid writing into experiments, caches, or private sibling source tree.
    forbidden=('PCFT-STLA_exps','PCFT-STLA_cache','Backbone_exps','Base_SupportBench','Backbone')
    if any(x in output.parts for x in forbidden) or output==checkpoint.parent or output==base.parent:
        raise ValueError('Output must be a separate diagnostics directory, not source data or checkpoints')
    if output.exists() and any(output.iterdir()):raise FileExistsError('Refuse to overwrite nonempty diagnostic directory: '+str(output))
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    start=time.monotonic()
    if args.mode=='summary':
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        cfg=cp['config'];ident=cp['identity']
        if cfg['experiment']['target']!=args.target:raise ValueError('Target mismatch')
        summary={'target':args.target,'checkpoint':str(checkpoint),'schema':cp.get('schema'),
                 'identity':ident,'source':'existing_checkpoint_header','mode':'summary'}
        for name in ('test_results.json','train_partition.json'):
            f=checkpoint.parent/name
            if f.is_file():summary[name]=json.loads(f.read_text())
        epoch=checkpoint.parent/'epoch_history.jsonl'
        if epoch.is_file():summary['epoch_rows']=[json.loads(line) for line in epoch.read_text().splitlines() if line.strip()]
        output.mkdir(parents=True,exist_ok=True)
        json_write(output/'summary.json',summary)
        print('CPU_DIAGNOSTIC_COMPLETE summary',str(output),flush=True)
        return
    model,datasets,identity,cfg,graph,caches=load_model(checkpoint,base)
    if identity['target']!=args.target or identity['dataset']!=args.dataset:
        raise RuntimeError('Target/dataset mismatch after immutable checkpoint verification')
    ds=datasets[args.split]
    indices=choose_indices(ds,args.max_windows,args.seed)
    manifest=sample_manifest(ds,indices)
    # Parity requires one official batch only, no future labels enter model.
    validation_batch=next(iter(DataLoader(Subset(ds,indices[:min(2,len(indices))]),batch_size=2,num_workers=0)))
    with torch.inference_mode():
        parity=model.anchor.parity(validation_batch)
    print('CPU_DIAGNOSTIC_ANCHOR_PARITY_PASS',parity,flush=True)
    if len(indices)==0:raise RuntimeError('No selected windows')
    if args.mode=='deep' and len(indices)>args.max_deep_windows:raise RuntimeError('Unexpected oversized deep selection')
    collector=DiagnosticCollector(identity['mean'],identity['std'],model.target_idx,args.mode)
    dl=DataLoader(Subset(ds,indices.tolist()),batch_size=args.batch_size,num_workers=0,shuffle=False)
    names=args.interventions if args.mode in ('intervention','deep') else []
    model.eval()
    processed=0
    with torch.inference_mode():
        for batch in dl:
            n=int(batch['x_multi'].shape[0]);idx=indices[processed:processed+n]
            starts=[int(ds.meta.iloc[int(i)].window_start) for i in idx]
            # Preserve native prediction semantics; trace collection is optional and inert.
            out=model(batch,diagnostic=True,trace_edges=args.mode=='deep')
            collector.update(batch,out,idx,starts)
            if names:collector.update_interventions(model,batch,out,idx,names,args.max_fresh_age)
            processed+=n
            if processed%max(args.batch_size*10,64)==0 or processed==len(indices):
                print('CPU_DIAGNOSTIC_PROGRESS',f'{processed}/{len(indices)}',f'{time.monotonic()-start:.1f}s',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    grouped=collector.grouped.rows()
    summary={'target':args.target,'dataset':args.dataset,'split':args.split,'mode':args.mode,
         'checkpoint_sha256':sha256(checkpoint),'backbone_sha256':identity['backbone_sha256'],
         'graph_identity':identity['graph_identity'],'sample_manifest_sha256':manifest_sha(manifest),
         'windows':len(indices),'global':collector.global_stats.finish(),
         'elapsed_seconds':round(time.monotonic()-start,3),
         'cpu_threads':args.cpu_threads,'target_idx':model.target_idx,
         'interpretation':'Observational metrics and frozen-model counterfactual source interventions; not retrained ablations.'}
    interventions=collector.intervention_rows()
    json_write(output/'summary.json',summary)
    json_write(output/'run_manifest.json',dict(args=vars(args),identity=identity,cache_guards=caches,
        anchor_parity_max_abs=parity,checkpoint_source_path=str(checkpoint),
        dataset_source='canonical Base_SupportBench prebuilt caches',
        test_is_descriptive_only=bool(args.split.startswith('test_'))))
    write_csv(output/'sample_manifest.csv',manifest)
    write_csv(output/'c1_support_groups.csv',[r for r in grouped if r['dimension'] in
         ('target_coverage','last_observation_age','longest_missing_streak','valid_difference_pairs')])
    write_csv(output/'c2_source_groups.csv',[r for r in grouped if r['dimension'] in
         ('available_incoming_edges','weighted_source_age')])
    write_csv(output/'c2_message_statistics.csv',[r for r in grouped if r['dimension'] in
         ('available_incoming_edges','weighted_source_age','horizon')])
    write_csv(output/'horizon_results.csv',[r for r in grouped if r['dimension']=='horizon'])
    write_csv(output/'mask_results.csv',[r for r in grouped if r['dimension']=='evaluation_mask'])
    write_csv(output/'c3_action_statistics.csv',collector.action_rows())
    write_csv(output/'c3_gain_calibration.csv',collector.calibration.rows())
    write_csv(output/'selected_cases.csv',collector.case_rows())
    write_csv(output/'window_metrics.csv',collector.all_windows)
    json_write(output/'bootstrap_intervals.json',block_bootstrap(collector.all_windows,seed=args.seed))
    write_csv(output/'intervention_results.csv',interventions)
    if args.mode=='deep':write_csv(output/'edge_weight_summary.csv',collector.edge_rows())
    generate_report(output,summary,grouped,interventions)
    print('CPU_DIAGNOSTIC_COMPLETE',json.dumps({'target':args.target,'split':args.split,
        'windows':len(indices),'overall_improvement_pct':summary['global'].get('overall_improvement_pct'),
        'output':str(output)}),flush=True)

if __name__=='__main__':main()
