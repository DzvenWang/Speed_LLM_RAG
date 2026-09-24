#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,time,torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from online_presure.config import load_config
from online_presure.contract import assert_cache_ready
from online_presure.backbone import load_patchstg
from online_presure.data import OnlineCache
from online_presure.stream import run_stream
from online_presure.utils import seed_all,write_json,log

VARIANTS=['frozen','slow_only','rag_similarity_global','rag_similarity_compositional','rag_utility_compositional']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--split',default=None)
    ap.add_argument('--variant',default='rag_utility_compositional',choices=VARIANTS); ap.add_argument('--run-tag',default=None)
    ap.add_argument('--project-root',default=None); ap.add_argument('--checkpoint',default=None); ap.add_argument('--max-queries',type=int,default=None)
    ap.add_argument('--replay-memory-size',type=int,default=None); ap.add_argument('--evidence-memory-size',type=int,default=None)
    ap.add_argument('--top-k',type=int,default=None); ap.add_argument('--prefilter-k',type=int,default=None); ap.add_argument('--pool-k',type=int,default=None)
    ap.add_argument('--node-top-k',type=int,default=None); ap.add_argument('--patch-top-k',type=int,default=None);
    ap.add_argument('--utility-probes',type=int,default=None); ap.add_argument('--utility-min-examples',type=int,default=None);
    ap.add_argument('--utility-helpful-prob-min',type=float,default=None); ap.add_argument('--utility-harmful-prob-max',type=float,default=None); ap.add_argument('--slow-batch-size',type=int,default=None)
    ap.add_argument('--slow-steps',type=int,default=None); ap.add_argument('--slow-lr',type=float,default=None)
    ap.add_argument('--recency-tau',type=float,default=None); ap.add_argument('--completion-max-gate',type=float,default=None)
    ap.add_argument('--max-external-weight',type=float,default=None)
    a=ap.parse_args(); cfg=load_config(ROOT,a.project_root); seed_all(cfg['experiment']['seed'])
    if a.split: cfg['experiment']['stream_split']=a.split
    for k,v in [('replay_memory_size',a.replay_memory_size),('batch_size',a.slow_batch_size),('steps_per_update',a.slow_steps),('learning_rate',a.slow_lr)]:
        if v is not None: cfg['adaptation'][k]=v
    if a.evidence_memory_size is not None: cfg['evidence_memory']['capacity']=a.evidence_memory_size
    for k,v in [('top_k',a.top_k),('prefilter_k',a.prefilter_k),('recency_tau_steps',a.recency_tau)]:
        if v is not None: cfg['retrieval'][k]=v
    if a.pool_k is not None: cfg['compositional']['candidate_pool_k']=a.pool_k
    if a.node_top_k is not None: cfg['utility']['top_k_per_patch']=a.node_top_k
    if a.patch_top_k is not None: cfg['utility']['top_k_per_patch']=a.patch_top_k
    if a.utility_probes is not None: cfg['utility']['probes_per_query']=a.utility_probes
    if a.utility_min_examples is not None: cfg['utility']['min_examples']=a.utility_min_examples
    if a.utility_helpful_prob_min is not None: cfg['utility']['helpful_prob_min']=a.utility_helpful_prob_min
    if a.utility_harmful_prob_max is not None: cfg['utility']['harmful_prob_max']=a.utility_harmful_prob_max
    if a.completion_max_gate is not None: cfg['completion']['max_gate']=a.completion_max_gate
    if a.max_external_weight is not None: cfg['fusion']['max_external_weight']=a.max_external_weight
    assert_cache_ready(cfg,a.stage,a.split); ds=OnlineCache(cfg,a.stage,a.split)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model,ck_path,ck,source=load_patchstg(cfg,a.stage,device,a.checkpoint)
    tag=a.run_tag or f"online_presure_v330.{a.variant}.{time.strftime('%Y%m%d_%H%M%S')}"
    out=Path(cfg['project']['exp_root'])/cfg['experiment']['subgraph']/a.stage/tag; out.mkdir(parents=True,exist_ok=True)
    write_json(out/'resolved_config.json',cfg)
    log(f"RUN tag={tag} stage={a.stage} split={ds.split} variant={a.variant} queries={len(ds) if a.max_queries is None else min(len(ds),a.max_queries)} device={device}")
    s=run_stream(cfg,ds,model,ck_path,ck,source,a.variant,out,device,a.max_queries)
    oracle=s['retrieved_entry_oracle_by_k_vs_slow_pct']
    log('SUMMARY '+json.dumps({'variant':a.variant,'slow_gain_pct':s['slow_gains_pct']['masked_mae'],
        'final_gain_pct':s['final_gains_pct']['masked_mae'],'rag_increment_vs_slow_pct':s['rag_increment_vs_slow_pct']['masked_mae'],
        'perfect_completion_oracle_vs_slow_pct':s['perfect_completion_oracle_vs_slow_pct']['masked_mae'],
        'entry_oracle_at8_vs_slow_pct':oracle.get('8',{}).get('masked_mae'),
        'entry_oracle_at16_vs_slow_pct':oracle.get('16',{}).get('masked_mae'),
        'entry_oracle_at32_vs_slow_pct':oracle.get('32',{}).get('masked_mae'),
        'entry_oracle_at64_vs_slow_pct':oracle.get('64',{}).get('masked_mae'),
        'fill_coverage':s['missing_completion']['fill_coverage'],'filled_reconstruction_mae_norm':s['missing_completion']['filled_reconstruction_mae_norm'],
        'base_masked_mae':s['base']['masked_mae'],'slow_masked_mae':s['slow_online']['masked_mae'],'final_masked_mae':s['online_presure_v33']['masked_mae'],
        'utility_examples':s.get('utility',{}).get('examples'),'utility_class_counts':s.get('utility',{}).get('class_counts'),
        'utility_ready_fraction':s.get('utility',{}).get('ready_fraction'),'mean_selected_p_helpful':s.get('utility',{}).get('mean_selected_p_helpful'),
        'sampled_patch_utility_oracle_pct':s.get('utility',{}).get('sampled_single_patch_oracle_vs_probe_slow_pct',{}).get('masked_mae')},sort_keys=True))
    log(f"OUT_DIR={out}")
if __name__=='__main__': main()
