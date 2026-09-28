#!/usr/bin/env python3
from pathlib import Path
import ast, json, re, sys
ROOT=next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd()); sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode=True
from baseline.registry import ORDER, METHODS
from baseline.utils import load_yaml, write_json
cfg=load_yaml(ROOT/'configs/baseline.yaml'); results={}

def text(rel,method):
    p=ROOT/'vendor'/method/rel
    return p.read_text(encoding='utf-8',errors='ignore') if p.exists() else ''

for m in ORDER:
    spec=METHODS[m]; repo=ROOT/'vendor'/m
    r={'method':m,'configured_commit':spec['commit'],'source_policy':spec['source_policy'],'execution_state':spec['execution_state'],'table_group':spec['table_group'],'entrypoints':{},'hard_findings':[],'pass_static':True}
    for rel in spec['official_entrypoints']:
        p=repo/rel; item={'exists':p.exists()}
        if p.exists() and p.suffix=='.py':
            try: ast.parse(p.read_text(encoding='utf-8',errors='ignore')); item['syntax']='PASS'
            except SyntaxError as e: item['syntax']=f'FAIL {e}'; r['pass_static']=False
        r['entrypoints'][rel]=item
        if not p.exists(): r['pass_static']=False
    if m=='SLPF':
        t=text('lib/dataloader.py',m); fixed=bool(re.search(r'np\.random\.choice\(.{0,300}num_unsensed_locs',t,re.S)) and 'sensed_locations' in t and 'unsensed_locations' in t
        r['fixed_sensed_unsensed_partition']=fixed
        if fixed: r['hard_findings'].append('SLPF is fixed sensed/unsensed partition only; keep it out of dynamic-mask main table.')
    elif m=='RAFT':
        t=text('exp/exp_long_term_forecasting.py',m); test_each=('test_loss = self.vali(test_data' in t) or ('test_loss' in t and 'for epoch' in t)
        r['stock_loop_uses_test_during_training']=test_each
        if test_each: r['hard_findings'].append('Stock RAFT loop evaluates Test during training; formal selection must use external Val-only runner.')
        rr=text('layers/Retrieval.py',m); r['retrieval_prepare_uses_train_data']=('prepare_dataset(self, train_data)' in rr and 'self.y_data_all' in rr)
    elif m=='RAST':
        t=text('src/scripts/main/RAST_PEMS03.py',m); r['declares_universal_rast_arch']=('MODEL_ARCH = RAST' in t); r['requires_easydict']=('from easydict import EasyDict' in t)
    elif m=='LEAF':
        t=text('dataset.py',m); r['stock_dataset_complete_continuous_signal']=('TrafficDataset' in t and 'original_signal' in t and 'train_signal' in t)
        if r['stock_dataset_complete_continuous_signal']: r['hard_findings'].append('LEAF stock data path assumes a complete continuous signal; use isolated external dynamic-view adaptation only for the LLM-specific comparison.')
    elif m=='ImputeFormer':
        r['final_forecast_gate']='FROZEN_BACKBONE.json required after imputation'
    elif m=='GinAR':
        t=text('main_ginar.py',m); arch=text('model1/ginar_arch.py',m)
        r['official_12_to_12_example']=('input_len= 12' in t and 'out_len=12' in t)
        r['official_model_shape_contract']=('history_data' in arch and 'return x.squeeze' in arch)
        r['hard_findings'].append('Stock GinAR script constructs Test and reports it after training; v0.4 uses only the official architecture under an external Train/Val runner.')
    elif m=='SARAF':
        exp=text('exp/exp_long_term_forecasting.py',m); model=text('models/SARAF.py',m); rt=text('layers/Retrieval.py',m)
        r['stock_loop_uses_test_during_training']=('test_loss = self.vali(test_data' in exp)
        r['stock_prepare_dataset_builds_test_retrieval']=('test_temporal_rt = self.rt.retrieve_all(test_data' in model)
        r['retrieval_has_stationarity']=('compute_dataset_stationarity' in rt and 'compute_adaptive_lambda' in rt and 'compute_adaptive_sigma' in rt)
        if r['stock_loop_uses_test_during_training'] or r['stock_prepare_dataset_builds_test_retrieval']:
            r['hard_findings'].append('Stock SARAF touches Test during training/precompute; v0.4 external runner builds Train-only memory and Train/Val retrieval only.')
    results[m]=r
out=Path(cfg['project']['exp_root'])/'_audit'/'runtime_contract_audit.json'; write_json(out,results)
for m,r in results.items(): print(m,json.dumps({'pass_static':r['pass_static'],'group':r['table_group'],'execution_state':r['execution_state'],'hard_findings':r['hard_findings']},sort_keys=True))
print('wrote',out)
if not all(r['pass_static'] for r in results.values()): raise SystemExit(2)
print('RUNTIME CONTRACT STATIC AUDIT PASS')
