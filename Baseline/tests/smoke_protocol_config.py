from pathlib import Path
import yaml, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from baseline.registry import METHODS, ORDER, MAIN_ORDER, LLM_ORDER, SUBTRACK_ORDER
cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text()); locks=yaml.safe_load((ROOT/'configs/official_sources.yaml').read_text())['official_sources']
assert cfg['version']=='0.4.0'
assert cfg['protocol']['no_local_sampling'] is True
assert cfg['protocol']['retrieval_memory']=='train_only'
assert cfg['protocol']['retrieval_memory_signal']=='full_train_history'
assert cfg['protocol']['retrieval_query_signal']=='incomplete_query_history'
assert cfg['protocol_assertions']['development']=={'train_rows':12000,'val_rows':5000}
assert cfg['protocol_assertions']['formal']['val_rows']==5000
assert cfg['protocol_assertions']['formal']['train_policy']=='all_eligible_unique_groups'
assert cfg['protocol_assertions']['evaluation']=={'test_rows':5000}
assert cfg['experiment']['formal_seeds']==[10,20,30]
assert MAIN_ORDER==['ImputeFormer','GinAR','RAFT','RAST','SARAF']
assert LLM_ORDER==['LEAF'] and SUBTRACK_ORDER==['SLPF']
for m in ORDER:
    assert METHODS[m]['repo']==locks[m]['remote']
    assert METHODS[m]['source_policy']==locks[m].get('policy','exact')
    if locks[m].get('policy','exact')=='exact': assert METHODS[m]['commit']==locks[m]['commit']
    else: assert METHODS[m]['commit'] is None and locks[m]['ref']=='main'
assert cfg['runners']['SARAF']['n_period']==1  # current official SARAF forward consumes one retrieval branch
print('PROTOCOL CONFIG SMOKE PASS')
