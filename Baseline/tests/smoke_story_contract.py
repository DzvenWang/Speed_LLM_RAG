from pathlib import Path
import sys,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from baseline.registry import MAIN_ORDER,LLM_ORDER,SUBTRACK_ORDER
cfg=yaml.safe_load((ROOT/'configs/baseline.yaml').read_text())
assert cfg['baseline_story']['main_table']==['FrozenBackbone']+MAIN_ORDER
assert cfg['baseline_story']['llm_specific']==LLM_ORDER
assert cfg['baseline_story']['partial_sensing_subtrack']==SUBTRACK_ORDER
assert 'GinAR' in cfg['methods'] and 'SARAF' in cfg['methods']
print('BASELINE STORY CONTRACT SMOKE PASS')
