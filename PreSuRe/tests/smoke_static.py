#!/usr/bin/env python3
from pathlib import Path
import ast,re,yaml
R=Path(__file__).resolve().parents[1]
for p in R.rglob('*.py'): ast.parse(p.read_text(),filename=str(p))
c=yaml.safe_load((R/'configs/presure.yaml').read_text());assert c['version']=='0.3.0';assert c['project']['exp_root'].endswith('/PreSuRe_exps');assert c['project']['log_root'].endswith('/PreSuRe_logs');assert int(c['local_routing']['n_regions'])>=4
for p in R.glob('pbs/*.pbs'):
 s=p.read_text();assert '#PBS -j oe' in s and '/PreSuRe_logs/out/' in s and 'presure_live_log_start' in s
 if '#PBS -q gpuvolta' in s:
  mg=re.search(r'^#PBS -l ngpus=(\d+)\s*$',s,re.M);mc=re.search(r'^#PBS -l ncpus=(\d+)\s*$',s,re.M);assert mg and mc;assert int(mc.group(1))==12*int(mg.group(1))
for f in ['prepare_backbone_artifacts.py','build_retrieval_candidates.py','run_oracle.py','run_variant.py','summarize.py']:
 assert (R/'scripts'/f).exists()
rv=(R/'scripts/run_variant.py').read_text();assert 'HorizonUtilityNet' in rv and 'RegionHorizonUtilityNet' in rv and 'LocalSafeGate' in rv;assert 'horizon_local_features' in rv and 'region_horizon_local_features' in rv
ro=(R/'scripts/run_oracle.py').read_text();assert "choices=['A','B','C','D','E','F']" in ro
br=(R/'scripts/build_retrieval_candidates.py').read_text();assert 'node_regions.npy' in br and 'spectral_graph_regions' in br
print('PRESURE STATIC CONTRACT SMOKE PASS')
