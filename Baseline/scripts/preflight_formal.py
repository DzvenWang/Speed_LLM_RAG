#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, subprocess, sys
from pathlib import Path

GADI_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.utils import load_yaml
from baseline.shared_cache import cache_path
from baseline.backbones.reference_backbone import resolve_reference_checkpoint, SUPPORTED_BACKBONES
from baseline.compat.imputeformer_tsl_shim import install_imputeformer_tsl_shim

ap = argparse.ArgumentParser()
ap.add_argument('--method', required=True, choices=['GinAR','ImputeFormer','LRTC-TMCP','SPRINT','RAFT','RAST','SARAF','MoGERNN','ReMFT','RAFT-PatchSTG','RAFT-STLAformer','SARAF-PatchSTG','RAST-PatchSTG','RAST-STLAformer'])
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--stage', choices=['formal','evaluation'], default='formal')
ap.add_argument('--severity', type=int, choices=[20,40,60,90])
ap.add_argument('--backbone', choices=SUPPORTED_BACKBONES)
ap.add_argument('--backbone-checkpoint', default=None)
a = ap.parse_args()
cfg = load_yaml(ROOT / 'configs/baseline.yaml')
if a.method in ('RAST-STLAformer','RAFT-STLAformer') and a.backbone != 'STLAformer':
    raise SystemExit(f'{a.method} requires --backbone STLAformer')
if a.method in ('RAST-PatchSTG','RAFT-PatchSTG') and a.backbone != 'PatchSTG':
    raise SystemExit(f'{a.method} requires --backbone PatchSTG')
if a.stage=='evaluation' and a.severity is None: raise SystemExit('--severity is required for evaluation')
method_seeds = [int(x) for x in cfg.get('methods', {}).get(a.method, {}).get('formal_seeds', cfg['experiment']['formal_seeds'])]
if a.seed not in method_seeds:
    raise SystemExit(f'Invalid formal seed {a.seed} for {a.method}; allowed={method_seeds}')
cmd=[cfg['project']['python'], str(ROOT / 'scripts/validate_cache.py'), '--stage', a.stage]
if a.stage=='evaluation': cmd += ['--severity', str(a.severity)]
subprocess.check_call(cmd)

need_reference = bool(cfg.get('methods',{}).get(a.method,{}).get('requires_reference_backbone_for_final_forecast',False))
if need_reference:
    if not a.backbone:
        raise SystemExit(f'--backbone is required for {a.method}')
    resolve_reference_checkpoint(
        cfg,a.backbone,seed=int(cfg.get('reference_backbone',{}).get('seed',10)),
        checkpoint=a.backbone_checkpoint,
    )

# Method-specific runtime dependency checks.
#
# The locked ImputeFormer source ships its modified TSL package inside
# vendor/ImputeFormer/experiments/tsl.  The official runtime prepends the
# experiments directory to sys.path, so checking find_spec('tsl') before
# adding that path produces a false negative and must not be used.
if a.method == 'ImputeFormer':
    impute_exp = ROOT / 'vendor' / 'ImputeFormer' / 'experiments'
    bundled_tsl = impute_exp / 'tsl' / '__init__.py'
    bundled_model = impute_exp / 'imputeformer' / 'models' / '__init__.py'

    if not bundled_tsl.exists():
        raise SystemExit(
            f'Missing bundled ImputeFormer TSL implementation: {bundled_tsl}'
        )
    if not bundled_model.exists():
        raise SystemExit(
            f'Missing bundled ImputeFormer model package: {bundled_model}'
        )

    if str(impute_exp) not in sys.path:
        sys.path.insert(0, str(impute_exp))

    try:
        shim = install_imputeformer_tsl_shim()
        from imputeformer.models import ImputeFormerModel
    except Exception as e:
        raise SystemExit(
            'Failed to import ImputeFormer with the minimal TSL compatibility '
            f'shim: {type(e).__name__}: {e}'
        ) from e

    print('PASS ImputeFormer minimal TSL compatibility shim', shim)
    print(f'PASS ImputeFormerModel import {ImputeFormerModel.__module__}')

for dep in ({'RAST':['faiss'], 'RAST-PatchSTG':['faiss'], 'RAST-STLAformer':['faiss'], 'SPRINT':['termcolor'], 'MoGERNN':['einops'], 'ReMFT':['transformers','accelerate','einops']}.get(a.method, [])):
    if importlib.util.find_spec(dep) is None:
        raise SystemExit(f'Missing required dependency for {a.method}: {dep}')
runner = {
    'GinAR':'ginar_runner.py', 'ImputeFormer':'imputeformer_runner.py', 'LRTC-TMCP':'lrtc_tmcp_runner.py', 'RAFT':'raft_runner.py',
    'RAST':'rast_runner.py', 'RAST-PatchSTG':'rast_runner.py', 'RAST-STLAformer':'rast_runner.py', 'SARAF':'saraf_runner.py',
    'RAFT-PatchSTG':'retrieval_reference_control_runner.py', 'RAFT-STLAformer':'retrieval_reference_control_runner.py', 'SARAF-PatchSTG':'retrieval_reference_control_runner.py',
    'SPRINT':'sprint_runner.py', 'MoGERNN':'mogernn_runner.py', 'ReMFT':'remft_runner.py',
}[a.method]
if a.method == 'ReMFT':
    subprocess.check_call([cfg['project']['python'], str(ROOT/'scripts/check_remft_runtime.py')])
if not (ROOT / 'baseline/runners' / runner).exists():
    raise SystemExit(f'Missing runner {runner}')
print(f'FORMAL PREFLIGHT PASS method={a.method} seed={a.seed} stage={a.stage}')
