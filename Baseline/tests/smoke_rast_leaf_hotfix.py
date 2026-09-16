#!/usr/bin/env python3
from pathlib import Path
import ast,yaml
R=Path(__file__).resolve().parents[1]
c=yaml.safe_load((R/'configs/baseline.yaml').read_text())
assert c['project']['leaf_python']==c['project']['python']
assert c['project']['python'].endswith('/envs/TFP/bin/python')
assert (R/'baseline/compat/rast_optional_backbones.py').exists()
rr=(R/'baseline/runners/rast_runner.py').read_text(); assert 'install_rast_native_import_shims' in rr and 'RAST NATIVE IMPORT SHIMS=' in rr
lr=(R/'baseline/runners/leaf_runner.py').read_text(); assert "choices=['branches','selector']" in lr and 'GraphBranch' in lr and 'HypergraphBranch' in lr
for p in [R/'pbs/methods/development_primary.pbs',R/'pbs/methods/leaf_development.pbs',R/'pbs/methods/leaf_formal.pbs',R/'pbs/methods/leaf_selector.pbs']:
    s=p.read_text(); assert '/Baseline_logs/out/' in s; assert '/envs/LEAF/' not in s; assert '/envs/TFP/bin/python' in s
for p in [R/'baseline/compat/rast_optional_backbones.py',R/'baseline/runners/leaf_runner.py']:
    ast.parse(p.read_text(),filename=str(p))
# Frozen vendor source must not contain our compatibility code.
assert 'install_rast_native_import_shims' not in (R/'vendor/RAST/src/arch/rast_arch.py').read_text()
print('RAST+LEAF HOTFIX CONTRACT SMOKE PASS')
