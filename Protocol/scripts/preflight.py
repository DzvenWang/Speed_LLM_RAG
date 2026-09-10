#!/usr/bin/env python3
from pathlib import Path
import importlib.util, sys

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Protocol')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'TraffiDent_SupportBench'))
sys.dont_write_bytecode = True
from shared_protocol.utils import load_yaml

cfg = load_yaml(ROOT / 'configs/protocol.yaml')
print('Shared Protocol preflight v' + str(cfg['version']))
failed = False
for p in [
    ROOT,
    Path(cfg['project']['supportbench_project']),
    Path(cfg['project']['benchmark_root']),
    Path(cfg['project']['support_artifact_root']),
]:
    ok = p.exists()
    print(('PASS' if ok else 'FAIL'), p)
    failed |= not ok
for mod in ['numpy', 'pandas', 'yaml', 'scipy', 'sklearn']:
    ok = importlib.util.find_spec(mod) is not None
    print(('PASS' if ok else 'FAIL'), mod)
    failed |= not ok
try:
    import supportbench
    from supportbench.dataset import SupportBench
    if getattr(supportbench, '__version__', None) != str(cfg['source']['supportbench_version']):
        print('FAIL SupportBench package version', getattr(supportbench, '__version__', None)); failed = True
    else:
        print('PASS SupportBench package version', supportbench.__version__)
    b = SupportBench(cfg['experiment']['dataset'], root=cfg['project']['benchmark_root'])
    ok = int(b.N) == 388 and int(b.H) == int(cfg['experiment']['history']) and int(b.P) == int(cfg['experiment']['horizon'])
    print(('PASS' if ok else 'FAIL'), 'SupportBench geometry', 'N=', b.N, 'H=', b.H, 'P=', b.P)
    failed |= not ok
    for split in ['val', 'test']:
        p = Path(cfg['project']['support_artifact_root']) / cfg['experiment']['dataset'] / f'{split}_challenge_pool.npz'
        ok = p.exists(); print(('PASS' if ok else 'FAIL'), p); failed |= not ok
except Exception as e:
    print('FAIL SupportBench', repr(e)); failed = True
if failed:
    raise SystemExit(2)
print('PREFLIGHT PASS')
