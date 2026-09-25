#!/usr/bin/env python3
from pathlib import Path
import argparse
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for p in [ROOT, REPO_ROOT / 'TraffiDent_SupportBench']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.dont_write_bytecode = True

from shared_protocol.builder import filtered_views
from shared_protocol.utils import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'protocol_s40.yaml'))
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    print('Shared Protocol preflight v' + str(cfg['version']))
    print('protocol=', cfg['protocol']['name'])
    print('severity=', cfg['experiment']['severity'])
    print('allowed_stages=', cfg['protocol']['allowed_stages'])
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
        print('INFO SupportBench package version', getattr(supportbench, '__version__', 'unknown'))
        b = SupportBench(cfg['experiment']['dataset'], root=cfg['project']['benchmark_root'])
        expected_bench = str(cfg['source']['supportbench_benchmark_version'])
        actual_bench = str(b.manifest.get('version', 'unknown'))
        ok_version = actual_bench == expected_bench
        print(('PASS' if ok_version else 'FAIL'), 'SupportBench benchmark version', actual_bench, 'expected=', expected_bench)
        failed |= not ok_version
        ok = int(b.H) == int(cfg['experiment']['history']) and int(b.P) == int(cfg['experiment']['horizon'])
        print(('PASS' if ok else 'FAIL'), 'SupportBench geometry', 'N=', b.N, 'H=', b.H, 'P=', b.P)
        failed |= not ok

        # Always require Test coverage for the requested severity because all four
        # protocols freeze paired evaluation manifests now.
        test = filtered_views(b, cfg, 'test', False)
        fams = list(cfg['experiment']['families'])
        counts = test.family.astype(str).value_counts().to_dict()
        missing = [f for f in fams if int(counts.get(f, 0)) == 0]
        ok = len(missing) == 0
        print(('PASS' if ok else 'FAIL'), 'Test severity/family coverage', counts)
        if missing:
            print('MISSING TEST FAMILIES', missing)
        failed |= not ok

        if 'development' in set(map(str, cfg['protocol']['allowed_stages'])):
            train = filtered_views(b, cfg, 'train', False)
            tcounts = train.family.astype(str).value_counts().to_dict()
            missing_train = [f for f in fams if int(tcounts.get(f, 0)) == 0]
            ok = len(missing_train) == 0
            print(('PASS' if ok else 'FAIL'), 'Train severity/family coverage', tcounts)
            if missing_train:
                print('MISSING TRAIN FAMILIES', missing_train)
            failed |= not ok

        needs_val_challenge = (
            'development' in set(map(str, cfg['protocol']['allowed_stages']))
            and str(cfg.get('protocol', {}).get('development_val_pool_policy', 'challenge')).lower() == 'challenge'
        )
        p = Path(cfg['project']['support_artifact_root']) / cfg['experiment']['dataset'] / 'val_challenge_pool.npz'
        if needs_val_challenge:
            ok = p.exists(); print(('PASS' if ok else 'FAIL'), p); failed |= not ok
        else:
            print('SKIP val challenge pool', p)
    except Exception as e:
        print('FAIL SupportBench', repr(e)); failed = True
    if failed:
        raise SystemExit(2)
    print('PREFLIGHT PASS')


if __name__ == '__main__':
    main()
