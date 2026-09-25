#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for p in [ROOT, REPO_ROOT / 'TraffiDent_SupportBench']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.dont_write_bytecode = True

from shared_protocol.builder import build_stage, validate_stage
from shared_protocol.utils import load_yaml
from supportbench.dataset import SupportBench


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['development', 'formal', 'evaluation'], required=True)
    p.add_argument('--force', action='store_true')
    p.add_argument('--config', default=str(ROOT / 'configs' / 'protocol_s40.yaml'))
    a = p.parse_args()
    cfg = load_yaml(a.config)
    cfg['_config_path'] = str(Path(a.config).resolve())
    allowed = set(map(str, cfg.get('protocol', {}).get('allowed_stages', [])))
    if a.stage not in allowed:
        raise SystemExit(
            f"REFUSE: stage={a.stage} disabled for {cfg['protocol']['name']}; allowed={sorted(allowed)}"
        )
    b = SupportBench(cfg['experiment']['dataset'], root=cfg['project']['benchmark_root'])
    out = build_stage(b, cfg, a.stage, force=a.force)
    try:
        manifest = validate_stage(cfg, a.stage)
    except Exception:
        (out / 'FROZEN.json').unlink(missing_ok=True)
        raise
    print('PROTOCOL_NAME=' + str(cfg['protocol']['name']))
    print('PROTOCOL_SEVERITY=' + str(cfg['experiment']['severity']))
    print('PROTOCOL_STAGE=' + a.stage)
    print('PROTOCOL_STAGE_HASH=' + manifest['stage_hash'])
    print('PROTOCOL_FROZEN=' + str(out / 'FROZEN.json'))
    if a.stage == 'evaluation':
        print('EVALUATION_MANIFEST_FROZEN=true')
        print('EVALUATION_ACCESS=DIRECT_UNLOCKED')
    print('PROTOCOL BUILD PASS')


if __name__ == '__main__':
    main()
