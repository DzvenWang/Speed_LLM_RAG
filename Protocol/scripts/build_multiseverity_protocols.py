#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for p in [ROOT, REPO_ROOT / 'TraffiDent_SupportBench']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared_protocol.utils import load_yaml


CONFIGS = [ROOT / 'configs' / f'protocol_s{s}.yaml' for s in ('20', '40', '60', '90')]


class Tee:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open('w', encoding='utf-8', buffering=1)

    def write(self, text: str):
        print(text, end='', flush=True)
        self.f.write(text)
        self.f.flush()

    def close(self):
        self.f.close()


def run(cmd: list[str], tee: Tee):
    tee.write('RUN ' + ' '.join(map(str, cmd)) + '\n')
    p = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )
    assert p.stdout is not None
    for line in p.stdout:
        tee.write(line)
    rc = p.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)


def main():
    ap = argparse.ArgumentParser(description='Build/freeze the paired 20/40/60/90 Protocol v2 family.')
    ap.add_argument('--force', action='store_true', help='Delete existing v2 protocol artifacts/pairing before rebuild.')
    ap.add_argument('--evaluation-only', action='store_true', help='Skip s40 development/formal and only freeze four paired evaluations.')
    ap.add_argument('--log-file', default=None)
    a = ap.parse_args()

    main_cfg = load_yaml(ROOT / 'configs' / 'protocol_s40.yaml')
    dataset = main_cfg['experiment']['dataset']
    artifact_root = Path(main_cfg['project']['artifact_root'])
    log_root = Path(main_cfg['project']['log_root'])
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = Path(a.log_file) if a.log_file else log_root / f'ProtocolMultiSeverity.{dataset}.{stamp}.log'
    tee = Tee(log_path)

    try:
        tee.write(f'LOG_FILE {log_path}\n')
        tee.write(f'PROJECT_ROOT {REPO_ROOT}\n')
        tee.write(f'PROTOCOL_ROOT {ROOT}\n')
        tee.write(f'PYTHON {sys.executable}\n')
        tee.write(f'DATASET {dataset}\n')
        tee.write('SEVERITIES 20,40,60,90\n')
        tee.write('FAMILIES ' + ','.join(map(str, main_cfg['experiment']['families'])) + '\n')
        tee.write(f'FORCE {a.force}\n')
        tee.write(f'EVALUATION_ONLY {a.evaluation_only}\n')
        if not a.evaluation_only:
            formal_total = int(main_cfg['stages']['formal']['train_size'])
            formal_families = list(main_cfg['experiment']['families'])
            tee.write(f'FORMAL_TARGET_ROWS {formal_total}\n')
            tee.write(f'FORMAL_TARGET_PER_FAMILY {formal_total // len(formal_families)}\n')
        tee.write(f'START_TIME {dt.datetime.now().isoformat(timespec="seconds")}\n')

        if a.force:
            for cfgp in CONFIGS:
                c = load_yaml(cfgp)
                p = artifact_root / dataset / str(c['protocol']['name'])
                if p.exists():
                    tee.write(f'REMOVE {p}\n')
                    shutil.rmtree(p)
            pair_root = artifact_root / dataset / '_paired' / main_cfg['protocol']['paired_group_id']
            if pair_root.exists():
                tee.write(f'REMOVE {pair_root}\n')
                shutil.rmtree(pair_root)

        tee.write('========== PREFLIGHT ALL START ==========\n')
        for cfg in CONFIGS:
            run([sys.executable, str(ROOT / 'scripts' / 'preflight.py'), '--config', str(cfg)], tee)
        tee.write('========== PREFLIGHT ALL DONE ==========\n')

        run([sys.executable, str(ROOT / 'scripts' / 'check_train_grid.py')], tee)

        if not a.evaluation_only:
            cfg40 = ROOT / 'configs' / 'protocol_s40.yaml'
            for stage in ('development', 'formal'):
                cmd = [sys.executable, str(ROOT / 'scripts' / 'build_protocol.py'), '--config', str(cfg40), '--stage', stage]
                if a.force:
                    cmd.append('--force')
                run(cmd, tee)
            run([sys.executable, str(ROOT / 'scripts' / 'validate_formal_balance.py'), '--config', str(cfg40)], tee)

        tee.write('========== FREEZE PAIRED EVALUATIONS START ==========\n')
        # s40 first establishes the frozen reference pairing; all others resolve against it.
        for s in ('40', '20', '60', '90'):
            cfg = ROOT / 'configs' / f'protocol_s{s}.yaml'
            cmd = [sys.executable, str(ROOT / 'scripts' / 'build_protocol.py'), '--config', str(cfg), '--stage', 'evaluation']
            if a.force:
                cmd.append('--force')
            run(cmd, tee)
        tee.write('========== FREEZE PAIRED EVALUATIONS DONE ==========\n')

        run([sys.executable, str(ROOT / 'scripts' / 'validate_paired_evaluation.py')], tee)

        tee.write('========== SUPPORT SUBSET AUDIT START ==========\n')
        for s in ('20', '40', '60', '90'):
            cfg = ROOT / 'configs' / f'protocol_s{s}.yaml'
            run([sys.executable, str(ROOT / 'scripts' / 'audit_support_subsets.py'), '--config', str(cfg), '--stage', 'evaluation'], tee)
        tee.write('========== SUPPORT SUBSET AUDIT DONE ==========\n')

        tee.write('========== STATUS ALL ==========\n')
        for cfg in CONFIGS:
            run([sys.executable, str(ROOT / 'scripts' / 'status.py'), '--config', str(cfg)], tee)

        tee.write('EVALUATION_ACCESS DIRECT_UNLOCKED\n')
        tee.write('MULTI-SEVERITY PROTOCOL BUILD PASS\n')
        tee.write(f'END_TIME {dt.datetime.now().isoformat(timespec="seconds")}\n')
    except Exception as e:
        tee.write(f'MULTI-SEVERITY PROTOCOL BUILD FAILED: {type(e).__name__}: {e}\n')
        tee.write(f'END_TIME {dt.datetime.now().isoformat(timespec="seconds")}\n')
        raise
    finally:
        tee.close()


if __name__ == '__main__':
    main()
