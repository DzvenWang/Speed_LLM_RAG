#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents if p.name == "Baseline"), Path(__file__).resolve().parents[1])
PROJECT = ROOT.parent

GROUPS = {
    "independent": ("GinAR", "RAFT", "RAST", "SARAF", "ReMFT"),
    "reference": ("ImputeFormer", "LRTC-TMCP"),
}
TARGETS = ("speed", "flow")
SEVERITIES = (20, 40, 60, 90)
RUNNERS = {
    "ImputeFormer": "imputeformer_runner.py",
    "LRTC-TMCP": "lrtc_tmcp_runner.py",
    "GinAR": "ginar_runner.py",
    "RAFT": "raft_runner.py",
    "RAST": "rast_runner.py",
    "SARAF": "saraf_runner.py",
    "ReMFT": "remft_runner.py",
}


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class TimestampTee:
    """Timestamp every logical line written by the smoke driver."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("w", encoding="utf-8", buffering=1)
        self._buffer = ""

    def _emit(self, line: str) -> None:
        rendered = f"[{timestamp()}] {line}\n"
        sys.__stdout__.write(rendered)
        sys.__stdout__.flush()
        self.f.write(rendered)
        self.f.flush()

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)
        return len(s)

    def flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""
        sys.__stdout__.flush()
        self.f.flush()

    def close(self) -> None:
        self.flush()
        self.f.close()


def emit_child_line(line: str, tee: TimestampTee) -> None:
    """Child stdout/stderr bypasses sys.stdout to avoid double timestamping."""
    tee._emit(line.rstrip("\r\n"))


def run(cmd, env, tee: TimestampTee):
    print("CMD:", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        emit_child_line(line, tee)
    rc = proc.wait()
    if rc:
        raise subprocess.CalledProcessError(rc, cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=GROUPS, required=True)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--backbone", default="STLAformer")
    ap.add_argument("--skip-evaluation-preflight", action="store_true")
    a = ap.parse_args()

    methods = GROUPS[a.group]
    total = len(methods) * 2
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "Independent" if a.group == "independent" else "ReferenceBackboneDependent"
    log = PROJECT / "Baseline_logs" / "smoke_primary" / a.group / f"BaselineSmoke{tag}.{stamp}.log"
    tee = TimestampTee(log)
    old = sys.stdout
    sys.stdout = tee
    passed = 0
    failed = []
    t0 = time.time()

    try:
        print(f"BASELINE_PRIMARY_SMOKE_START group={a.group} cases={total} seed={a.seed}")
        print(f"METHODS={list(methods)}")
        print(f"AGGREGATE_LOG={log}")

        for target in TARGETS:
            for method in methods:
                label = f"{method}/{target}"
                c0 = time.time()
                print("=" * 100)
                print("CASE START", label)
                print("=" * 100)

                env = os.environ.copy()
                env.update(
                    SPEED_LLM_RAG_ROOT=str(PROJECT),
                    TRAFFIDENT_TARGET=target,
                    PYTHONUNBUFFERED="1",
                    PYTHONDONTWRITEBYTECODE="1",
                    PYTHONUTF8="1",
                    PYTHONIOENCODING="utf-8",
                )

                try:
                    pre=[sys.executable, "scripts/preflight.py", "--stage", "formal", "--method", method]
                    if a.group == "reference": pre += ["--backbone", a.backbone]
                    run(pre, env, tee)
                    pf=[
                        sys.executable,"scripts/preflight_formal.py","--stage","formal",
                        "--method",method,"--seed",str(a.seed),
                    ]
                    if a.group == "reference": pf += ["--backbone", a.backbone]
                    run(pf,env,tee)
                    if method == "RAST":
                        run([sys.executable, "scripts/check_rast_runtime.py"], env, tee)
                    if method == "ReMFT":
                        run([sys.executable, "scripts/check_remft_runtime.py"], env, tee)
                    
                    cmd = [
                        sys.executable,
                        "-u",
                        str(ROOT / "baseline" / "runners" / RUNNERS[method]),
                        "--stage",
                        "formal",
                        "--seed",
                        str(a.seed),
                        "--smoke",
                    ]
                    if method == "RAST":
                        cmd += ["--variant", "native"]
                    if a.group == "reference":
                        cmd += ["--backbone", a.backbone]
                    run(cmd, env, tee)

                    if not a.skip_evaluation_preflight:
                        for sev in SEVERITIES:
                            ep=[sys.executable,"scripts/preflight.py","--stage","evaluation","--method",method,"--severity",str(sev)]
                            if a.group == "reference": ep += ["--backbone", a.backbone]
                            run(ep,env,tee)
                            ef=[sys.executable,"scripts/preflight_formal.py","--stage","evaluation","--method",method,"--seed",str(a.seed),"--severity",str(sev)]
                            if a.group == "reference": ef += ["--backbone", a.backbone]
                            run(ef,env,tee)

                    passed += 1
                    print(f"CASE PASS {label} seconds={time.time() - c0:.1f}")
                except Exception as exc:
                    failed.append((label, repr(exc)))
                    print(f"CASE FAIL {label}: {exc!r}")
                    if not a.continue_on_error:
                        raise

        print("=" * 100)
        print(
            f'PRIMARY_SMOKE_{"PASS" if not failed else "PARTIAL"} '
            f"group={a.group} {passed}/{total} seconds={time.time() - t0:.1f}"
        )
        print("MODE=train_smoke_plus_evaluation_contract_smoke")
        if failed:
            for label, err in failed:
                print("FAILED", label, err)
            raise SystemExit(1)
    finally:
        sys.stdout = old
        tee.close()
        print(f"[{timestamp()}] Aggregate smoke log: {log}", flush=True)


if __name__ == "__main__":
    main()
