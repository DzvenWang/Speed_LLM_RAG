#!/usr/bin/env python3
"""Cross-platform SupportBench multi-severity rebuild with persistent logs.

Modes
-----
auto (default):
    Use a full canonical rebuild when prerequisite generated artifacts are
    missing; otherwise use the faster multi-severity incremental rebuild.
full:
    Run preflight and stages 01..07 for the requested subgraph.
incremental:
    Rebuild core/layouts, sensing views, resolved protocols, audit and API
    smoke test while reusing existing windows and support memory.

This entry point never hard-codes a Windows or Gadi path. Project paths are
resolved by the normal SupportBench config loader.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def emit(message: str, fp) -> None:
    print(message, flush=True)
    fp.write(message + "\n")
    fp.flush()


def run_command(cmd: list[str], fp, stage: str) -> None:
    emit(f"========== {stage} START ==========" , fp)
    emit("RUN " + " ".join(cmd), fp)
    t0 = time.time()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        fp.write(line)
        fp.flush()

    rc = proc.wait()
    elapsed = time.time() - t0
    if rc != 0:
        emit(f"========== {stage} FAILED rc={rc} elapsed={elapsed:.1f}s ==========", fp)
        raise subprocess.CalledProcessError(rc, cmd)
    emit(f"========== {stage} DONE elapsed={elapsed:.1f}s ==========", fp)


def stage_cmd(script: str, config: Path, subgraph: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "build" / script),
        "--config", str(config),
        "--subgraph", str(subgraph),
    ]


def preflight_cmd(config: Path) -> list[str]:
    return [sys.executable, str(ROOT / "preflight.py"), "--config", str(config)]


def resolve_subgraph_dir(config: Path, subgraph: str) -> Path:
    # Use the project's own portable config loader so ${SPEED_LLM_RAG_ROOT}
    # and inferred project roots behave identically on Windows and Gadi.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from supportbench.config import load_config

    cfg = load_config(str(config))
    return Path(cfg["project"]["benchmark_root"]) / subgraph


def prerequisites_present(sg_dir: Path) -> tuple[bool, list[Path]]:
    required = [
        sg_dir / "windows" / "train_full.npy",
        sg_dir / "windows" / "val_full.npy",
        sg_dir / "windows" / "test_full.npy",
        sg_dir / "windows" / "train_development.npy",
        sg_dir / "windows" / "val_development.npy",
        sg_dir / "windows" / "test_development.npy",
        sg_dir / "memory" / "memory_manifest.json",
        sg_dir / "memory" / "episode_window_starts_full.npy",
    ]
    missing = [p for p in required if not p.exists()]
    return len(missing) == 0, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "supportbench.yaml"))
    ap.add_argument("--subgraph", default="TraffiDent_Net_06")
    ap.add_argument("--mode", choices=["auto", "full", "incremental"], default="auto")
    ap.add_argument("--log-dir", default=str(ROOT / "logs"))
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    config = Path(args.config).resolve()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        log_path = Path(args.log_file)
        if not log_path.is_absolute():
            log_path = log_dir / log_path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"SupportBenchMultiSeverity.{args.subgraph}.{stamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    sg_dir = resolve_subgraph_dir(config, args.subgraph)
    ready, missing = prerequisites_present(sg_dir)
    effective_mode = args.mode
    if effective_mode == "auto":
        effective_mode = "incremental" if ready else "full"

    with log_path.open("w", encoding="utf-8", buffering=1) as fp:
        emit(f"LOG_FILE {log_path.resolve()}", fp)
        emit(f"PROJECT_ROOT {ROOT.parent.resolve()}", fp)
        emit(f"PROCESS_ROOT {ROOT.resolve()}", fp)
        emit(f"PYTHON {Path(sys.executable).resolve()}", fp)
        emit(f"CONFIG {config}", fp)
        emit(f"SUBGRAPH {args.subgraph}", fp)
        emit(f"REQUESTED_MODE {args.mode}", fp)
        emit(f"EFFECTIVE_MODE {effective_mode}", fp)
        emit(f"SUBGRAPH_DIR {sg_dir}", fp)
        emit(f"START_TIME {datetime.now().isoformat(timespec='seconds')}", fp)

        if missing:
            emit("MISSING_PREREQUISITES", fp)
            for p in missing:
                emit(f"  - {p}", fp)

        try:
            run_command(preflight_cmd(config), fp, "STAGE 00 preflight")

            if effective_mode == "full":
                stages = [
                    ("01_build_core.py", "STAGE 01 canonical core"),
                    ("02_build_windows.py", "STAGE 02 forecast windows"),
                    ("03_build_sensing_views.py", "STAGE 03 controlled sensing views"),
                    ("04_build_support_memory.py", "STAGE 04 train-only support memory"),
                    ("05_build_protocols.py", "STAGE 05 experimental protocols"),
                    ("06_audit.py", "STAGE 06 benchmark audit"),
                    ("07_smoke_api.py", "STAGE 07 canonical API smoke"),
                ]
            else:
                stages = [
                    ("01_build_core.py", "STAGE 01 canonical core"),
                    ("03_build_sensing_views.py", "STAGE 03 controlled sensing views"),
                    # Views/layout IDs may change when predictive structured
                    # generation changes, so resolved protocols are refreshed.
                    ("05_build_protocols.py", "STAGE 05 experimental protocols"),
                    ("06_audit.py", "STAGE 06 benchmark audit"),
                    ("07_smoke_api.py", "STAGE 07 canonical API smoke"),
                ]

            for script, label in stages:
                run_command(stage_cmd(script, config, args.subgraph), fp, label)

        except Exception as exc:
            emit(f"MULTI-SEVERITY SUPPORTBENCH BUILD FAILED: {type(exc).__name__}: {exc}", fp)
            emit(f"END_TIME {datetime.now().isoformat(timespec='seconds')}", fp)
            raise

        emit("MULTI-SEVERITY SUPPORTBENCH BUILD PASS", fp)
        emit(f"END_TIME {datetime.now().isoformat(timespec='seconds')}", fp)

    print(f"LOG SAVED: {log_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
