#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from srd.config import load_config


def _write_line(handles, line: str):
    print(line, end="", flush=True)
    for h in handles:
        h.write(line); h.flush()


def run(cmd, main_log, stage_log_path: Path):
    cmd = list(map(str, cmd))
    stage_log_path.parent.mkdir(parents=True, exist_ok=True)
    banner = "\n>> " + " ".join(cmd) + "\n"
    _write_line([main_log], banner)
    with stage_log_path.open("w", encoding="utf-8", errors="replace") as stage_log:
        stage_log.write(banner); stage_log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            _write_line([main_log, stage_log], line)
        rc = proc.wait()
        if rc != 0:
            msg = f"\n[ERROR] command failed with exit code {rc}: {' '.join(cmd)}\n"
            _write_line([main_log, stage_log], msg)
            raise subprocess.CalledProcessError(rc, cmd)


def latest_v03(cfg, stage):
    base = Path(cfg["project"]["exp_root"]) / cfg["experiment"]["subgraph"] / stage
    if not base.exists(): return None
    runs = sorted([p for p in base.glob("srd_v03.*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for p in runs:
        if (p / "retrieval" / "retrieval_topk.npz").exists() and (p / "window_ceiling_sweep" / "summary.json").exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="PreSuRe State-Retrieval Diagnostic v0.4 — Realizable Utility Benchmark")
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--source-run", default=None, help="Completed SRD v0.3 experiment. Default: latest complete v0.3 run.")
    ap.add_argument("--max-k", type=int, default=50)
    ap.add_argument("--scopes", default="1,2,3,4,6")
    ap.add_argument("--protocols", default="blocked,random")
    ap.add_argument("--models", default="similarity_rule,logistic_benefit,ridge_utility,histgb_utility,mlp_utility")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--calib-fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=10)
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args()

    cfg = load_config(ROOT)
    source = Path(a.source_run).resolve() if a.source_run else latest_v03(cfg, a.stage)
    if source is None:
        raise SystemExit("No complete srd_v03.* source run found. Pass --source-run or run scripts/run_v03_source_pipeline.py first.")

    tag = time.strftime("srd_v04.%Y%m%d_%H%M%S")
    dataset = cfg["experiment"]["subgraph"]
    base = Path(cfg["project"]["exp_root"]) / dataset / a.stage / tag
    log_dir = Path(cfg["project"]["log_root"]) / dataset / a.stage / tag
    base.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    main_log_path = log_dir / "run.log"
    out = base / "realizable_benchmark"

    run_meta = {
        "version": "0.4.0", "tag": tag, "dataset": dataset, "stage": a.stage,
        "source_run": str(source), "max_k": a.max_k, "scopes": a.scopes,
        "protocols": a.protocols, "models": a.models, "folds": a.folds,
        "calib_fraction": a.calib_fraction, "seed": a.seed,
        "experiment_dir": str(base), "log_dir": str(log_dir), "argv": sys.argv,
    }
    (log_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    with main_log_path.open("w", encoding="utf-8", errors="replace") as main_log:
        header = (
            "PreSuRe State-Retrieval Diagnostic v0.4.0 — Realizable Utility Benchmark\n"
            f"Run tag      : {tag}\nDataset      : {dataset}\nStage        : {a.stage}\n"
            f"Source v0.3  : {source}\nMax K        : {a.max_k}\nScopes       : {a.scopes}\n"
            f"Protocols    : {a.protocols}\nModels       : {a.models}\nFolds        : {a.folds}\n"
            f"Experiment   : {base}\nLog directory: {log_dir}\n"
        )
        _write_line([main_log], header)
        try:
            pre = [sys.executable, ROOT / "scripts" / "08_realizable_preflight.py",
                   "--stage", a.stage, "--source-run", source, "--max-k", str(a.max_k), "--scopes", a.scopes]
            run(pre, main_log, log_dir / "00_realizable_preflight.log")
            if a.preflight_only:
                _write_line([main_log], f"\nSRD v0.4 PREFLIGHT COMPLETE\nSource run: {source}\nLogs: {log_dir}\n")
                return
            cmd = [sys.executable, ROOT / "scripts" / "08_run_realizable_benchmark.py",
                   "--source-run", source, "--out", out, "--max-k", str(a.max_k),
                   "--scopes", a.scopes, "--protocols", a.protocols, "--models", a.models,
                   "--folds", str(a.folds), "--calib-fraction", str(a.calib_fraction), "--seed", str(a.seed)]
            run(cmd, main_log, log_dir / "01_realizable_benchmark.log")
            summary = out / "summary.json"
            comparison = out / "model_comparison.csv"
            footer = (
                "\nSRD v0.4.0 COMPLETE\n"
                f"Source run      : {source}\nRun directory   : {base}\n"
                f"Realizable output: {out}\nModel comparison: {comparison}\nSummary          : {summary}\n"
                f"Log directory   : {log_dir}\nMain log         : {main_log_path}\n"
            )
            _write_line([main_log], footer)
        except Exception as exc:
            _write_line([main_log], f"\nSRD v0.4 FAILED: {type(exc).__name__}: {exc}\nLogs preserved at: {log_dir}\n")
            raise


if __name__ == "__main__":
    main()
