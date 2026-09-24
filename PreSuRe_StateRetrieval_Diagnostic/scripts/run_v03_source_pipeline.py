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
        h.write(line)
        h.flush()


def run(cmd, main_log, stage_log_path: Path):
    cmd = list(map(str, cmd))
    stage_log_path.parent.mkdir(parents=True, exist_ok=True)
    banner = "\n>> " + " ".join(cmd) + "\n"
    _write_line([main_log], banner)
    with stage_log_path.open("w", encoding="utf-8", errors="replace") as stage_log:
        stage_log.write(banner)
        stage_log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            _write_line([main_log, stage_log], line)
        rc = proc.wait()
        if rc != 0:
            msg = f"\n[ERROR] command failed with exit code {rc}: {' '.join(cmd)}\n"
            _write_line([main_log, stage_log], msg)
            raise subprocess.CalledProcessError(rc, cmd)


def _parse_scopes(text, cfg):
    if text:
        vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    else:
        vals = [int(x) for x in cfg.get("ceiling_sweep", {}).get("recent_steps", [1,2,3,4,6])]
    H = int(cfg["experiment"]["history"])
    return sorted(set(max(1, min(H, x)) for x in vals))


def main():
    ap = argparse.ArgumentParser(description="PreSuRe State-Retrieval Diagnostic v0.3")
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--queries", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--max-k", type=int, default=50)
    ap.add_argument("--scopes", default=None, help="Comma-separated recent suffix lengths; default 1,2,3,4,6")
    ap.add_argument("--skip-suffix-oracle", action="store_true")
    ap.add_argument("--skip-utility-datasets", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args()

    cfg = load_config(ROOT)
    scopes = _parse_scopes(a.scopes, cfg)
    transport = str(cfg.get("ceiling_sweep", {}).get("transport", "raw_copy"))
    tag = time.strftime("srd_v03.%Y%m%d_%H%M%S")
    dataset = cfg["experiment"]["subgraph"]
    base = Path(cfg["project"]["exp_root"]) / dataset / a.stage / tag
    log_dir = Path(cfg["project"]["log_root"]) / dataset / a.stage / tag
    log_dir.mkdir(parents=True, exist_ok=True)
    main_log_path = log_dir / "run.log"

    common = ["--stage", a.stage]
    ck = [] if not a.checkpoint else ["--checkpoint", a.checkpoint]
    q = [] if a.queries is None else ["--queries", str(a.queries)]
    mw = [] if a.max_windows is None else ["--max-windows", str(a.max_windows)]

    run_meta = {
        "version": "0.3.0",
        "tag": tag,
        "dataset": dataset,
        "stage": a.stage,
        "queries": a.queries,
        "max_windows": a.max_windows,
        "max_k": a.max_k,
        "scopes": scopes,
        "transport": transport,
        "checkpoint_override": a.checkpoint,
        "preflight_only": a.preflight_only,
        "experiment_dir": str(base),
        "log_dir": str(log_dir),
        "argv": sys.argv,
    }
    (log_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    with main_log_path.open("w", encoding="utf-8", errors="replace") as main_log:
        header = (
            "PreSuRe State-Retrieval Diagnostic v0.3.0\n"
            f"Run tag      : {tag}\n"
            f"Dataset      : {dataset}\n"
            f"Stage        : {a.stage}\n"
            f"Transport    : {transport}\n"
            f"Scope sweep  : {scopes}\n"
            f"Max K        : {a.max_k}\n"
            f"Experiment   : {base}\n"
            f"Log directory: {log_dir}\n"
        )
        _write_line([main_log], header)

        try:
            run([sys.executable, ROOT / "scripts" / "preflight.py", *common, *ck], main_log, log_dir / "00_preflight.log")
            if a.preflight_only:
                _write_line([main_log], f"\nSRD v0.3 PREFLIGHT COMPLETE\nLog directory: {log_dir}\nMain log     : {main_log_path}\n")
                return

            run([sys.executable, ROOT / "scripts" / "00_check_backbone.py", *common, *ck, "--queries", "32"], main_log, log_dir / "01_check_backbone.log")
            run([sys.executable, ROOT / "scripts" / "01_build_state_memory.py", *common, *ck, "--run-tag", tag, *mw], main_log, log_dir / "02_build_state_memory.log")
            memory = base / "memory" / "state_memory.npz"

            if not a.skip_suffix_oracle:
                run([sys.executable, ROOT / "scripts" / "02_run_patch_oracle.py", *common, *ck, "--run-tag", tag, *q], main_log, log_dir / "03_recent_suffix_oracle.log")

            run([sys.executable, ROOT / "scripts" / "03_run_similarity_retrieval.py", "--memory", memory, *common, *ck, "--run-tag", tag, *q], main_log, log_dir / "04_recent_focused_retrieval.log")
            retrieval = base / "retrieval" / "retrieval_topk.npz"

            run([
                sys.executable, ROOT / "scripts" / "05_run_window_ceiling_sweep.py",
                "--retrieval", retrieval,
                "--stage", a.stage,
                "--run-tag", tag,
                "--max-k", str(a.max_k),
                "--recent-steps", ",".join(map(str, scopes)),
                *ck,
            ], main_log, log_dir / "05_window_rawcopy_ceiling_sweep.log")

            if not a.skip_utility_datasets:
                for idx, L in enumerate(scopes, 1):
                    scope = f"recent{L}"
                    ce = base / f"oracle_{transport}_{scope}" / "candidate_eval.npz"
                    utility = base / f"oracle_{transport}_{scope}" / "utility_dataset.csv"
                    run([
                        sys.executable, ROOT / "scripts" / "06_build_utility_dataset.py",
                        "--retrieval", retrieval,
                        "--candidate-eval", ce,
                        "--out", utility,
                    ], main_log, log_dir / f"06_{idx:02d}_utility_{transport}_{scope}.log")

            run([sys.executable, ROOT / "scripts" / "07_summarize.py", base], main_log, log_dir / "07_summarize.log")
            footer = (
                "\nSRD v0.3.0 COMPLETE\n"
                f"Run directory: {base}\n"
                f"Log directory: {log_dir}\n"
                f"Main log     : {main_log_path}\n"
            )
            _write_line([main_log], footer)
        except Exception as exc:
            _write_line([main_log], f"\nSRD FAILED: {type(exc).__name__}: {exc}\nLogs preserved at: {log_dir}\n")
            raise


if __name__ == "__main__":
    main()
