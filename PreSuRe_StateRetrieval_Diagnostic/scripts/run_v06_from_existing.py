#!/usr/bin/env python3
"""Run only the v0.6 forecast-utility stage on an existing v0.5.x diagnostic run.

v0.6.1 adds live console logging plus persistent stage logs under project.log_root.
"""
from pathlib import Path
import argparse, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from srd.config import load_config


def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _emit(handles, text):
    print(text, end="", flush=True)
    for h in handles:
        h.write(text)
        h.flush()


def call(cmd, main_log, stage_log):
    cmd = [str(x) for x in cmd]
    stage_log = Path(stage_log)
    stage_log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    banner = f"\n[{_ts()}] >> {' '.join(cmd)}\n"
    _emit([main_log], banner)
    with stage_log.open("w", encoding="utf-8", errors="replace") as sl:
        sl.write(banner); sl.flush()
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            _emit([main_log, sl], line)
        rc = p.wait()
        elapsed = time.time() - started
        trailer = f"[{_ts()}] STAGE_RETURN_CODE={rc} ELAPSED_SEC={elapsed:.1f}\n"
        _emit([main_log, sl], trailer)
        if rc:
            raise subprocess.CalledProcessError(rc, cmd)


def main():
    ap = argparse.ArgumentParser(description="Upgrade an existing v0.5.x run with v0.6 Forecast-Utility Diagnostic")
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--patch-size", type=int, default=None)
    ap.add_argument("--patch-count", type=int, default=None)
    ap.add_argument("--transport", default="affine_alignment")
    ap.add_argument("--scopes", default="1,3,6,12")
    ap.add_argument("--k-values", default="1,5,10,20,50")
    ap.add_argument("--same-episode-pool-cap", type=int, default=512)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    a = ap.parse_args()

    cfg = load_config(ROOT)
    dataset = cfg["experiment"]["subgraph"]
    run = Path(cfg["project"]["exp_root"]) / dataset / a.stage / a.run_tag
    logs = Path(cfg["project"]["log_root"]) / dataset / a.stage / a.run_tag
    logs.mkdir(parents=True, exist_ok=True)
    master_log_path = logs / "v06_upgrade.log"

    retrieval = run / "patch_retrieval" / "patch_retrieval_topk.npz"
    if not retrieval.exists():
        raise FileNotFoundError(f"Existing retrieval file not found: {retrieval}")
    needed = [
        run / "patch_layout" / "patch_layout.json",
        run / "patch_restoration_oracle" / "summary.json",
        run / "transport_realizable" / "summary.json",
    ]
    if not any(run.glob("completion_oracle_*/summary.json")):
        needed.append(run / "completion_oracle_*" / "summary.json")
    missing = [str(p) for p in needed if "*" not in str(p) and not p.exists()]
    if missing:
        raise RuntimeError(f"Existing run is incomplete; missing: {missing}")

    common = ["--stage", a.stage, "--run-tag", a.run_tag]
    ck = [] if not a.checkpoint else ["--checkpoint", a.checkpoint]
    layout = []
    if a.patch_size is not None:
        layout += ["--patch-size", str(a.patch_size)]
    if a.patch_count is not None:
        layout += ["--patch-count", str(a.patch_count)]

    with master_log_path.open("w", encoding="utf-8", errors="replace") as ml:
        _emit([ml], (
            f"[{_ts()}] Patch-Level Forecast-Utility Diagnostic v0.6.1 upgrade\n"
            f"RUN_TAG={a.run_tag}\n"
            f"DATASET={dataset}\n"
            f"STAGE={a.stage}\n"
            f"TRANSPORT={a.transport}\n"
            f"SCOPES={a.scopes}\n"
            f"K_VALUES={a.k_values}\n"
            f"SAME_EPISODE_POOL_CAP={a.same_episode_pool_cap}\n"
            f"EVAL_BATCH_SIZE={a.eval_batch_size}\n"
            f"RUN_DIR={run}\n"
            f"LOG_DIR={logs}\n"
            f"RETRIEVAL={retrieval}\n"
        ))
        try:
            call([
                sys.executable, ROOT / "scripts" / "06_run_forecast_utility_diagnostic.py",
                "--retrieval", retrieval, *common, *ck, *layout,
                "--transport", a.transport, "--scopes", a.scopes, "--k-values", a.k_values,
                "--same-episode-pool-cap", str(a.same_episode_pool_cap),
                "--eval-batch-size", str(a.eval_batch_size),
            ], ml, logs / "06_forecast_utility_oracle.log")
            call([
                sys.executable, ROOT / "scripts" / "06_summarize_patch_diagnostic.py", run
            ], ml, logs / "07_summary.log")
            _emit([ml], (
                f"\n[{_ts()}] V0.6_FORECAST_UTILITY_UPGRADE_COMPLETE\n"
                f"RUN={run}\nLOGS={logs}\n"
            ))
        except Exception as e:
            _emit([ml], (
                f"\n[{_ts()}] V0.6_FORECAST_UTILITY_UPGRADE_FAILED "
                f"{type(e).__name__}: {e}\nLOGS={logs}\n"
            ))
            raise


if __name__ == "__main__":
    main()
