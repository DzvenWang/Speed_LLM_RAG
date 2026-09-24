#!/usr/bin/env python3
from pathlib import Path
import argparse, json, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from srd.config import load_config


def _write(handles, text):
    print(text, end="", flush=True)
    for h in handles:
        h.write(text); h.flush()


def run(cmd, main_log, stage_log):
    cmd = [str(x) for x in cmd]
    stage_log = Path(stage_log); stage_log.parent.mkdir(parents=True, exist_ok=True)
    banner = "\n>> " + " ".join(cmd) + "\n"
    _write([main_log], banner)
    with stage_log.open("w", encoding="utf-8", errors="replace") as sl:
        sl.write(banner); sl.flush()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
        assert p.stdout is not None
        for line in p.stdout:
            _write([main_log, sl], line)
        rc = p.wait()
        if rc:
            raise subprocess.CalledProcessError(rc, cmd)


def main():
    ap = argparse.ArgumentParser(description="PreSuRe Patch Retrieval + Predictive-Utility Selector v0.7.1")
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--patch-size", type=int, default=None)
    ap.add_argument("--patch-count", type=int, default=None)
    ap.add_argument("--queries", type=int, default=None)
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--scopes", default="1,3,6,12")
    ap.add_argument("--oracle-k-values", default="1,5,10,20,50")
    ap.add_argument("--oracle-transport", default="affine_alignment")
    ap.add_argument("--transport-methods", default="raw_copy,delta_anchor,affine_alignment")
    ap.add_argument("--weighted-k", default="5,10")
    ap.add_argument("--transport-oracle-k", type=int, default=20)
    ap.add_argument("--forecast-oracle-transport", default="affine_alignment")
    ap.add_argument("--forecast-oracle-k-values", default="1,5,10,20,50")
    ap.add_argument("--same-episode-pool-cap", type=int, default=512)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--selector-models", default="similarity_rule,logistic_gate,ridge_utility,histgb_utility,two_stage_histgb")
    ap.add_argument("--selector-protocol", default="expanding", choices=["expanding", "blocked"])
    ap.add_argument("--selector-cv-blocks", type=int, default=6)
    ap.add_argument("--selector-patch-caps", default="1,2,4,6,8,12,16")
    ap.add_argument("--skip-v07", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args()

    cfg = load_config(ROOT)
    tag = time.strftime("patchdiag_v070.%Y%m%d_%H%M%S")
    dataset = cfg["experiment"]["subgraph"]
    base = Path(cfg["project"]["exp_root"]) / dataset / a.stage / tag
    logs = Path(cfg["project"]["log_root"]) / dataset / a.stage / tag
    logs.mkdir(parents=True, exist_ok=True)
    main_log_path = logs / "run.log"

    common = ["--stage", a.stage, "--run-tag", tag]
    ck = [] if not a.checkpoint else ["--checkpoint", a.checkpoint]
    layout_args = []
    if a.patch_size is not None: layout_args += ["--patch-size", str(a.patch_size)]
    if a.patch_count is not None: layout_args += ["--patch-count", str(a.patch_count)]
    q = [] if a.queries is None else ["--queries", str(a.queries)]
    mw = [] if a.max_windows is None else ["--max-windows", str(a.max_windows)]
    meta = {
        "version": "0.7.1", "tag": tag, "dataset": dataset, "stage": a.stage,
        "queries": a.queries, "max_windows": a.max_windows, "top_k": a.top_k,
        "scopes": a.scopes, "oracle_k_values": a.oracle_k_values,
        "oracle_transport": a.oracle_transport, "transport_methods": a.transport_methods,
        "weighted_k": a.weighted_k, "transport_oracle_k": a.transport_oracle_k,
        "forecast_oracle_transport": a.forecast_oracle_transport,
        "forecast_oracle_k_values": a.forecast_oracle_k_values,
        "same_episode_pool_cap": a.same_episode_pool_cap,
        "checkpoint_override": a.checkpoint, "patch_size_override": a.patch_size,
        "patch_count_override": a.patch_count, "experiment_dir": str(base), "log_dir": str(logs),
        "argv": sys.argv,
    }
    (logs / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    with main_log_path.open("w", encoding="utf-8", errors="replace") as ml:
        _write([ml], (
            "PreSuRe Patch Retrieval + Predictive-Utility Selector v0.7.1\n"
            f"Run tag       : {tag}\nDataset       : {dataset}\nStage         : {a.stage}\n"
            f"Queries       : {a.queries or cfg['experiment']['diagnostic_queries']}\n"
            f"Train windows : {a.max_windows or cfg['patch_retrieval']['max_train_windows']}\n"
            f"Top-K         : {a.top_k}\nScopes        : {a.scopes}\n"
            f"Forecast O.   : {a.forecast_oracle_transport}\n"
            f"Experiment    : {base}\nLogs          : {logs}\n"
        ))
        try:
            run([sys.executable, ROOT / "scripts" / "preflight.py", "--stage", a.stage, *ck], ml, logs / "00_preflight.log")
            run([sys.executable, ROOT / "scripts" / "01_inspect_patch_layout.py", *common, *ck, *layout_args], ml, logs / "01_patch_layout.log")
            if a.preflight_only:
                _write([ml], "\nPATCH DIAGNOSTIC PREFLIGHT COMPLETE\n")
                return

            run([
                sys.executable, ROOT / "scripts" / "02_run_patch_restoration_oracle.py",
                *common, *ck, *layout_args, *q, "--scopes", a.scopes, "--eval-batch-size", str(a.eval_batch_size)
            ], ml, logs / "02_patch_restoration_oracle.log")

            run([
                sys.executable, ROOT / "scripts" / "03_run_patch_retrieval.py",
                *common, *ck, *layout_args, *q, *mw, "--top-k", str(a.top_k)
            ], ml, logs / "03_patch_retrieval.log")
            retrieval = base / "patch_retrieval" / "patch_retrieval_topk.npz"

            run([
                sys.executable, ROOT / "scripts" / "04_run_patch_completion_oracle.py",
                "--retrieval", retrieval, *common, *ck, *layout_args, "--transport", a.oracle_transport,
                "--scopes", a.scopes, "--k-values", a.oracle_k_values,
                "--eval-batch-size", str(a.eval_batch_size)
            ], ml, logs / "04_patch_completion_oracle.log")

            run([
                sys.executable, ROOT / "scripts" / "05_run_patch_transport_realizable.py",
                "--retrieval", retrieval, *common, *ck, *layout_args, "--scopes", a.scopes,
                "--methods", a.transport_methods, "--weighted-k", a.weighted_k,
                "--oracle-k", str(a.transport_oracle_k), "--eval-batch-size", str(a.eval_batch_size)
            ], ml, logs / "05_transport_realizable.log")

            run([
                sys.executable, ROOT / "scripts" / "06_run_forecast_utility_diagnostic.py",
                "--retrieval", retrieval, *common, *ck, *layout_args,
                "--transport", a.forecast_oracle_transport, "--scopes", a.scopes,
                "--k-values", a.forecast_oracle_k_values,
                "--same-episode-pool-cap", str(a.same_episode_pool_cap),
                "--eval-batch-size", str(a.eval_batch_size),
            ], ml, logs / "06_forecast_utility_oracle.log")

            run([sys.executable, ROOT / "scripts" / "06_summarize_patch_diagnostic.py", base], ml, logs / "07_summary.log")
            if not a.skip_v07:
                run([sys.executable, ROOT / "scripts" / "07_build_predictive_utility_dataset.py", "--run-dir", base, "--stage", a.stage, *ck], ml, logs / "08_v07_build_dataset.log")
                run([sys.executable, ROOT / "scripts" / "08_run_predictive_utility_selector.py", "--run-dir", base, "--stage", a.stage, *ck,
                     "--models", a.selector_models, "--scopes", a.scopes, "--max-k", str(a.top_k),
                     "--protocol", a.selector_protocol, "--cv-blocks", str(a.selector_cv_blocks),
                     "--patch-caps", a.selector_patch_caps, "--transport", a.forecast_oracle_transport,
                     "--eval-batch-size", str(a.eval_batch_size)], ml, logs / "09_v07_selector.log")
                run([sys.executable, ROOT / "scripts" / "09_summarize_v07.py", base], ml, logs / "10_v07_summary.log")
                run([sys.executable, ROOT / "scripts" / "10_fit_final_selector.py", base, "--primary-scope", "12", "--max-k", str(a.top_k), "--patch-caps", a.selector_patch_caps], ml, logs / "11_v07_final_fit.log")
            _write([ml], f"\nPRE SURE DIAGNOSTIC v0.7.1 COMPLETE\nRun directory: {base}\nLog directory: {logs}\n")
        except Exception as e:
            _write([ml], f"\nPATCH DIAGNOSTIC FAILED: {type(e).__name__}: {e}\nLogs preserved at: {logs}\n")
            raise


if __name__ == "__main__":
    main()
