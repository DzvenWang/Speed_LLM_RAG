#!/usr/bin/env python3
"""Post-run verification for STDN_Special v4."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

RE_HEADER = re.compile(r"stage=\w+\s+model=STDN.*?batch_size=(\d+)")
RE_EPOCH = re.compile(r"epoch=(\d+)\s+train_loss=.*?\s+val_MAE=.*?\s+val_support_MAE=.*?\s+lr=([0-9.eE+-]+)")
RE_EARLY = re.compile(r"early_stop\s+epoch=(\d+)\s+patience=(\d+)")
RE_OPT = re.compile(r"\[STDN_SPECIAL_EFFECTIVE\]\s+optimizer=\S+\s+lr=([0-9.eE+-]+)\s+weight_decay=([0-9.eE+-]+)")
RE_RECIPE = re.compile(
    r"\[STDN_SPECIAL_EFFECTIVE\]\s+recipe\s+batch_size=(\d+)\s+patience=(\d+)\s+max_epochs=(\d+)\s+"
    r"lr=([0-9.eE+-]+)\s+weight_decay=([0-9.eE+-]+)\s+fallback_batch_sizes=\[\]"
)


def close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-12, abs(b) * 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "configs" / "experiments.json"))
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())[a.experiment]
    text = Path(a.log).read_text(errors="replace")
    errors = []
    observed = {}

    recipes = RE_RECIPE.findall(text)
    if not recipes:
        errors.append("no v4 effective recipe marker found")
    else:
        bs, pat, epochs, lr, wd = recipes[-1]
        observed["recipe"] = {"batch_size": int(bs), "patience": int(pat), "max_epochs": int(epochs), "lr": float(lr), "weight_decay": float(wd)}
        if int(bs) != int(cfg["batch_size"]): errors.append(f"recipe batch={bs}, requested={cfg['batch_size']}")
        if int(pat) != int(cfg["patience"]): errors.append(f"recipe patience={pat}, requested={cfg['patience']}")
        if int(epochs) != int(cfg["epochs"]): errors.append(f"recipe max_epochs={epochs}, requested={cfg['epochs']}")
        if not close(float(lr), float(cfg["learning_rate"])): errors.append(f"recipe lr={lr}, requested={cfg['learning_rate']}")
        if not close(float(wd), float(cfg["weight_decay"])): errors.append(f"recipe wd={wd}, requested={cfg['weight_decay']}")

    headers = RE_HEADER.findall(text)
    if headers:
        observed["runner_batch_size"] = int(headers[-1])
        if int(headers[-1]) != int(cfg["batch_size"]):
            errors.append(f"runner batch_size={headers[-1]} but requested={cfg['batch_size']}")
    else:
        errors.append("no STDN runner batch_size header found")

    opts = RE_OPT.findall(text)
    if opts:
        lr, wd = map(float, opts[-1])
        observed["optimizer_lr"] = lr
        observed["optimizer_weight_decay"] = wd
        if not close(lr, float(cfg["learning_rate"])): errors.append(f"optimizer lr={lr}, requested={cfg['learning_rate']}")
        if not close(wd, float(cfg["weight_decay"])): errors.append(f"optimizer wd={wd}, requested={cfg['weight_decay']}")
    else:
        errors.append("no effective optimizer audit found")

    epoch_lines = RE_EPOCH.findall(text)
    if epoch_lines:
        observed["first_epoch_logged_lr"] = float(epoch_lines[0][1])
        # The line is emitted after scheduler.step(), so a scheduler may already
        # have changed LR. Do not use it as the primary initialization check.
    else:
        errors.append("no completed epoch line found")

    early = RE_EARLY.findall(text)
    if early:
        observed["early_stop_epoch"] = int(early[-1][0])
        observed["early_stop_patience"] = int(early[-1][1])
        if int(early[-1][1]) != int(cfg["patience"]):
            errors.append(f"early-stop patience={early[-1][1]} but requested={cfg['patience']}")
    else:
        observed["early_stop_patience"] = None

    result = {"experiment": a.experiment, "requested": cfg, "observed": observed, "errors": errors, "status": "PASS" if not errors else "FAIL"}
    print("STDN_SPECIAL_EFFECTIVE_VERIFICATION=" + json.dumps(result, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
