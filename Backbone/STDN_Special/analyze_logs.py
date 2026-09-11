#!/usr/bin/env python3
"""Parse STDN diagnostic logs into a compact attribution table."""
from __future__ import annotations
import argparse
import csv
import re
from pathlib import Path

RE_EPOCH = re.compile(r"epoch=(\d+)\s+train_loss=([0-9.eE+-]+)\s+val_MAE=([0-9.eE+-]+)\s+val_support_MAE=([0-9.eE+-]+)\s+lr=([0-9.eE+-]+)")
RE_HEADER = re.compile(r"batch_size=(\d+).*?params=([0-9,]+)")
RE_RECIPE = re.compile(r"optimizer=\w+\s+lr=([0-9.eE+-]+).*?wd=([0-9.eE+-]+)")


def parse(path: Path):
    text = path.read_text(errors="replace")
    rows = []
    for m in RE_EPOCH.finditer(text):
        rows.append({
            "epoch": int(m.group(1)),
            "train": float(m.group(2)),
            "val": float(m.group(3)),
            "support": float(m.group(4)),
            "lr_epoch": float(m.group(5)),
        })
    if not rows:
        return None
    best = min(rows, key=lambda x: x["val"])
    h = RE_HEADER.search(text)
    recipe = RE_RECIPE.search(text)
    return {
        "log": path.name,
        "epochs_run": max(x["epoch"] for x in rows),
        "best_epoch": best["epoch"],
        "best_val_MAE": best["val"],
        "support_at_best": best["support"],
        "train_at_best": best["train"],
        "generalization_gap": best["val"] - best["train"],
        "batch_size": int(h.group(1)) if h else "",
        "recipe_lr": float(recipe.group(1)) if recipe else "",
        "recipe_wd": float(recipe.group(2)) if recipe else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="Log files or directories. Defaults to STDN_Special/logs and PBS output in cwd.")
    a = ap.parse_args()
    base = Path(__file__).resolve().parent
    items = [Path(x) for x in a.paths] if a.paths else [base / "logs", base]
    logs = []
    for p in items:
        if p.is_file():
            logs.append(p)
        elif p.is_dir():
            logs.extend([x for x in p.rglob("*") if x.is_file() and (x.suffix in {".log", ".out", ""} or "STDNSpecial" in x.name)])
    parsed = []
    seen = set()
    for p in logs:
        if p.resolve() in seen:
            continue
        seen.add(p.resolve())
        try:
            r = parse(p)
            if r:
                parsed.append(r)
        except Exception:
            pass
    parsed.sort(key=lambda r: r["best_val_MAE"])
    out_csv = base / "artifacts" / "diagnostic_summary.csv"
    out_md = base / "artifacts" / "diagnostic_summary.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["log", "batch_size", "recipe_lr", "recipe_wd", "epochs_run", "best_epoch", "best_val_MAE", "support_at_best", "train_at_best", "generalization_gap"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(parsed)
    lines = ["| Log | Batch | Best epoch | Val MAE | Support MAE | Train MAE | Gap |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in parsed:
        lines.append(f"| {r['log']} | {r['batch_size']} | {r['best_epoch']} | {r['best_val_MAE']:.6f} | {r['support_at_best']:.6f} | {r['train_at_best']:.6f} | {r['generalization_gap']:.6f} |")
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out_csv}\nWrote {out_md}")


if __name__ == "__main__":
    main()
