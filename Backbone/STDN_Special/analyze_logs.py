#!/usr/bin/env python3
"""Parse verified STDN_Special v4 logs into an attribution table."""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

RE_EXP = re.compile(r"experiment:\s*([A-Za-z0-9_.-]+)")
RE_REQ = re.compile(r"REQUESTED_CONFIG:\s*(\{.*\})")
RE_EPOCH = re.compile(r"epoch=(\d+)\s+train_loss=([0-9.eE+-]+)\s+val_MAE=([0-9.eE+-]+)\s+val_support_MAE=([0-9.eE+-]+)\s+lr=([0-9.eE+-]+)")
RE_HEADER = re.compile(r"stage=\w+\s+model=STDN.*?batch_size=(\d+)")
RE_EARLY = re.compile(r"early_stop\s+epoch=(\d+)\s+patience=(\d+)")
RE_OPT = re.compile(r"\[STDN_SPECIAL_EFFECTIVE\]\s+optimizer=\S+\s+lr=([0-9.eE+-]+)\s+weight_decay=([0-9.eE+-]+)")
RE_RECIPE = re.compile(r"\[STDN_SPECIAL_EFFECTIVE\]\s+recipe\s+batch_size=(\d+)\s+patience=(\d+)\s+max_epochs=(\d+)")
RE_VERIFY = re.compile(r"STDN_SPECIAL_EFFECTIVE_VERIFICATION=(\{.*\})")


def parse(path: Path):
    text = path.read_text(errors="replace")
    rows = []
    for m in RE_EPOCH.finditer(text):
        rows.append({"epoch": int(m.group(1)), "train": float(m.group(2)), "val": float(m.group(3)), "support": float(m.group(4)), "lr_epoch": float(m.group(5))})
    if not rows:
        return None
    best = min(rows, key=lambda x: x["val"])
    expm = RE_EXP.findall(text)
    reqm = RE_REQ.findall(text)
    headers = RE_HEADER.findall(text)
    early = RE_EARLY.findall(text)
    opt = RE_OPT.findall(text)
    recipe = RE_RECIPE.findall(text)
    ver = RE_VERIFY.findall(text)
    requested = json.loads(reqm[-1]) if reqm else {}
    verification = json.loads(ver[-1]) if ver else {}
    return {
        "experiment": expm[-1] if expm else path.stem,
        "log": path.name,
        "verified": verification.get("status", "UNVERIFIED"),
        "requested_batch": requested.get("batch_size", ""),
        "effective_batch": int(headers[-1]) if headers else "",
        "requested_patience": requested.get("patience", ""),
        "effective_patience": int(early[-1][1]) if early else (int(recipe[-1][1]) if recipe else ""),
        "optimizer_lr": float(opt[-1][0]) if opt else "",
        "optimizer_wd": float(opt[-1][1]) if opt else "",
        "epochs_run": max(x["epoch"] for x in rows),
        "best_epoch": best["epoch"],
        "best_val_MAE": best["val"],
        "support_at_best": best["support"],
        "train_at_best": best["train"],
        "generalization_gap": best["val"] - best["train"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="Logs or directories. Defaults to STDN_Special/logs.")
    a = ap.parse_args()
    base = Path(__file__).resolve().parent
    items = [Path(x) for x in a.paths] if a.paths else [base / "logs"]
    logs = []
    for p in items:
        if p.is_file(): logs.append(p)
        elif p.is_dir(): logs.extend([x for x in p.rglob("*.log") if x.is_file()])
    parsed = []
    for p in logs:
        try:
            r = parse(p)
            if r: parsed.append(r)
        except Exception as exc:
            print(f"WARN parse {p}: {exc}")
    parsed.sort(key=lambda r: (r["verified"] != "PASS", r["best_val_MAE"]))

    out_csv = base / "artifacts" / "diagnostic_summary_v4.csv"
    out_md = base / "artifacts" / "diagnostic_summary_v4.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["experiment","verified","requested_batch","effective_batch","requested_patience","effective_patience","optimizer_lr","optimizer_wd","epochs_run","best_epoch","best_val_MAE","support_at_best","train_at_best","generalization_gap","log"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(parsed)

    lines = [
        "| Experiment | Verified | Batch req/eff | Patience req/eff | LR | WD | Best epoch | Val MAE | Support MAE | Gap |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in parsed:
        lines.append(
            f"| {r['experiment']} | {r['verified']} | {r['requested_batch']}/{r['effective_batch']} | "
            f"{r['requested_patience']}/{r['effective_patience'] or '-'} | {r['optimizer_lr']} | {r['optimizer_wd']} | "
            f"{r['best_epoch']} | {r['best_val_MAE']:.6f} | {r['support_at_best']:.6f} | {r['generalization_gap']:.6f} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {out_csv}\nWrote {out_md}")


if __name__ == "__main__":
    main()
