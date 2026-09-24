#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts._common import setup, run_dir, default_tag
from srd.contract import assert_cache_ready
from srd.backbone import load_patchstg_adapter
from srd.patch_layout import extract_patch_layout
from srd.utils import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="development", choices=["development", "formal"])
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--patch-size", type=int, default=None, help="Explicit PatchSTG spatial patch width in reordered slots (debug override)")
    ap.add_argument("--patch-count", type=int, default=None, help="Explicit number of PatchSTG spatial patches (debug override)")
    a = ap.parse_args()
    cfg = setup(a.stage)
    assert_cache_ready(cfg, a.stage)
    tag = a.run_tag or default_tag("patchdiag_v060")
    out = run_dir(cfg, a.stage, tag) / "patch_layout"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter, cp, ck, source = load_patchstg_adapter(cfg, a.stage, device, a.checkpoint)
    layout = extract_patch_layout(adapter.model, ck, backbone_root=cfg["project"]["backbone_root"], patch_size_override=a.patch_size, patch_count_override=a.patch_count)
    payload = layout.to_jsonable()
    payload.update({"checkpoint": str(cp), "checkpoint_source": source})
    write_json(out / "patch_layout.json", payload)
    rows = []
    for pid, nodes in enumerate(layout.groups):
        for j, node in enumerate(nodes):
            rows.append({"patch_id": pid, "position_in_patch": j, "node_id": int(node)})
    pd.DataFrame(rows).to_csv(out / "patch_nodes.csv", index=False)
    print(json.dumps({k: v for k, v in payload.items() if k != "patches"}, indent=2))
    print(f"PATCH_LAYOUT={out / 'patch_layout.json'}")


if __name__ == "__main__":
    main()
