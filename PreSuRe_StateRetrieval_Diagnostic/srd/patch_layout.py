from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import inspect
import math
import re
import numpy as np


def _np1(x):
    if x is None:
        return None
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    a = np.asarray(x)
    return a.reshape(-1)


def _scalar(v):
    try:
        if hasattr(v, "item") and np.asarray(v).size == 1:
            v = v.item()
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return int(v)
    except Exception:
        return None
    return None


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else str(k)
        out[key] = v
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
    return out


def _walk_model_scalars(model, max_depth: int = 3) -> dict[str, int]:
    """Collect integer-like scalar attributes from model and submodules.

    Some Backbone adapters use the official PatchSTG constructor but do not copy
    spatial hyperparameters onto the top-level module. Walking submodules makes
    layout recovery robust to those wrappers without changing the frozen model.
    """
    out: dict[str, int] = {}
    seen = set()

    def rec(obj, prefix: str, depth: int):
        if obj is None or id(obj) in seen or depth > max_depth:
            return
        seen.add(id(obj))
        try:
            items = vars(obj).items()
        except Exception:
            return
        for k, v in items:
            key = f"{prefix}.{k}" if prefix else str(k)
            sv = _scalar(v)
            if sv is not None:
                out[key] = sv
                continue
            if isinstance(v, dict):
                for dk, dv in _flatten_dict(v, key).items():
                    dsv = _scalar(dv)
                    if dsv is not None:
                        out[dk] = dsv
                continue
            # Recurse only into nn.Module-like objects / light namespaces; avoid
            # tensors, arrays and large containers.
            mod = getattr(v, "_modules", None)
            if mod is not None or hasattr(v, "__dict__") and v.__class__.__module__ != "builtins":
                rec(v, key, depth + 1)

    rec(model, "model", 0)
    return out


def _lookup_scalar(model, ck, names):
    names = [str(n).lower() for n in names]
    # Top-level model attributes first.
    for n in names:
        for attr in dir(model):
            if attr.lower() == n:
                try:
                    val = _scalar(getattr(model, attr))
                    if val is not None:
                        return val, f"model.{attr}"
                except Exception:
                    pass

    # Then recursively inspect instantiated model/submodule state.
    for key, val in _walk_model_scalars(model).items():
        tail = key.split(".")[-1].lower()
        if tail in names:
            return int(val), key

    # Finally inspect every checkpoint/config dictionary, not only model_config.
    roots = []
    if isinstance(ck, dict):
        roots.append(("checkpoint", ck))
        if isinstance(ck.get("model_config"), dict):
            roots.insert(0, ("checkpoint.model_config", ck["model_config"]))
    for prefix, obj in roots:
        flat = _flatten_dict(obj)
        for key, val in flat.items():
            tail = key.split(".")[-1].lower()
            if tail in names:
                sv = _scalar(val)
                if sv is not None:
                    return sv, f"{prefix}.{key}"
    return None, None


def _parse_numeric_assignments(path: Path) -> dict[str, int]:
    """Parse simple key=value / key: value numeric config entries."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}
    out = {}
    # Works for .conf/.ini/.yaml and simple Python kwargs/default blocks.
    pat = re.compile(r"(?im)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|:)\s*['\"]?([0-9]+(?:\.0+)?)['\"]?\s*(?:[#;,].*)?$")
    for m in pat.finditer(text):
        try:
            out[m.group(1).lower()] = int(float(m.group(2)))
        except Exception:
            pass
    return out


def _vendor_profile_hint(backbone_root: str | Path | None, total_slots: int):
    """Recover the PatchSTG patch count from the pinned vendor profile.

    The Speed_LLM_RAG PatchSTG adapter is based on the official SD profile. The
    node count / KDTree padding is adapted at the adapter boundary, while the
    profile's *number of spatial patches* is the stable architectural quantity.
    Therefore, when sps/factors are not persisted in a checkpoint, use
    spn/factors from SD.conf to recover patch_count, then derive the actual slot
    width from the instantiated model's reo_all_idx length.
    """
    if not backbone_root:
        return None
    b = Path(backbone_root)
    candidates = [
        b / "vendor" / "PatchSTG" / "config" / "SD.conf",
        b / "vendor" / "PatchSTG" / "configs" / "SD.conf",
        b / "vendor" / "PatchSTG" / "config" / "SD.ini",
        b / "vendor" / "PatchSTG" / "configs" / "SD.ini",
    ]
    # Case-insensitive fallback for Windows/source-tree variations.
    vendor = b / "vendor" / "PatchSTG"
    if vendor.exists():
        try:
            candidates.extend(sorted(vendor.rglob("*SD*.conf")))
            candidates.extend(sorted(vendor.rglob("*SD*.ini")))
            candidates.extend(sorted(vendor.rglob("*SD*.yaml")))
            candidates.extend(sorted(vendor.rglob("*SD*.yml")))
        except Exception:
            pass

    seen = set()
    for path in candidates:
        try:
            rp = path.resolve()
        except Exception:
            rp = path
        if rp in seen or not path.exists() or not path.is_file():
            continue
        seen.add(rp)
        vals = _parse_numeric_assignments(path)
        spn = vals.get("spn")
        factors = vals.get("factors", vals.get("factor"))
        patch_num = vals.get("spatial_patch_num", vals.get("spa_patch_num", vals.get("patch_num")))
        if patch_num is None and spn and factors and factors > 0 and spn % factors == 0:
            patch_num = spn // factors
        if patch_num and patch_num > 0 and total_slots % patch_num == 0:
            return {
                "patch_size": int(total_slots // patch_num),
                "patch_count": int(patch_num),
                "source": f"vendor_profile:{path}",
                "profile_values": vals,
            }
    return None


@dataclass(frozen=True)
class PatchLayout:
    groups: tuple[np.ndarray, ...]
    patch_size_slots: int
    total_reordered_slots: int
    node_num: int
    source: str
    metadata: dict[str, Any]

    @property
    def patch_count(self) -> int:
        return len(self.groups)

    def to_jsonable(self):
        return {
            "patch_count": self.patch_count,
            "patch_size_slots": int(self.patch_size_slots),
            "total_reordered_slots": int(self.total_reordered_slots),
            "node_num": int(self.node_num),
            "source": self.source,
            "metadata": self.metadata,
            "patches": [
                {"patch_id": int(i), "node_count": int(len(g)), "nodes": [int(x) for x in g.tolist()]}
                for i, g in enumerate(self.groups)
            ],
        }


def _infer_patch_size(model, ck, total_slots: int, backbone_root=None, patch_size_override=None, patch_count_override=None):
    if patch_size_override is not None:
        ps = int(patch_size_override)
        if ps <= 0 or total_slots % ps != 0:
            raise RuntimeError(f"Invalid patch_size_override={ps} for reordered_slots={total_slots}")
        return ps, "explicit_patch_size_override", {"patch_size_override": ps}
    if patch_count_override is not None:
        pc = int(patch_count_override)
        if pc <= 0 or total_slots % pc != 0:
            raise RuntimeError(f"Invalid patch_count_override={pc} for reordered_slots={total_slots}")
        return total_slots // pc, "explicit_patch_count_override", {"patch_count_override": pc}

    direct, src = _lookup_scalar(model, ck, [
        "spatial_patch_size", "spa_patch_size", "spatial_patch_len", "spa_patch_len", "patch_size_slots",
    ])
    if direct and direct > 0 and total_slots % int(direct) == 0:
        return int(direct), src, {}

    sps, sps_src = _lookup_scalar(model, ck, ["sps"])
    factors, factors_src = _lookup_scalar(model, ck, ["factors", "factor"])
    spn, spn_src = _lookup_scalar(model, ck, ["spn"])

    meta = {"sps": sps, "sps_source": sps_src, "factors": factors, "factors_source": factors_src,
            "spn": spn, "spn_source": spn_src}
    if sps and factors and sps > 0 and factors > 0:
        ps = int(sps * factors)
        if total_slots % ps == 0:
            return ps, f"{sps_src}*{factors_src}", meta

    if spn and factors and spn > 0 and factors > 0:
        patch_count = int(spn // factors)
        if patch_count > 0 and total_slots % patch_count == 0:
            return int(total_slots // patch_count), f"slots/({spn_src}/{factors_src})", meta

    patch_count, pc_src = _lookup_scalar(model, ck, [
        "spatial_patch_num", "spa_patch_num", "spatial_patch_count", "spa_patch_count",
        "patch_num", "num_patches", "patch_count", "patches",
    ])
    if patch_count and patch_count > 0 and total_slots % patch_count == 0:
        meta.update({"patch_count_hint": patch_count, "patch_count_hint_source": pc_src})
        return int(total_slots // patch_count), f"slots/{pc_src}", meta

    vendor = _vendor_profile_hint(backbone_root, total_slots)
    if vendor is not None:
        meta.update({
            "vendor_profile_source": vendor["source"],
            "vendor_profile_patch_count": vendor["patch_count"],
            "vendor_profile_values": vendor["profile_values"],
        })
        return int(vendor["patch_size"]), f"slots/vendor_patch_count({vendor['patch_count']})", meta

    # Diagnostic payload helps debug non-standard adapters instead of hiding a guess.
    rel = {}
    for k, v in _walk_model_scalars(model).items():
        lk = k.lower()
        if any(t in lk for t in ("patch", "factor", "sps", "spn", "recur", "tree")):
            rel[k] = v
    raise RuntimeError(
        "Could not infer PatchSTG spatial patch size. The frozen model/checkpoint does not persist sps/factors "
        "and the pinned vendor SD profile could not be read. "
        f"total_reordered_slots={total_slots}; relevant_model_scalars={rel}. "
        "Use --patch-count or --patch-size only if you intentionally use a non-standard PatchSTG profile."
    )


def extract_patch_layout(model, ck, backbone_root=None, patch_size_override=None, patch_count_override=None) -> PatchLayout:
    """Recover the actual PatchSTG spatial patches from the frozen model."""
    if not hasattr(model, "reo_all_idx"):
        raise RuntimeError("PatchSTG model does not expose reo_all_idx; cannot recover exact spatial patches")

    reorder = _np1(getattr(model, "reo_all_idx"))
    if reorder is None or reorder.size == 0:
        raise RuntimeError("PatchSTG reo_all_idx is empty")

    node_num = int(getattr(model, "node_num", ck.get("model_config", {}).get("node_num", -1)))
    if node_num <= 0:
        vals = reorder[np.isfinite(reorder)]
        node_num = int(vals.max()) + 1 if vals.size else -1
    if node_num <= 0:
        raise RuntimeError("Could not infer PatchSTG node_num")

    patch_size, patch_size_source, meta = _infer_patch_size(
        model, ck, int(reorder.size), backbone_root=backbone_root,
        patch_size_override=patch_size_override, patch_count_override=patch_count_override,
    )
    patch_count = int(math.ceil(reorder.size / patch_size))

    ori = _np1(getattr(model, "ori_parts_idx", None))
    reo = _np1(getattr(model, "reo_parts_idx", None))
    use_mapping = ori is not None and reo is not None and ori.size == reo.size and ori.size >= node_num

    groups = []
    if use_mapping:
        pos_to_node = {}
        for p, n in zip(reo.astype(np.int64), ori.astype(np.int64)):
            if 0 <= p < reorder.size and 0 <= n < node_num:
                pos_to_node[int(p)] = int(n)
        for pid in range(patch_count):
            s, e = pid * patch_size, min((pid + 1) * patch_size, reorder.size)
            nodes = [pos_to_node[p] for p in range(s, e) if p in pos_to_node]
            groups.append(np.asarray(nodes, dtype=np.int64))
        source = "reo_parts_idx_to_ori_parts_idx"
    else:
        seen = set()
        for pid in range(patch_count):
            s, e = pid * patch_size, min((pid + 1) * patch_size, reorder.size)
            nodes = []
            for n in reorder[s:e].astype(np.int64):
                if 0 <= int(n) < node_num and int(n) not in seen:
                    seen.add(int(n)); nodes.append(int(n))
            groups.append(np.asarray(nodes, dtype=np.int64))
        source = "reo_all_idx_fallback"

    groups = tuple(g for g in groups if len(g) > 0)
    if not groups:
        raise RuntimeError("Recovered zero non-empty PatchSTG patches")

    all_nodes = np.concatenate(groups)
    uniq, counts = np.unique(all_nodes, return_counts=True)
    duplicated = uniq[counts > 1]
    missing = np.setdiff1d(np.arange(node_num, dtype=np.int64), uniq)
    extra = uniq[(uniq < 0) | (uniq >= node_num)]
    if duplicated.size or missing.size or extra.size:
        raise RuntimeError(
            "Recovered PatchSTG layout failed partition check: "
            f"patches={len(groups)} duplicated={duplicated[:20].tolist()} missing={missing[:20].tolist()} extra={extra[:20].tolist()}"
        )

    metadata = {
        **meta,
        "patch_size_source": patch_size_source,
        "used_restore_mapping": bool(use_mapping),
        "node_counts": [int(len(g)) for g in groups],
        "min_node_count": int(min(len(g) for g in groups)),
        "max_node_count": int(max(len(g) for g in groups)),
        "partition_verified": True,
    }
    return PatchLayout(groups=groups, patch_size_slots=int(patch_size), total_reordered_slots=int(reorder.size),
                       node_num=int(node_num), source=source, metadata=metadata)
