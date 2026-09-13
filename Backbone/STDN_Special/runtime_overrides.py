#!/usr/bin/env python3
"""Narrow runtime overrides for STDN_Special v4.

Design goals
------------
1. Keep vendor/STDN frozen.
2. Do not edit the shared Backbone configuration on disk.
3. Override only cfg['models']['STDN'] after Backbone loads YAML.
4. Verify the resulting STDN recipe immediately before training.
5. Avoid v3's broad function-argument monkey patching (e.g. a generic
   parameter named ``batch`` is *not* a batch-size target).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


REQUEST_TO_BACKBONE = {
    "batch_size": "batch_size",
    "learning_rate": "learning_rate",
    "weight_decay": "weight_decay",
    "patience": "patience",
    "epochs": "max_epochs",
}


def expected_recipe(requested: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "batch_size": int(requested["batch_size"]),
        "learning_rate": float(requested["learning_rate"]),
        "weight_decay": float(requested["weight_decay"]),
        "patience": int(requested["patience"]),
        "max_epochs": int(requested["epochs"]),
    }


def apply_stdn_recipe(cfg_doc: Dict[str, Any], requested: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate one already-loaded Backbone config document in memory.

    The diagnostic must be exact.  We therefore disable Backbone's OOM batch
    fallback for STDN_Special instead of silently changing the requested batch
    size after training has started.
    """
    if not isinstance(cfg_doc, dict):
        raise TypeError("Backbone YAML did not produce a mapping")
    models = cfg_doc.get("models")
    if not isinstance(models, dict) or "STDN" not in models:
        raise KeyError("Backbone config has no models.STDN entry")
    mc = models["STDN"]
    if not isinstance(mc, dict):
        raise TypeError("Backbone models.STDN entry is not a mapping")

    exp = expected_recipe(requested)
    for key, value in exp.items():
        mc[key] = value
    mc["fallback_batch_sizes"] = []
    return cfg_doc


def _recipe_snapshot(mc: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "batch_size", "fallback_batch_sizes", "optimizer", "learning_rate",
        "adam_eps", "weight_decay", "scheduler", "step_size", "lr_gamma",
        "max_epochs", "patience", "grad_clip", "train_loss",
    ]
    return {k: mc.get(k) for k in keys if k in mc}


def assert_stdn_recipe(cfg_doc: Dict[str, Any], requested: Dict[str, Any], context: str) -> Dict[str, Any]:
    mc = cfg_doc["models"]["STDN"]
    exp = expected_recipe(requested)
    errors = []
    for key, wanted in exp.items():
        got = mc.get(key)
        if isinstance(wanted, float):
            if got is None or abs(float(got) - wanted) > max(1e-12, abs(wanted) * 1e-9):
                errors.append(f"{key}={got!r}, expected={wanted!r}")
        elif int(got) != wanted:
            errors.append(f"{key}={got!r}, expected={wanted!r}")
    fb = mc.get("fallback_batch_sizes", [])
    if list(fb or []) != []:
        errors.append(f"fallback_batch_sizes={fb!r}, expected=[] for exact diagnostic")
    if errors:
        raise RuntimeError(f"STDN_Special recipe mismatch at {context}: " + "; ".join(errors))
    return _recipe_snapshot(mc)


def install_yaml_recipe_override(requested: Dict[str, Any], report: List[dict], audit: Dict[str, Any]) -> None:
    """Patch only backbone.utils.load_yaml.

    scripts/run_experiment.py imports ``load_yaml`` from backbone.utils when its
    module is executed.  Installing this override first guarantees the local
    ``mc = dict(cfg['models']['STDN'])`` in backbone.runner sees the requested
    patience/max_epochs/batch/LR/WD values without changing shared files.
    """
    import backbone.utils as bu

    original = bu.load_yaml

    def patched_load_yaml(path):  # type: ignore[no-untyped-def]
        doc = original(path)
        if isinstance(doc, dict) and isinstance(doc.get("models"), dict) and "STDN" in doc["models"]:
            before = _recipe_snapshot(dict(doc["models"]["STDN"]))
            apply_stdn_recipe(doc, requested)
            after = assert_stdn_recipe(doc, requested, "patched load_yaml")
            audit["yaml_recipe_before"] = before
            audit["yaml_recipe_after"] = after
            print(
                "[STDN_SPECIAL_EFFECTIVE] recipe "
                f"batch_size={after['batch_size']} patience={after['patience']} "
                f"max_epochs={after['max_epochs']} lr={float(after['learning_rate']):.12g} "
                f"weight_decay={float(after['weight_decay']):.12g} fallback_batch_sizes=[]",
                flush=True,
            )
        return doc

    bu.load_yaml = patched_load_yaml  # type: ignore[assignment]
    report.append({
        "kind": "yaml_recipe_override",
        "status": "installed",
        "target": "cfg.models.STDN",
        "requested": expected_recipe(requested),
        "fallback_batch_sizes": [],
    })


def install_runner_guards(requested: Dict[str, Any], report: List[dict], audit: Dict[str, Any]) -> None:
    """Install read-only assertions around the host runner.

    These wrappers *verify* the config/optimizer; they do not force a different
    batch into unrelated functions and do not mutate vendor code.
    """
    import backbone.runner as br

    original_train_model = br.train_model
    original_make_optimizer = br.make_optimizer

    def guarded_train_model(model_name, cfg, *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(model_name) == "STDN":
            snap = assert_stdn_recipe(cfg, requested, "train_model entry")
            audit["train_model_recipe"] = snap
            print(
                "[STDN_SPECIAL_EFFECTIVE] train_model_recipe=" + json.dumps(snap, sort_keys=True),
                flush=True,
            )
        return original_train_model(model_name, cfg, *args, **kwargs)

    def guarded_make_optimizer(model, mc):  # type: ignore[no-untyped-def]
        # make_optimizer is reached after _train_once copied models.STDN into mc.
        exp = expected_recipe(requested)
        for key in ("learning_rate", "weight_decay"):
            got = float(mc.get(key, 0.0))
            wanted = float(exp[key])
            if abs(got - wanted) > max(1e-12, abs(wanted) * 1e-9):
                raise RuntimeError(f"Optimizer recipe mismatch: {key}={got}, expected={wanted}")
        opt = original_make_optimizer(model, mc)
        if not opt.param_groups:
            raise RuntimeError("Optimizer has no parameter groups")
        lr = float(opt.param_groups[0].get("lr", float("nan")))
        wd = float(opt.param_groups[0].get("weight_decay", 0.0))
        audit["optimizer"] = {"class": type(opt).__name__, "lr": lr, "weight_decay": wd}
        print(
            f"[STDN_SPECIAL_EFFECTIVE] optimizer={type(opt).__name__} lr={lr:.12g} weight_decay={wd:.12g}",
            flush=True,
        )
        if abs(lr - float(exp["learning_rate"])) > max(1e-12, abs(float(exp["learning_rate"])) * 1e-9):
            raise RuntimeError(f"Effective optimizer lr={lr}, expected={exp['learning_rate']}")
        if abs(wd - float(exp["weight_decay"])) > max(1e-12, abs(float(exp["weight_decay"])) * 1e-9):
            raise RuntimeError(f"Effective optimizer weight_decay={wd}, expected={exp['weight_decay']}")
        return opt

    guarded_train_model.__name__ = getattr(original_train_model, "__name__", "train_model")
    guarded_make_optimizer.__name__ = getattr(original_make_optimizer, "__name__", "make_optimizer")
    br.train_model = guarded_train_model  # type: ignore[assignment]
    br.make_optimizer = guarded_make_optimizer  # type: ignore[assignment]
    report.append({"kind": "runner_guard", "status": "installed", "targets": ["train_model", "make_optimizer"]})


def import_and_patch_backbone(requested: Dict[str, Any], report: List[dict], audit: Dict[str, Any]) -> List[Any]:
    """Import host modules once, then install the narrow v4 hooks."""
    import importlib

    modules = []
    for name in ("backbone.utils", "backbone.runner", "backbone.adapters", "backbone.data"):
        modules.append(importlib.import_module(name))
    install_yaml_recipe_override(requested, report, audit)
    install_runner_guards(requested, report, audit)
    return modules


def source_audit(root: Path, out_path: Path) -> Dict[str, Any]:
    needles = (
        "cfg['models']", "mc = dict", "batch_size", "patience", "max_epochs",
        "make_optimizer", "early_stop", "STDN", "load_yaml",
    )
    files = [
        root / "backbone" / "runner.py",
        root / "backbone" / "utils.py",
        root / "backbone" / "adapters.py",
        root / "scripts" / "run_experiment.py",
    ]
    result: Dict[str, Any] = {"files": {}}
    for path in files:
        if not path.exists():
            continue
        lines = path.read_text(errors="replace").splitlines()
        hits = []
        for i, line in enumerate(lines, 1):
            if any(n.lower() in line.lower() for n in needles):
                hits.append({"line": i, "text": line[:500]})
        result["files"][str(path)] = hits
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result
