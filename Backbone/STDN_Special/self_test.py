#!/usr/bin/env python3
from __future__ import annotations
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from runtime_overrides import apply_stdn_recipe, assert_stdn_recipe
from time_features import resolve_time_features_safe


def test_recipe():
    req = {"batch_size": 64, "learning_rate": 2e-4, "weight_decay": 1e-4, "patience": 30, "epochs": 200}
    doc = {"models": {"STDN": {"batch_size": 64, "fallback_batch_sizes": [32, 16], "learning_rate": 1e-3, "weight_decay": 0.0, "patience": 10, "max_epochs": 200}}}
    apply_stdn_recipe(doc, req)
    snap = assert_stdn_recipe(doc, req, "self-test")
    assert snap["patience"] == 30 and snap["fallback_batch_sizes"] == []


def test_datetime_parser_rejects_numeric_index():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "core").mkdir()
        T = 7 * 288
        ts = pd.date_range("2026-01-05 00:00:00", periods=T, freq="5min")
        # Numeric index intentionally comes first: this is the v3/cache bug reproducer.
        pd.DataFrame({"index": np.arange(T), "timestamp": ts.astype(str)}).to_csv(root / "core" / "time_index.csv", index=False)
        bench = SimpleNamespace(
            traffic=np.zeros((T, 1), dtype=np.float32),
            cfg={"forecast": {"freq_minutes": 5}},
            sg_dir=root,
        )
        r = resolve_time_features_safe(bench)
        assert r.column == "timestamp", r
        assert np.unique(r.tod).size == 288
        assert np.unique(r.dow).size == 7
        assert not (np.all(r.tod == 0) and np.all(r.dow == 3))


def main():
    test_recipe()
    test_datetime_parser_rejects_numeric_index()
    print("STDN_Special v4.1 SELF TEST PASS")


if __name__ == "__main__":
    main()
