from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import hashlib
import json

import numpy as np
import pandas as pd

from .constants import DATA_CONTRACT_REVISION, TIME_FEATURE_REVISION


@dataclass(frozen=True)
class ParsedTimeFeatures:
    tod: np.ndarray
    dow: np.ndarray
    source: str
    column: Optional[str]
    parser: str
    freq_minutes: int
    steps_per_day: int
    diagnostics: Dict[str, Any]


def _column_name_score(name: str) -> int:
    low = str(name).strip().lower()
    exact = {
        "timestamp", "datetime", "date_time", "time", "date",
        "datetime_utc", "timestamp_utc",
    }
    if low in exact:
        return 100
    score = 0
    if "timestamp" in low:
        score += 80
    if "datetime" in low:
        score += 70
    if "date" in low:
        score += 40
    if "time" in low:
        score += 35
    if low in {"index", "idx", "id", "row", "t"}:
        score -= 100
    return score


def _numeric_epoch_parse(series: pd.Series) -> Tuple[Optional[pd.Series], Optional[str]]:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().mean() < 0.95:
        return None, None
    finite = vals[np.isfinite(vals)]
    if len(finite) < 2:
        return None, None
    med = float(np.median(np.abs(finite.to_numpy(dtype=np.float64))))
    # An ordinary row index must never be interpreted as nanoseconds after Unix epoch.
    if med < 1e8:
        return None, None
    if med >= 1e17:
        unit = "ns"
    elif med >= 1e14:
        unit = "us"
    elif med >= 1e11:
        unit = "ms"
    else:
        unit = "s"
    parsed = pd.to_datetime(vals, errors="coerce", unit=unit, utc=True)
    if parsed.notna().mean() < 0.95:
        return None, None
    return pd.Series(parsed).dt.tz_convert(None), unit


def _validate_datetime_series(parsed: pd.Series, T: int, freq_minutes: int) -> Tuple[bool, Dict[str, Any]]:
    p = parsed.iloc[:T]
    diag: Dict[str, Any] = {
        "length": int(len(p)),
        "valid_fraction": float(p.notna().mean()) if len(p) else 0.0,
    }
    if len(p) < T or p.notna().mean() < 0.95:
        return False, diag

    q = p.dropna().reset_index(drop=True)
    if len(q) < 2:
        return False, diag
    ns = q.astype("int64").to_numpy()
    delta_min = np.diff(ns) / 60_000_000_000.0
    positive = delta_min[delta_min > 0]
    median_delta = float(np.median(positive)) if len(positive) else None
    diag["median_positive_delta_minutes"] = median_delta
    diag["monotonic_non_decreasing"] = bool(np.all(np.diff(ns) >= 0))
    diag["unique_timestamps"] = int(q.nunique())

    tod = ((q.dt.hour * 60 + q.dt.minute) // int(freq_minutes)).to_numpy(np.int16)
    dow = q.dt.dayofweek.to_numpy(np.int8)
    steps_per_day = int(round(24 * 60 / freq_minutes))
    diag["tod_unique"] = int(np.unique(tod).size)
    diag["dow_unique"] = int(np.unique(dow).size)

    cadence_ok = (
        median_delta is not None
        and abs(median_delta - float(freq_minutes)) <= max(1.0, 0.25 * freq_minutes)
    )
    diversity_ok = diag["tod_unique"] >= min(24, steps_per_day)
    if T >= 2 * steps_per_day:
        diversity_ok = diversity_ok and diag["dow_unique"] >= 2
    ok = bool(diag["monotonic_non_decreasing"] and cadence_ok and diversity_ok)
    diag["cadence_ok"] = bool(cadence_ok)
    diag["diversity_ok"] = bool(diversity_ok)
    return ok, diag


def resolve_time_features(
    bench,
    *,
    allow_index_fallback: bool = False,
    index_origin_dow: int = 0,
) -> ParsedTimeFeatures:
    """Resolve TOD/DOW while refusing unsafe numeric-index datetime parsing.

    A real timestamp column is preferred. If none is safe and index fallback is
    explicitly enabled, TOD/DOW are derived from the known fixed sampling grid.
    """
    T = int(bench.traffic.shape[0])
    freq = int(bench.cfg["forecast"]["freq_minutes"])
    if freq <= 0 or (24 * 60) % freq != 0:
        raise RuntimeError(f"Unsupported forecast frequency: {freq} minutes")
    steps_per_day = int((24 * 60) // freq)
    path = Path(bench.sg_dir) / "core" / "time_index.csv"
    attempts = []

    if path.exists():
        df = pd.read_csv(path)
        ranked = sorted(list(df.columns), key=lambda c: _column_name_score(str(c)), reverse=True)
        for col in ranked:
            s = df[col]
            score = _column_name_score(str(col))
            parsed = None
            parser = None

            if pd.api.types.is_numeric_dtype(s):
                if score > 0:
                    parsed, unit = _numeric_epoch_parse(s)
                    if parsed is not None:
                        parser = f"numeric_epoch_{unit}"
            else:
                p = pd.to_datetime(s, errors="coerce", utc=True)
                if p.notna().mean() >= 0.95:
                    parsed = pd.Series(p).dt.tz_convert(None)
                    parser = "datetime_string"

            if parsed is None:
                attempts.append({
                    "column": str(col), "score": int(score), "accepted": False,
                    "reason": "not a safe datetime candidate",
                })
                continue

            ok, diag = _validate_datetime_series(parsed, T, freq)
            attempts.append({
                "column": str(col), "score": int(score), "parser": parser,
                "accepted": bool(ok), **diag,
            })
            if not ok:
                continue

            p = parsed.iloc[:T]
            tod = ((p.dt.hour * 60 + p.dt.minute) // freq).to_numpy(np.int16)
            dow = p.dt.dayofweek.to_numpy(np.int8)
            result = ParsedTimeFeatures(
                tod=tod,
                dow=dow,
                source="core/time_index.csv",
                column=str(col),
                parser=str(parser),
                freq_minutes=freq,
                steps_per_day=steps_per_day,
                diagnostics={"attempts": attempts},
            )
            assert_resolved_time_features(result)
            return result

    if not allow_index_fallback:
        raise RuntimeError(
            "No safe datetime column was found in core/time_index.csv and index fallback is disabled. "
            "Enable time_features.allow_index_modulo_fallback only when the dataset is known to be a "
            "continuous fixed-frequency series. Attempts=" + json.dumps(attempts[:20], default=str)
        )

    origin = int(index_origin_dow)
    if not 0 <= origin <= 6:
        raise ValueError(f"index_origin_dow must be in [0,6], got {origin}")
    idx = np.arange(T, dtype=np.int64)
    result = ParsedTimeFeatures(
        tod=(idx % steps_per_day).astype(np.int16),
        dow=((origin + idx // steps_per_day) % 7).astype(np.int8),
        source="index_modulo_fallback",
        column=None,
        parser="index_modulo",
        freq_minutes=freq,
        steps_per_day=steps_per_day,
        diagnostics={
            "attempts": attempts,
            "index_origin_dow": origin,
            "warning": "DOW labels use a synthetic day-0 alignment; the weekly cycle remains internally consistent.",
        },
    )
    assert_resolved_time_features(result)
    return result


def _array_sha256(a: np.ndarray) -> str:
    x = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(x.dtype).encode("utf-8"))
    h.update(str(tuple(x.shape)).encode("utf-8"))
    h.update(x.tobytes(order="C"))
    return h.hexdigest()


def time_feature_fingerprint(result: ParsedTimeFeatures) -> str:
    h = hashlib.sha256()
    for value in (
        TIME_FEATURE_REVISION,
        result.source,
        result.column or "",
        result.parser,
        str(result.freq_minutes),
        str(result.steps_per_day),
        _array_sha256(result.tod),
        _array_sha256(result.dow),
    ):
        h.update(value.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def audit_resolved_time_features(result: ParsedTimeFeatures) -> Dict[str, Any]:
    tod = np.asarray(result.tod, dtype=np.int64)
    dow = np.asarray(result.dow, dtype=np.int64)
    errors = []
    if tod.ndim != 1 or dow.ndim != 1 or len(tod) != len(dow):
        errors.append(f"Expected aligned 1-D TOD/DOW arrays, got tod={tod.shape}, dow={dow.shape}")
    if tod.size:
        if int(tod.min()) < 0 or int(tod.max()) >= result.steps_per_day:
            errors.append(f"TOD outside [0,{result.steps_per_day - 1}]")
        if int(dow.min()) < 0 or int(dow.max()) > 6:
            errors.append("DOW outside [0,6]")
        if np.unique(tod).size < min(24, result.steps_per_day):
            errors.append(f"Only {np.unique(tod).size} unique TOD values")
        if len(tod) >= 2 * result.steps_per_day and np.unique(dow).size < 2:
            errors.append(f"Only {np.unique(dow).size} unique DOW values")
        if len(tod) >= 2:
            tod_ok = tod[1:] == ((tod[:-1] + 1) % result.steps_per_day)
            expected_dow = (dow[:-1] + (tod[:-1] == result.steps_per_day - 1).astype(np.int64)) % 7
            dow_ok = dow[1:] == expected_dow
            tod_consistency = float(tod_ok.mean())
            dow_consistency = float(dow_ok.mean())
            if tod_consistency < 0.999999:
                errors.append(f"Full-series TOD step consistency={tod_consistency:.6f}")
            if dow_consistency < 0.999999:
                errors.append(f"Full-series DOW rollover consistency={dow_consistency:.6f}")
        else:
            tod_consistency = None
            dow_consistency = None
    else:
        errors.append("Empty time-feature arrays")
        tod_consistency = None
        dow_consistency = None

    return {
        "data_contract_revision": DATA_CONTRACT_REVISION,
        "time_feature_revision": TIME_FEATURE_REVISION,
        "source": result.source,
        "column": result.column,
        "parser": result.parser,
        "freq_minutes": int(result.freq_minutes),
        "steps_per_day": int(result.steps_per_day),
        "length": int(len(tod)),
        "tod_min": int(tod.min()) if tod.size else None,
        "tod_max": int(tod.max()) if tod.size else None,
        "tod_unique": int(np.unique(tod).size),
        "dow_min": int(dow.min()) if dow.size else None,
        "dow_max": int(dow.max()) if dow.size else None,
        "dow_unique": int(np.unique(dow).size),
        "tod_step_consistency": tod_consistency,
        "dow_rollover_consistency": dow_consistency,
        "fingerprint": time_feature_fingerprint(result),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "diagnostics": result.diagnostics,
    }


def assert_resolved_time_features(result: ParsedTimeFeatures) -> Dict[str, Any]:
    report = audit_resolved_time_features(result)
    if report["status"] != "PASS":
        raise RuntimeError("Invalid resolved time features: " + json.dumps(report, default=str))
    return report


def summarize_array(path: Path) -> Dict[str, Any]:
    a = np.load(path, mmap_mode="r", allow_pickle=False)
    u = np.unique(np.asarray(a))
    return {
        "path": str(path),
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "min": int(u.min()) if u.size else None,
        "max": int(u.max()) if u.size else None,
        "n_unique": int(u.size),
        "unique_head": [int(x) for x in u[:32]],
    }


def _sequence_checks(tod: np.ndarray, dow: np.ndarray, steps_per_day: int) -> Dict[str, Any]:
    if tod.size == 0:
        return {"checked_rows": 0, "tod_step_consistency": None, "dow_rollover_consistency": None}
    rows = np.linspace(0, tod.shape[0] - 1, min(512, tod.shape[0]), dtype=np.int64)
    t = np.asarray(tod[rows], dtype=np.int64)
    d = np.asarray(dow[rows], dtype=np.int64)
    if t.shape[1] < 2:
        return {"checked_rows": int(len(rows)), "tod_step_consistency": None, "dow_rollover_consistency": None}
    expected_t = (t[:, :-1] + 1) % steps_per_day
    expected_d = (d[:, :-1] + (t[:, :-1] == steps_per_day - 1).astype(np.int64)) % 7
    return {
        "checked_rows": int(len(rows)),
        "tod_step_consistency": float((expected_t == t[:, 1:]).mean()),
        "dow_rollover_consistency": float((expected_d == d[:, 1:]).mean()),
    }


def audit_cache_time_features(
    cache: Path,
    *,
    steps_per_day: int = 288,
    expected_splits: Optional[set[str]] = None,
) -> Dict[str, Any]:
    cache = Path(cache)
    report: Dict[str, Any] = {
        "data_contract_revision": DATA_CONTRACT_REVISION,
        "time_feature_revision": TIME_FEATURE_REVISION,
        "cache": str(cache),
        "splits": {},
        "errors": [],
        "warnings": [],
    }
    split_names = sorted({p.name.split("_tod_hist.npy")[0] for p in cache.glob("*_tod_hist.npy")})
    if not split_names:
        report["errors"].append("No *_tod_hist.npy files found")
    if expected_splits is not None and set(split_names) != set(expected_splits):
        report["errors"].append(
            f"Time-feature split set mismatch: expected={sorted(expected_splits)} got={split_names}"
        )

    for split in split_names:
        paths = {
            "tod_hist": cache / f"{split}_tod_hist.npy",
            "dow_hist": cache / f"{split}_dow_hist.npy",
            "tod_future": cache / f"{split}_tod_future.npy",
            "dow_future": cache / f"{split}_dow_future.npy",
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            report["errors"].append(f"{split}: missing {missing}")
            continue
        info = {k: summarize_array(p) for k, p in paths.items()}
        th = np.load(paths["tod_hist"], mmap_mode="r")
        dh = np.load(paths["dow_hist"], mmap_mode="r")
        tf = np.load(paths["tod_future"], mmap_mode="r")
        df = np.load(paths["dow_future"], mmap_mode="r")
        if not (th.shape[0] == dh.shape[0] == tf.shape[0] == df.shape[0]):
            report["errors"].append(f"{split}: time-feature row counts differ")
            continue
        joined_t = np.concatenate([np.asarray(th), np.asarray(tf)], axis=1)
        joined_d = np.concatenate([np.asarray(dh), np.asarray(df)], axis=1)
        seq = _sequence_checks(joined_t, joined_d, steps_per_day)
        split_errors = []

        for key in ("tod_hist", "tod_future"):
            s = info[key]
            if s["min"] is None or s["min"] < 0 or s["max"] >= steps_per_day:
                split_errors.append(f"{key} outside [0,{steps_per_day - 1}]")
            if s["n_unique"] < min(24, steps_per_day):
                split_errors.append(f"{key} has only {s['n_unique']} unique TOD values")
        for key in ("dow_hist", "dow_future"):
            s = info[key]
            if s["min"] is None or s["min"] < 0 or s["max"] > 6:
                split_errors.append(f"{key} outside [0,6]")
            if s["n_unique"] < 2:
                split_errors.append(f"{key} has only {s['n_unique']} unique DOW values")
        if seq.get("tod_step_consistency") is not None and float(seq["tod_step_consistency"]) < 0.999:
            split_errors.append(f"TOD within-window step consistency={seq['tod_step_consistency']:.6f}")
        if seq.get("dow_rollover_consistency") is not None and float(seq["dow_rollover_consistency"]) < 0.999:
            split_errors.append(f"DOW rollover consistency={seq['dow_rollover_consistency']:.6f}")

        signature = (
            info["tod_hist"]["n_unique"] == 1 and info["tod_hist"]["min"] == 0
            and info["tod_future"]["n_unique"] == 1 and info["tod_future"]["min"] == 0
            and info["dow_hist"]["n_unique"] == 1 and info["dow_hist"]["min"] == 3
            and info["dow_future"]["n_unique"] == 1 and info["dow_future"]["min"] == 3
        )
        if signature:
            split_errors.append(
                "Detected TOD=0/DOW=3 constant signature, consistent with parsing an ordinary numeric index "
                "as nanoseconds after Unix epoch (1970-01-01 was Thursday)."
            )
        info["sequence_checks"] = seq
        info["errors"] = split_errors
        report["splits"][split] = info
        report["errors"].extend([f"{split}: {x}" for x in split_errors])

    report["status"] = "PASS" if not report["errors"] else "FAIL"
    return report
