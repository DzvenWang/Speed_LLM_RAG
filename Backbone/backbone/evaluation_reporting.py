from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence
import math
import json

import numpy as np
import pandas as pd


METRIC_SPECS = [
    ('MAE', 'MAE'),
    ('RMSE', 'RMSE'),
    ('MAPE', 'MAPE'),
    ('MAPE_count', 'MAPE_count'),
    ('MAPE_coverage', 'MAPE_coverage'),
    ('Masked_MAPE', 'directly_masked_MAPE'),
    ('H12_MAPE', 'H12_MAPE'),
    ('Masked_MAE', 'directly_masked_MAE'),
    ('H12_MAE', 'H12_MAE'),
]


def _float_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def severity_row(summary: Mapping, severity: int, seed: int) -> dict:
    row = {'severity': int(severity), 'seed': int(seed)}
    for label, key in METRIC_SPECS:
        row[label] = _float_or_nan(summary.get(key))
    return row


def scenario_rows(summary: Mapping, families: Sequence[str], severity: int, seed: int,
                  query_counts: Mapping[str, int] | None = None) -> list[dict]:
    rows: list[dict] = []
    query_counts = query_counts or {}
    for fam in families:
        row = {
            'severity': int(severity),
            'seed': int(seed),
            'scenario': str(fam),
            'queries': int(query_counts.get(str(fam), 0)),
        }
        for label, key in METRIC_SPECS:
            row[label] = _float_or_nan(summary.get(f'{key}__{fam}'))
        rows.append(row)
    return rows


def write_result_json(path: str | Path, *, severity: dict, scenarios: list[dict], metadata: Mapping | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {'severity': severity, 'scenarios': scenarios}
    if metadata:
        payload['metadata'] = dict(metadata)
    p.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding='utf-8')


def _fmt_value(value, decimals=4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return '-'
    if not math.isfinite(x):
        return '-'
    return f'{x:.{decimals}f}'


def _format_table(rows: Sequence[Mapping], columns: Sequence[tuple[str, str, str]]) -> list[str]:
    """Return an ASCII table.

    columns: (key, heading, alignment), where alignment is '<' or '>'.
    """
    if not rows:
        return ['(no rows)']
    rendered: list[list[str]] = []
    for row in rows:
        vals = []
        for key, _, _ in columns:
            value = row.get(key, '')
            if key == 'severity':
                vals.append(f's{int(value)}')
            elif key in {'seed', 'queries', 'MAPE_count'}:
                try:
                    vals.append(str(int(value)) if math.isfinite(float(value)) else '-')
                except (TypeError, ValueError, OverflowError):
                    vals.append('-')
            elif key in {x[0] for x in METRIC_SPECS}:
                vals.append(_fmt_value(value))
            else:
                vals.append(str(value))
        rendered.append(vals)
    widths = []
    for j, (_, heading, _) in enumerate(columns):
        widths.append(max(len(heading), *(len(r[j]) for r in rendered)))
    header = '  '.join(
        f'{heading:{align}{widths[j]}}' for j, (_, heading, align) in enumerate(columns)
    )
    sep = '  '.join('-' * w for w in widths)
    body = []
    for r in rendered:
        body.append('  '.join(f'{r[j]:{columns[j][2]}{widths[j]}}' for j in range(len(columns))))
    return [header, sep, *body]


def severity_table_lines(rows: Sequence[Mapping], include_seed: bool = False) -> list[str]:
    columns: list[tuple[str, str, str]] = [('severity', 'Severity', '<')]
    if include_seed:
        columns.append(('seed', 'Seed', '>'))
    columns += [
        ('MAE', 'MAE', '>'),
        ('RMSE', 'RMSE', '>'),
        ('MAPE', 'MAPE', '>'),
        ('MAPE_count', 'MAPE_count', '>'),
        ('MAPE_coverage', 'MAPE_coverage', '>'),
        ('Masked_MAPE', 'Masked_MAPE', '>'),
        ('H12_MAPE', 'H12_MAPE', '>'),
        ('Masked_MAE', 'Masked_MAE', '>'),
        ('H12_MAE', 'H12_MAE', '>'),
    ]
    return _format_table(rows, columns)


def scenario_table_lines(rows: Sequence[Mapping], include_seed: bool = False) -> list[str]:
    columns: list[tuple[str, str, str]] = [('severity', 'Severity', '<')]
    if include_seed:
        columns.append(('seed', 'Seed', '>'))
    columns += [
        ('scenario', 'Scenario', '<'),
        ('queries', 'Queries', '>'),
        ('MAE', 'MAE', '>'),
        ('RMSE', 'RMSE', '>'),
        ('MAPE', 'MAPE', '>'),
        ('MAPE_count', 'MAPE_count', '>'),
        ('MAPE_coverage', 'MAPE_coverage', '>'),
        ('Masked_MAPE', 'Masked_MAPE', '>'),
        ('H12_MAPE', 'H12_MAPE', '>'),
        ('Masked_MAE', 'Masked_MAE', '>'),
        ('H12_MAE', 'H12_MAE', '>'),
    ]
    return _format_table(rows, columns)


def aggregate_mean(rows: Sequence[Mapping], group_keys: Sequence[str]) -> list[dict]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    metrics = [x[0] for x in METRIC_SPECS]
    aggs: list[dict] = []
    for keys, group in df.groupby(list(group_keys), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {k: v for k, v in zip(group_keys, keys)}
        rec['n_seeds'] = int(group['seed'].nunique()) if 'seed' in group else 1
        if 'queries' in group:
            rec['queries'] = int(round(pd.to_numeric(group['queries'], errors='coerce').mean()))
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors='coerce')
            rec[metric] = float(vals.mean())
            rec[f'{metric}_std'] = float(vals.std(ddof=0))
        aggs.append(rec)
    return aggs
