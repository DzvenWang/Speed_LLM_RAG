from __future__ import annotations

import hashlib
from collections import deque

import pandas as pd


class _Dinic:
    def __init__(self, n: int):
        self.g = [[] for _ in range(int(n))]

    def add_edge(self, u: int, v: int, cap: int):
        fwd = [int(v), len(self.g[v]), int(cap), int(cap)]
        rev = [int(u), len(self.g[u]), 0, 0]
        self.g[u].append(fwd)
        self.g[v].append(rev)
        return int(u), len(self.g[u]) - 1

    def max_flow(self, s: int, t: int) -> int:
        flow = 0
        n = len(self.g)
        INF = 10**18
        while True:
            level = [-1] * n
            level[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for e in self.g[u]:
                    if e[2] > 0 and level[e[0]] < 0:
                        level[e[0]] = level[u] + 1
                        q.append(e[0])
            if level[t] < 0:
                break
            it = [0] * n

            def dfs(u: int, pushed: int) -> int:
                if u == t:
                    return pushed
                while it[u] < len(self.g[u]):
                    ei = it[u]
                    e = self.g[u][ei]
                    if e[2] > 0 and level[e[0]] == level[u] + 1:
                        take = dfs(e[0], min(pushed, e[2]))
                        if take:
                            e[2] -= take
                            self.g[e[0]][e[1]][2] += take
                            return take
                    it[u] += 1
                return 0

            while True:
                pushed = dfs(s, INF)
                if not pushed:
                    break
                flow += pushed
        return int(flow)


def quotas_for_total(families: list[str], total: int) -> dict[str, int]:
    base = int(total) // len(families)
    rem = int(total) - base * len(families)
    return {f: base + int(i < rem) for i, f in enumerate(families)}


def balanced_unique_group_pick(df: pd.DataFrame, families: list[str], total: int, seed: int) -> pd.DataFrame:
    fams = list(families)
    if not fams:
        raise ValueError('At least one sensing family is required.')
    d = df[df.family.astype(str).isin(fams)].copy()
    if len(d) == 0:
        raise ValueError('No eligible views for shared protocol sampling.')
    d['view_group_id'] = d.view_group_id.astype(str)
    d['family'] = d.family.astype(str)

    gmeta = (
        d.groupby('view_group_id', sort=False)
        .agg(
            window_start=('window_start', 'min'),
            available_families=('family', lambda x: tuple(sorted(set(x), key=fams.index))),
        )
        .reset_index()
    )
    req = int(total) if int(total) > 0 else int(len(gmeta))
    if len(gmeta) < req:
        raise ValueError(f'Need {req} unique groups, have {len(gmeta)}.')

    quotas = quotas_for_total(fams, req)
    available = {
        f: int(gmeta.available_families.map(lambda a, ff=f: ff in a).sum())
        for f in fams
    }
    bad = {f: (quotas[f], available[f]) for f in fams if available[f] < quotas[f]}
    if bad:
        raise ValueError(f'Per-family capacity is insufficient: {bad}')

    pattern_groups: dict[tuple[str, ...], list[tuple[str, int]]] = {}
    for r in gmeta.itertuples(index=False):
        pattern_groups.setdefault(tuple(r.available_families), []).append(
            (str(r.view_group_id), int(r.window_start))
        )
    patterns = sorted(pattern_groups, key=lambda p: (len(p), p))

    F, P = len(fams), len(patterns)
    source, fam_offset, pat_offset = 0, 1, 1 + F
    sink = pat_offset + P
    mf = _Dinic(sink + 1)
    for fi, f in enumerate(fams):
        mf.add_edge(source, fam_offset + fi, quotas[f])
    refs = {}
    for fi, f in enumerate(fams):
        for pi, pat in enumerate(patterns):
            if f in pat:
                refs[(fi, pi)] = mf.add_edge(fam_offset + fi, pat_offset + pi, len(pattern_groups[pat]))
    for pi, pat in enumerate(patterns):
        mf.add_edge(pat_offset + pi, sink, len(pattern_groups[pat]))

    got = mf.max_flow(source, sink)
    if got != req:
        capacities = {str(p): len(pattern_groups[p]) for p in patterns}
        raise ValueError(
            'No exact balanced unique-group assignment: '
            f'need={req}, got={got}, quotas={quotas}, family_available={available}, patterns={capacities}'
        )

    alloc = {pat: {f: 0 for f in fams} for pat in patterns}
    for (fi, pi), (u, ei) in refs.items():
        e = mf.g[u][ei]
        alloc[patterns[pi]][fams[fi]] = int(e[3] - e[2])

    selected: list[tuple[str, str]] = []
    for pat in patterns:
        groups = sorted(
            pattern_groups[pat],
            key=lambda x: hashlib.sha256(f'{seed}:{x[0]}'.encode('utf-8')).hexdigest(),
        )
        need = sum(alloc[pat].values())
        groups = groups[:need]
        cursor = 0
        for f in fams:
            n = alloc[pat][f]
            for gid, _ in groups[cursor:cursor + n]:
                selected.append((gid, f))
            cursor += n
        if cursor != len(groups):
            raise AssertionError('Internal protocol sampler allocation mismatch.')

    rows = []
    for gid, fam in selected:
        cand = d[(d.view_group_id == gid) & (d.family == fam)].sort_values(['window_start', 'view_id'])
        if len(cand) == 0:
            raise AssertionError((gid, fam))
        rows.append(cand.iloc[0])
    out = pd.DataFrame(rows).reset_index(drop=True)
    if len(out) != req:
        raise AssertionError(f'Expected {req} rows, got {len(out)}.')
    if out.view_group_id.astype(str).duplicated().any():
        raise AssertionError('Duplicate view_group_id in protocol selection.')
    counts = out.family.astype(str).value_counts().to_dict()
    if any(int(counts.get(f, 0)) != int(quotas[f]) for f in fams):
        raise AssertionError(f'Family quota mismatch: expected={quotas}, actual={counts}')
    return out.sort_values(['window_start', 'family', 'view_id']).reset_index(drop=True)
