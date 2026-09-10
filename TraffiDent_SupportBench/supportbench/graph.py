from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import canonicalize_nodes, ensure_dir, log, stable_seed, write_json


def undirected_neighbors(adj: np.ndarray, node: int) -> np.ndarray:
    a = np.asarray(adj)
    mask = (np.abs(a[node]) > 0) | (np.abs(a[:, node]) > 0)
    mask[node] = False
    return np.flatnonzero(mask).astype(np.int64)


def direction_progress(nodes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    nodes = canonicalize_nodes(nodes)
    n = len(nodes)
    progress = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for _, row in nodes.iterrows():
        i = int(row['local_idx'])
        d = str(row.get('direction', '')).strip().upper().replace('BOUND', '').replace('B', '')[:1]
        lat = float(row.get('lat', np.nan))
        lng = float(row.get('lng', np.nan))
        pm = float(row.get('postmile', np.nan))
        if d in {'N', 'S'} and np.isfinite(lat):
            progress[i] = lat if d == 'N' else -lat; valid[i] = True
        elif d in {'E', 'W'} and np.isfinite(lng):
            progress[i] = lng if d == 'E' else -lng; valid[i] = True
        elif d in {'N', 'E', 'S', 'W'} and np.isfinite(pm):
            progress[i] = (1.0 if d in {'N','E'} else -1.0) * pm; valid[i] = True
    return progress, valid


def build_directed_neighbors(nodes: pd.DataFrame, adj: np.ndarray, distance: np.ndarray | None, topk: int = 4) -> dict[str, np.ndarray]:
    nodes = canonicalize_nodes(nodes)
    n = len(nodes)
    progress, pvalid = direction_progress(nodes)
    up_ids = np.full((n, topk), -1, dtype=np.int64)
    down_ids = np.full((n, topk), -1, dtype=np.int64)
    up_w = np.zeros((n, topk), dtype=np.float32)
    down_w = np.zeros((n, topk), dtype=np.float32)
    for i in range(n):
        up, down = [], []
        for j in undirected_neighbors(adj, i):
            j = int(j)
            if str(nodes.loc[i, 'road_direction']) != str(nodes.loc[j, 'road_direction']):
                continue
            if not (pvalid[i] and pvalid[j]) or progress[i] == progress[j]:
                continue
            if distance is not None and distance.shape == (n,n) and np.isfinite(distance[i,j]) and distance[i,j] > 0:
                affinity = 1.0 / float(distance[i,j])
            else:
                affinity = max(float(abs(adj[i,j])) if np.isfinite(adj[i,j]) else 0.0, 1.0)
            (up if progress[j] < progress[i] else down).append((j, affinity))
        up.sort(key=lambda x: -x[1]); down.sort(key=lambda x: -x[1])
        for arr_id, arr_w, vals in [(up_ids, up_w, up[:topk]), (down_ids, down_w, down[:topk])]:
            if vals:
                raw = np.asarray([max(v[1], 1e-8) for v in vals], dtype=np.float64); raw /= raw.sum()
                arr_id[i, :len(vals)] = [v[0] for v in vals]
                arr_w[i, :len(vals)] = raw.astype(np.float32)
    return {'upstream_ids': up_ids, 'upstream_weights': up_w, 'downstream_ids': down_ids, 'downstream_weights': down_w}


def _component_nodes(adj: np.ndarray, start: int, allowed: set[int] | None = None) -> list[int]:
    start = int(start)
    if allowed is not None and start not in allowed:
        return []
    q = deque([start]); seen = {start}; out = []
    while q:
        u = q.popleft(); out.append(u)
        for v in undirected_neighbors(adj, u):
            v = int(v)
            if allowed is not None and v not in allowed:
                continue
            if v not in seen:
                seen.add(v); q.append(v)
    return out


def connected_components(adj: np.ndarray, nodes: Iterable[int] | None = None) -> list[np.ndarray]:
    n = int(adj.shape[0])
    allowed = set(range(n)) if nodes is None else set(int(x) for x in nodes)
    remaining = set(allowed); comps: list[np.ndarray] = []
    while remaining:
        s = next(iter(remaining))
        comp = _component_nodes(adj, s, allowed)
        arr = np.asarray(comp, dtype=np.int64)
        comps.append(arr)
        remaining.difference_update(comp)
    comps.sort(key=lambda x: (-len(x), int(x.min()) if len(x) else 0))
    return comps


def _geo_distance(nodes: pd.DataFrame, center: int, candidates: np.ndarray) -> np.ndarray:
    nodes = canonicalize_nodes(nodes)
    center = int(center); candidates = np.asarray(candidates, dtype=np.int64)
    lat0 = float(nodes.loc[center, 'lat']); lon0 = float(nodes.loc[center, 'lng'])
    lat = pd.to_numeric(nodes.loc[candidates, 'lat'], errors='coerce').to_numpy(float)
    lon = pd.to_numeric(nodes.loc[candidates, 'lng'], errors='coerce').to_numpy(float)
    if np.isfinite(lat0) and np.isfinite(lon0) and np.isfinite(lat).any() and np.isfinite(lon).any():
        # Equirectangular approximation is enough for deterministic spatial ordering.
        dy = (lat - lat0) * 111.0
        dx = (lon - lon0) * 111.0 * np.cos(np.deg2rad(lat0))
        d = np.sqrt(dx*dx + dy*dy)
        bad = ~np.isfinite(d)
        if np.any(bad): d[bad] = np.nanmax(d[~bad]) + 1.0 if np.any(~bad) else np.arange(len(d))[bad]
        return d
    pm0 = float(nodes.loc[center, 'postmile'])
    pm = pd.to_numeric(nodes.loc[candidates, 'postmile'], errors='coerce').to_numpy(float)
    if np.isfinite(pm0) and np.isfinite(pm).any():
        d = np.abs(pm - pm0)
        bad = ~np.isfinite(d)
        if np.any(bad): d[bad] = np.nanmax(d[~bad]) + 1.0 if np.any(~bad) else np.arange(len(d))[bad]
        return d
    return np.abs(candidates.astype(float) - float(center))


def compact_region(adj: np.ndarray, nodes: pd.DataFrame, center: int, target_size: int) -> tuple[np.ndarray, dict]:
    """Return an exact-size spatially compact region.

    The first connected component is expanded by BFS from `center`. If the graph
    is disconnected and the component is too small, the next geographically
    nearest component is added and expanded by BFS. This avoids the old arbitrary
    node-index fallback while keeping exact missing severity.
    """
    n = int(adj.shape[0]); target_size = max(1, min(int(target_size), n)); center = int(center)
    selected: list[int] = []; selected_set: set[int] = set(); component_sizes: list[int] = []
    current_seed = center
    while len(selected) < target_size:
        comp = _component_nodes(adj, current_seed)
        comp = [x for x in comp if x not in selected_set]
        if not comp:
            remaining = np.asarray([i for i in range(n) if i not in selected_set], dtype=np.int64)
            if not len(remaining): break
            current_seed = int(remaining[0]); continue
        need = target_size - len(selected)
        take = comp[:need]
        selected.extend(take); selected_set.update(take); component_sizes.append(len(take))
        if len(selected) >= target_size: break
        remaining = np.asarray([i for i in range(n) if i not in selected_set], dtype=np.int64)
        if not len(remaining): break
        # Prefer a new component close to the original centre.
        d = _geo_distance(nodes, center, remaining)
        current_seed = int(remaining[int(np.argmin(d))])
    arr = np.asarray(selected[:target_size], dtype=np.int64)
    # Compute connectivity on the actual selected induced graph.
    comps = connected_components(adj, arr)
    sizes = [len(c) for c in comps]
    largest = max(sizes) if sizes else 0
    meta = {
        'center': center,
        'target_size': int(target_size),
        'actual_size': int(len(arr)),
        'component_count': int(len(comps)),
        'largest_component_fraction': float(largest / max(len(arr), 1)),
    }
    return np.sort(arr), meta


def bfs_region(adj: np.ndarray, center: int, target_size: int) -> np.ndarray:
    """Compatibility helper: connected BFS only, no arbitrary disconnected fill."""
    n = adj.shape[0]
    target_size = max(1, min(int(target_size), n))
    seen = {int(center)}; q = deque([int(center)]); order = [int(center)]
    while q and len(order) < target_size:
        u = q.popleft()
        for v in undirected_neighbors(adj, u):
            v = int(v)
            if v not in seen:
                seen.add(v); q.append(v); order.append(v)
                if len(order) >= target_size: break
    return np.asarray(order[:target_size], dtype=np.int64)


def local_patch(adj: np.ndarray, target: int, hops: int = 1, max_nodes: int = 16) -> np.ndarray:
    seen = {int(target)}; frontier = {int(target)}
    for _ in range(int(hops)):
        nxt = set()
        for u in frontier:
            nxt.update(int(v) for v in undirected_neighbors(adj, u))
        nxt -= seen; seen |= nxt; frontier = nxt
    nodes = [int(target)] + sorted(x for x in seen if x != int(target))
    return np.asarray(nodes[:max_nodes], dtype=np.int64)


def _direction_groups(nodes: pd.DataFrame) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    c = canonicalize_nodes(nodes)
    for name, g in c.groupby('road_direction'):
        idx = np.sort(g['local_idx'].to_numpy(np.int64))
        # Directional outage requires an interpretable direction label.
        if len(idx) >= 2 and 'unknown' not in str(name).lower():
            groups[str(name)] = idx.tolist()
    return groups


def _directional_layout(groups: dict[str, list[int]], n_nodes: int, target_size: int, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    target_size = max(1, min(int(target_size), int(n_nodes)))
    eligible = sorted(set(int(x) for vals in groups.values() for x in vals))
    if len(eligible) < target_size:
        raise ValueError(f'Directional groups cover only {len(eligible)}/{n_nodes} nodes but target_size={target_size}')
    names = list(groups)
    order = rng.permutation(len(names))
    chosen: list[int] = []; used_groups: list[str] = []
    for oi in order:
        name = names[int(oi)]; vals = np.asarray(groups[name], dtype=np.int64)
        remaining = target_size - len(set(chosen))
        if remaining <= 0: break
        vals = np.asarray([x for x in vals if x not in set(chosen)], dtype=np.int64)
        if not len(vals): continue
        used_groups.append(name)
        if len(vals) <= remaining:
            chosen.extend(vals.tolist())
        else:
            chosen.extend(np.sort(rng.choice(vals, size=remaining, replace=False)).tolist())
    ids = np.asarray(sorted(set(chosen)), dtype=np.int64)
    if len(ids) != target_size:
        raise RuntimeError(f'Failed to construct exact directional layout: {len(ids)} != {target_size}')
    return ids, used_groups


def _unique_layouts(
    generator,
    count: int,
    prefix: str,
    seed_base: int,
    forbidden: set[tuple[int, ...]] | None = None,
    max_attempts: int = 20000,
    *,
    min_count: int | None = None,
    allow_partial: bool = False,
    stale_patience: int = 2500,
):
    """Generate a deterministic catalogue of unique node layouts.

    Large regional outages can have fewer distinct compact layouts than a
    nominal catalogue request, because many centres collapse to the same
    exact node set at high severity.  In that case the *number of catalogue
    entries* is not a scientific requirement; matched severity, compactness
    and ID/OOD disjointness are.

    When ``allow_partial`` is enabled, generation stops after a long stale
    period and accepts the feasible catalogue if at least ``min_count``
    layouts were found.  Falling below the configured minimum still fails
    loudly.
    """
    requested = int(count)
    minimum = requested if min_count is None else max(1, min(int(min_count), requested))
    forbidden = set() if forbidden is None else set(forbidden)
    out = []
    seen = set(forbidden)
    attempts = 0
    stale = 0
    while len(out) < requested and attempts < int(max_attempts):
        rng = np.random.default_rng(stable_seed(prefix, attempts, base=seed_base))
        nodes, meta = generator(rng)
        key = tuple(int(x) for x in np.asarray(nodes, dtype=np.int64))
        attempts += 1
        if key in seen:
            stale += 1
        else:
            seen.add(key)
            out.append((nodes, meta))
            stale = 0
        if allow_partial and len(out) >= minimum and stale >= int(stale_patience):
            break

    if len(out) < minimum:
        raise RuntimeError(
            f'Could create only {len(out)}/{requested} unique layouts for {prefix}; '
            f'minimum required={minimum}. The requested sensing catalogue is '
            'not feasible for this graph/severity.'
        )
    if len(out) < requested:
        log(
            f'layout catalogue adapted for {prefix}: requested={requested}, '
            f'feasible={len(out)}, minimum={minimum}, attempts={attempts}'
        )
    return out, seen


def build_layout_partitions(
    nodes: pd.DataFrame,
    adj: np.ndarray,
    out_dir: Path,
    subgraph: str,
    severities: list[float],
    regional_seen_fraction: float,
    regional_id_layouts_per_severity: int,
    regional_ood_layouts_per_severity: int,
    directional_id_layouts_per_severity: int,
    directional_ood_layouts_per_severity: int,
    seed: int,
    *,
    ood_severities: list[float] | None = None,
    regional_min_id_layouts_per_severity: int = 16,
    regional_min_ood_layouts_per_severity: int = 8,
    directional_min_id_layouts_per_severity: int = 16,
    directional_min_ood_layouts_per_severity: int = 8,
) -> dict:
    """Build fixed ID/OOD sensing layout catalogues.

    OOD means unseen layout identity at matched severity. It does *not* mean a
    smaller held-out node pool, preventing severity/layout confounding.

    Only severities explicitly listed in ``ood_severities`` receive OOD
    catalogues.  This avoids imposing unnecessary high-severity OOD catalogue
    cardinality constraints when the benchmark never evaluates those strata.
    """
    ood_set = {round(float(x), 8) for x in (ood_severities or [])}
    out_dir = ensure_dir(out_dir); nodes = canonicalize_nodes(nodes); n = len(nodes)
    rng = np.random.default_rng(stable_seed(subgraph, 'regional_center_partition', base=seed))
    centers = rng.permutation(np.arange(n, dtype=np.int64))
    cut = max(1, min(n-1 if n > 1 else 1, int(round(n * regional_seen_fraction))))
    regional_seen, regional_ood = np.sort(centers[:cut]), np.sort(centers[cut:])
    np.save(out_dir / 'regional_centers_id.npy', regional_seen)
    np.save(out_dir / 'regional_centers_ood.npy', regional_ood)

    groups = _direction_groups(nodes)
    write_json(out_dir / 'direction_groups.json', {'groups': groups, 'eligible_node_count': len(set(x for v in groups.values() for x in v))})

    regional_catalog = {'definition': 'fixed compact regional layouts; OOD uses held-out centres at matched severity', 'severities': {}}
    directional_catalog = {'definition': 'fixed direction-group layouts; ID and OOD node layouts are disjoint at matched severity', 'groups': groups, 'severities': {}}

    for sev in map(float, severities):
        token = f's{int(round(100*sev)):02d}'; target = max(1, min(n, int(round(n*sev))))

        def reg_gen(center_pool, split_name):
            def _g(rng_local):
                center = int(center_pool[int(rng_local.integers(0, len(center_pool)))]) if len(center_pool) else int(rng_local.integers(0,n))
                ids, meta = compact_region(adj, nodes, center, target)
                meta.update({'center_pool': split_name})
                return ids, meta
            return _g

        reg_id, reg_seen = _unique_layouts(
            reg_gen(regional_seen, 'id'), regional_id_layouts_per_severity,
            f'{subgraph}|regional|{token}|id', seed,
            min_count=regional_min_id_layouts_per_severity, allow_partial=True,
        )
        if round(sev, 8) in ood_set:
            reg_ood, _ = _unique_layouts(
                reg_gen(regional_ood, 'ood'), regional_ood_layouts_per_severity,
                f'{subgraph}|regional|{token}|ood', seed, forbidden=reg_seen,
                min_count=regional_min_ood_layouts_per_severity, allow_partial=True,
            )
        else:
            reg_ood = []
        regional_catalog['severities'][token] = {'id': [], 'ood': []}
        for dist, layouts in [('id',reg_id),('ood',reg_ood)]:
            for i,(ids,meta) in enumerate(layouts):
                regional_catalog['severities'][token][dist].append({
                    'layout_id': f'regional_{token}_{dist}_{i:03d}', 'nodes': ids.tolist(), **meta,
                })

        def dir_gen(rng_local):
            ids, used = _directional_layout(groups, n, target, rng_local)
            return ids, {'direction_groups': used}

        dir_id, dir_seen = _unique_layouts(
            dir_gen, directional_id_layouts_per_severity,
            f'{subgraph}|directional|{token}|id', seed,
            min_count=directional_min_id_layouts_per_severity, allow_partial=True,
        )
        if round(sev, 8) in ood_set:
            dir_ood, _ = _unique_layouts(
                dir_gen, directional_ood_layouts_per_severity,
                f'{subgraph}|directional|{token}|ood', seed, forbidden=dir_seen,
                min_count=directional_min_ood_layouts_per_severity, allow_partial=True,
            )
        else:
            dir_ood = []
        directional_catalog['severities'][token] = {'id': [], 'ood': []}
        for dist, layouts in [('id',dir_id),('ood',dir_ood)]:
            for i,(ids,meta) in enumerate(layouts):
                directional_catalog['severities'][token][dist].append({
                    'layout_id': f'directional_{token}_{dist}_{i:03d}', 'nodes': ids.tolist(), **meta,
                })

    write_json(out_dir / 'regional_layouts.json', regional_catalog)
    write_json(out_dir / 'directional_layouts.json', directional_catalog)
    manifest = {
        'regional_id_centers': int(len(regional_seen)),
        'regional_ood_centers': int(len(regional_ood)),
        'requested': {
            'regional_id_layouts_per_severity': int(regional_id_layouts_per_severity),
            'regional_ood_layouts_per_severity': int(regional_ood_layouts_per_severity),
            'directional_id_layouts_per_severity': int(directional_id_layouts_per_severity),
            'directional_ood_layouts_per_severity': int(directional_ood_layouts_per_severity),
        },
        'minimum': {
            'regional_id_layouts_per_severity': int(regional_min_id_layouts_per_severity),
            'regional_ood_layouts_per_severity': int(regional_min_ood_layouts_per_severity),
            'directional_id_layouts_per_severity': int(directional_min_id_layouts_per_severity),
            'directional_ood_layouts_per_severity': int(directional_min_ood_layouts_per_severity),
        },
        'actual_counts': {
            token: {
                'regional_id': int(len(regional_catalog['severities'][token]['id'])),
                'regional_ood': int(len(regional_catalog['severities'][token]['ood'])),
                'directional_id': int(len(directional_catalog['severities'][token]['id'])),
                'directional_ood': int(len(directional_catalog['severities'][token]['ood'])),
            }
            for token in regional_catalog['severities']
        },
        'ood_severities': sorted(float(x) for x in ood_set),
        'direction_groups': int(len(groups)),
        'ood_definition': 'unseen fixed sensing layout at matched missing severity',
    }
    write_json(out_dir / 'layout_partitions.json', manifest)
    return manifest


def predictive_source_contribution(dep_ids: np.ndarray, dep_weights: np.ndarray, n_nodes: int) -> np.ndarray:
    """Exact marginal contribution of masking each source node to mean context loss.

    For a fixed spatial outage, ``dependency_context_loss`` is linear in the
    masked-node indicator.  This vector therefore gives the objective increment
    obtained by adding one source node to a predictive layout.  It is derived
    only from the Train-only dependency graph.
    """
    ids=np.asarray(dep_ids,np.int64); w=np.asarray(dep_weights,np.float64)
    out=np.zeros(int(n_nodes),np.float64)
    for target in range(ids.shape[0]):
        good=ids[target]>=0
        if not np.any(good):
            continue
        ii=ids[target,good]; ww=w[target,good].astype(np.float64)
        den=float(ww.sum())
        if den<=0:
            ww=np.ones(len(ii),np.float64); den=float(len(ii))
        ww=ww/max(den,1e-12)
        for node,weight in zip(ii,ww):
            out[int(node)] += float(weight)
    # Chapter-3 C_q^loss averages over all N targets. Targets without a
    # predictive-neighbour set contribute zero, so the exact marginal objective
    # must use N rather than only the number of non-empty targets. This is only
    # a uniform scale change relative to v1.2.2, so existing greedy layout
    # identities are unchanged.
    if int(n_nodes)>0:
        out /= float(n_nodes)
    return out


def _layout_context_loss(ids: np.ndarray, contribution: np.ndarray) -> float:
    arr=np.asarray(ids,np.int64)
    c=np.asarray(contribution,np.float64)
    return float(np.sum(c[arr])) if len(arr) else 0.0


def _predictive_connected_layout(
    adj: np.ndarray,
    nodes: pd.DataFrame,
    target_size: int,
    contribution: np.ndarray,
    rng: np.random.Generator,
    *,
    start_pool_fraction: float=0.35,
    exploration_temperature: float=0.12,
) -> tuple[np.ndarray,dict]:
    """Greedy compact region maximising Train-only predictive-context loss.

    The old v1.2.1 implementation picked a high-centrality centre and then used
    an ordinary BFS, so the objective was quickly diluted when the region grew
    to 40% of the network.  Here every expansion step directly maximises the
    marginal context-loss contribution among the current graph frontier.  A
    small deterministic stochastic perturbation creates multiple fixed layouts
    for ID/OOD catalogues without using Validation/Test outcomes.
    """
    n=int(adj.shape[0]); target_size=max(1,min(int(target_size),n))
    c=np.asarray(contribution,np.float64).reshape(-1)
    if len(c)!=n: raise ValueError(f'contribution length {len(c)} != nodes {n}')
    scale=max(float(np.std(c)),1e-8)
    order=np.argsort(-c)
    pool_n=max(1,min(n,int(round(n*float(start_pool_fraction)))))
    pool=order[:pool_n]
    # Bias the seed strongly toward high-value sources while retaining enough
    # diversity for a fixed-layout catalogue.
    logits=c[pool]/max(scale,1e-8)
    logits=(logits-np.max(logits))/max(float(exploration_temperature),1e-4)
    probs=np.exp(np.clip(logits,-50,50)); probs/=max(float(probs.sum()),1e-12)
    center=int(rng.choice(pool,p=probs))

    selected=[center]; selected_set={center}; frontier=set(int(v) for v in undirected_neighbors(adj,center))
    while len(selected_set)<target_size:
        candidates=np.asarray(sorted(frontier-selected_set),np.int64)
        if not len(candidates):
            # Disconnected graph: start the next component at the highest-value
            # unselected source. This mirrors the general regional protocol's
            # exact-severity behaviour while keeping the predictive objective.
            candidates=np.asarray([i for i in range(n) if i not in selected_set],np.int64)
            if not len(candidates): break
            raw=c[candidates]
            pick=int(candidates[int(np.argmax(raw))])
        else:
            noise=rng.gumbel(0.0,1.0,len(candidates))*float(exploration_temperature)*scale
            score=c[candidates]+noise
            pick=int(candidates[int(np.argmax(score))])
        selected_set.add(pick); selected.append(pick)
        frontier.update(int(v) for v in undirected_neighbors(adj,pick))
        frontier.difference_update(selected_set)

    arr=np.asarray(sorted(selected_set),np.int64)
    if len(arr)>target_size:
        # Should not happen, but keep the contract exact.
        arr=arr[np.argsort(-c[arr])[:target_size]]; arr=np.sort(arr)
    comps=connected_components(adj,arr); sizes=[len(x) for x in comps]; largest=max(sizes) if sizes else 0
    meta={
        'center':int(center),'target_size':int(target_size),'actual_size':int(len(arr)),
        'component_count':int(len(comps)),'largest_component_fraction':float(largest/max(len(arr),1)),
        'predictive_context_loss_objective':_layout_context_loss(arr,c),
        'predictive_contribution_mean':float(np.mean(c[arr])) if len(arr) else 0.0,
        'optimizer':'connected_greedy_context_loss',
    }
    return arr,meta


def _predictive_directional_layout(
    groups: dict[str,list[int]],
    n_nodes: int,
    target_size: int,
    contribution: np.ndarray,
    rng: np.random.Generator,
    *,
    exploration_temperature: float=0.12,
) -> tuple[np.ndarray,list[str],float]:
    """Directional layout directly optimising Train-only context-loss utility."""
    target_size=max(1,min(int(target_size),int(n_nodes)))
    c=np.asarray(contribution,np.float64).reshape(-1)
    eligible=sorted(set(int(x) for vals in groups.values() for x in vals))
    if len(eligible)<target_size:
        raise ValueError(f'Directional groups cover only {len(eligible)}/{n_nodes} nodes but target_size={target_size}')
    scale=max(float(np.std(c)),1e-8)
    scored=[]
    for name,vals in groups.items():
        arr=np.asarray(vals,np.int64)
        if not len(arr): continue
        # Utility density prevents a very large low-value direction group from
        # dominating solely because it contains more nodes.
        density=float(np.mean(c[arr])); total=float(np.sum(c[arr]))
        noise=float(rng.gumbel(0.0,1.0))*float(exploration_temperature)*scale
        scored.append((density+noise,total,name,arr))
    scored.sort(key=lambda z:(-z[0],-z[1],z[2]))
    chosen=[]; used=[]; chosen_set=set()
    for _,_,name,arr in scored:
        remain=target_size-len(chosen_set)
        if remain<=0: break
        available=np.asarray([int(x) for x in arr if int(x) not in chosen_set],np.int64)
        if not len(available): continue
        used.append(name)
        noise=rng.gumbel(0.0,1.0,len(available))*float(exploration_temperature)*scale
        order=np.argsort(-(c[available]+noise))
        take=available[order[:remain]]
        chosen_set.update(map(int,take))
    if len(chosen_set)<target_size:
        rest=np.asarray([i for i in eligible if i not in chosen_set],np.int64)
        order=np.argsort(-c[rest]); chosen_set.update(map(int,rest[order[:target_size-len(chosen_set)]]))
    ids=np.asarray(sorted(chosen_set),np.int64)
    if len(ids)!=target_size:
        raise RuntimeError(f'Failed to construct exact predictive directional layout: {len(ids)} != {target_size}')
    return ids,used,_layout_context_loss(ids,c)


def build_predictive_layout_partitions(
    nodes: pd.DataFrame,
    adj: np.ndarray,
    dep_ids: np.ndarray,
    dep_weights: np.ndarray,
    out_dir: Path,
    subgraph: str,
    severities: list[float],
    seed: int,
    *,
    center_pool_fraction: float=0.35,
    exploration_temperature: float=0.12,
    regional_id_count: int=24,
    regional_ood_count: int=12,
    regional_min_id: int=8,
    regional_min_ood: int=4,
    directional_id_count: int=24,
    directional_ood_count: int=12,
    directional_min_id: int=8,
    directional_min_ood: int=4,
) -> dict:
    """Build future-free predictive structured layouts by direct objective optimisation.

    The objective is the exact mean Train-only predictive-context loss induced
    by a permanent spatial outage. No Validation/Test future, forecast error or
    proposed-model output is used. ID/OOD differ only by fixed layout identity
    at the same missing severity.
    """
    out_dir=ensure_dir(out_dir); nodes=canonicalize_nodes(nodes); n=len(nodes)
    contribution=predictive_source_contribution(dep_ids,dep_weights,n)
    groups=_direction_groups(nodes)
    reg={'definition':'Train-only dependency-context-loss-optimised compact regional layouts','severities':{}}
    direc={'definition':'Train-only dependency-context-loss-optimised directional layouts','groups':groups,'severities':{}}

    for sev in map(float,severities):
        token=f's{int(round(100*sev)):02d}'; target=max(1,min(n,int(round(n*sev))))
        def reg_gen(dist):
            def _g(rng_local):
                ids,meta=_predictive_connected_layout(
                    adj,nodes,target,contribution,rng_local,
                    start_pool_fraction=center_pool_fraction,
                    exploration_temperature=exploration_temperature,
                )
                meta.update({'layout_distribution':dist})
                return ids,meta
            return _g
        reg_id,seen=_unique_layouts(reg_gen('id'),regional_id_count,f'{subgraph}|predictive_regional|{token}|id',seed,min_count=regional_min_id,allow_partial=True)
        reg_ood,_=_unique_layouts(reg_gen('ood'),regional_ood_count,f'{subgraph}|predictive_regional|{token}|ood',seed,forbidden=seen,min_count=regional_min_ood,allow_partial=True)
        reg['severities'][token]={'id':[],'ood':[]}
        for dist,layouts in [('id',reg_id),('ood',reg_ood)]:
            for i,(ids,meta) in enumerate(layouts):
                reg['severities'][token][dist].append({'layout_id':f'predictive_regional_{token}_{dist}_{i:03d}','nodes':ids.tolist(),**meta})

        def dir_gen(rng_local):
            ids,used,obj=_predictive_directional_layout(
                groups,n,target,contribution,rng_local,
                exploration_temperature=exploration_temperature,
            )
            return ids,{'direction_groups':used,'predictive_context_loss_objective':obj,'predictive_contribution_mean':float(np.mean(contribution[ids])) if len(ids) else 0.0,'optimizer':'direction_group_greedy_context_loss'}
        dir_id,dseen=_unique_layouts(dir_gen,directional_id_count,f'{subgraph}|predictive_directional|{token}|id',seed,min_count=directional_min_id,allow_partial=True)
        dir_ood,_=_unique_layouts(dir_gen,directional_ood_count,f'{subgraph}|predictive_directional|{token}|ood',seed,forbidden=dseen,min_count=directional_min_ood,allow_partial=True)
        direc['severities'][token]={'id':[],'ood':[]}
        for dist,layouts in [('id',dir_id),('ood',dir_ood)]:
            for i,(ids,meta) in enumerate(layouts):
                direc['severities'][token][dist].append({'layout_id':f'predictive_directional_{token}_{dist}_{i:03d}','nodes':ids.tolist(),**meta})

    write_json(out_dir/'predictive_regional_layouts.json',reg)
    write_json(out_dir/'predictive_directional_layouts.json',direc)
    write_json(out_dir/'predictive_layout_metadata.json',{
        'definition':'support-oriented structured outages directly optimising Train-only predictive-context loss',
        'objective':'mean per-target dependency weight removed by the fixed spatial outage',
        'future_free':True,
        'center_pool_fraction':float(center_pool_fraction),
        'exploration_temperature':float(exploration_temperature),
        'severities':[float(x) for x in severities],
        'source_contribution_mean':float(np.mean(contribution)),
        'source_contribution_max':float(np.max(contribution)) if len(contribution) else 0.0,
    })
    return {'regional':reg,'directional':direc}

