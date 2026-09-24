from __future__ import annotations
import hashlib
import numpy as np


def _dense_adj(adj):
    if hasattr(adj, 'toarray'):
        a = adj.toarray()
    elif hasattr(adj, 'A'):
        a = np.asarray(adj.A)
    else:
        a = np.asarray(adj)
    # Some datasets expose multiple adjacency channels.
    if a.ndim == 3:
        a = np.max(np.abs(a), axis=0)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f'Expected square adjacency, got shape={a.shape}')
    a = np.asarray(a, np.float64)
    a[~np.isfinite(a)] = 0.0
    a = np.abs(a)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0.0)
    return a


def _kmeans(x, k, seed=10, iters=80):
    """Small deterministic numpy k-means used to avoid a new dependency."""
    x = np.asarray(x, np.float64)
    n = len(x)
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    # k-means++ initialization.
    centers = [int(rng.integers(n))]
    dist2 = ((x - x[centers[0]]) ** 2).sum(1)
    for _ in range(1, k):
        s = float(dist2.sum())
        idx = int(rng.integers(n)) if s <= 1e-12 else int(rng.choice(n, p=dist2 / s))
        centers.append(idx)
        dist2 = np.minimum(dist2, ((x - x[idx]) ** 2).sum(1))
    c = x[np.asarray(centers)].copy()
    labels = np.zeros(n, np.int32)
    for _ in range(iters):
        d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(2)
        new = d.argmin(1).astype(np.int32)
        # Repair empty clusters with farthest points.
        counts = np.bincount(new, minlength=k)
        if np.any(counts == 0):
            nearest = d[np.arange(n), new]
            order = np.argsort(-nearest)
            used = set()
            for r in np.flatnonzero(counts == 0):
                for idx in order:
                    if int(idx) not in used and counts[new[idx]] > 1:
                        counts[new[idx]] -= 1
                        new[idx] = int(r)
                        counts[r] += 1
                        used.add(int(idx))
                        break
        nc = np.stack([x[new == j].mean(0) for j in range(k)], 0)
        if np.array_equal(new, labels) and np.allclose(nc, c, atol=1e-8):
            labels = new
            break
        labels, c = new, nc
    # Canonicalize labels by first node index so reruns are stable.
    first = [(int(np.flatnonzero(labels == r)[0]), r) for r in range(k)]
    remap = {old: new for new, (_, old) in enumerate(sorted(first))}
    return np.asarray([remap[int(v)] for v in labels], np.int32)


def spectral_graph_regions(adj, n_regions=24, seed=10):
    """Partition nodes from the fixed traffic graph only (no Val/Test labels)."""
    a = _dense_adj(adj)
    n = a.shape[0]
    k = max(1, min(int(n_regions), n))
    deg = a.sum(1)
    # Handle isolated nodes without numerical blow-up.
    inv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    s = inv[:, None] * a * inv[None, :]
    try:
        vals, vecs = np.linalg.eigh(s)
        take = np.argsort(vals)[-k:]
        emb = vecs[:, take]
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        labels = _kmeans(emb, k, seed=seed)
        method = 'spectral_graph_kmeans_numpy'
    except Exception:
        # Safe deterministic fallback; keeps the experiment runnable but records it.
        labels = (np.arange(n) * k // n).astype(np.int32)
        method = 'balanced_index_fallback'
    return labels, method


def region_manifest(labels, method, adj=None):
    labels = np.asarray(labels, np.int32)
    r = int(labels.max()) + 1
    payload = {
        'method': method,
        'n_nodes': int(len(labels)),
        'n_regions': r,
        'region_sizes': [int((labels == i).sum()) for i in range(r)],
        'labels_sha256': hashlib.sha256(labels.tobytes()).hexdigest(),
        'uses_validation_future': False,
        'uses_test': False,
    }
    if adj is not None:
        a = _dense_adj(adj).astype(np.float32)
        payload['adjacency_sha256'] = hashlib.sha256(a.tobytes()).hexdigest()
    return payload
