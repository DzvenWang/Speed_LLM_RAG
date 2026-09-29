#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from supportbench.config import discover_source_root, load_config
from supportbench.graph import build_directed_neighbors, build_layout_partitions
from supportbench.splits import chronological_splits
from supportbench.utils import (
    canonicalize_nodes,
    ensure_dir,
    link_or_copy,
    load_channel_names,
    log,
    read_json,
    write_json,
)


def streaming_stats(arr: np.ndarray, valid: np.ndarray, start: int, end: int, chunk: int = 4096):
    c = arr.shape[2]
    count = np.zeros(c, np.int64); total = np.zeros(c, np.float64); total2 = np.zeros(c, np.float64)
    vmin = np.full(c, np.inf); vmax = np.full(c, -np.inf)
    for s in range(start, end, chunk):
        e = min(end, s+chunk)
        x = np.asarray(arr[s:e], np.float32); m = np.asarray(valid[s:e], bool)
        for j in range(c):
            vals = x[...,j][m[...,j]].astype(np.float64, copy=False)
            if not vals.size: continue
            count[j] += vals.size; total[j] += vals.sum(); total2[j] += np.square(vals).sum()
            vmin[j] = min(vmin[j], float(vals.min())); vmax[j] = max(vmax[j], float(vals.max()))
    mean = np.divide(total, count, out=np.zeros_like(total), where=count>0)
    var = np.divide(total2, count, out=np.zeros_like(total2), where=count>0) - mean**2
    std = np.sqrt(np.maximum(var, 1e-12))
    return count, mean, std, vmin, vmax


def build_valid_mask(traffic_path: Path, out_path: Path, speed_idx: int, cfg: dict) -> None:
    x = np.load(traffic_path, mmap_mode='r')
    H = x.shape
    if out_path.exists() or out_path.is_symlink(): out_path.unlink()
    mm = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.uint8, shape=H)
    chunk = 2048
    zero_frac_thr = float(cfg['forecast']['synchronised_zero_node_fraction'])
    invalidate = bool(cfg['forecast']['invalidate_synchronised_zero'])
    sync_count = 0
    for s in range(0, H[0], chunk):
        e = min(H[0], s+chunk)
        a = np.asarray(x[s:e], np.float32)
        valid = np.isfinite(a)
        if invalidate:
            speed = a[...,speed_idx]
            finite_speed = np.isfinite(speed)
            zero_by_node = (speed == 0) & finite_speed
            frac = zero_by_node.mean(axis=1)
            bad_t = frac >= zero_frac_thr
            sync_count += int(bad_t.sum())
            if np.any(bad_t): valid[bad_t,:,:] = False
        mm[s:e] = valid.astype(np.uint8)
    mm.flush()
    log(f'valid mask built: {out_path}; synchronized-zero timestamps invalidated={sync_count}')


def discover_subgraphs(root: Path, cfg: dict) -> list[Path]:
    expected = list(cfg['source'].get('expected_subgraphs', []))
    items = sorted([p for p in root.glob('TraffiDent_Net_*') if p.is_dir()], key=lambda p:p.name)
    if expected:
        missing = [x for x in expected if not (root/x).is_dir()]
        if missing: raise FileNotFoundError(f'Missing expected subgraphs under {root}: {missing}')
        items = [root/x for x in expected]
    return items


def build_one(src: Path, cfg: dict) -> None:
    out_root = Path(cfg['project']['benchmark_root']).resolve(); diag_root = Path(cfg['project']['diagnostics_root']).resolve()
    dst = ensure_dir(out_root / src.name); ensure_dir(diag_root / src.name)
    core = ensure_dir(dst/'core'); graph = ensure_dir(dst/'graph'); splits_dir = ensure_dir(dst/'splits')
    sf = cfg['source']; traffic_src=src/sf['traffic_file']; nodes_src=src/sf['nodes_file']
    if not traffic_src.exists() or not nodes_src.exists(): raise FileNotFoundError(f'{src}: missing {traffic_src.name} or {nodes_src.name}')
    traffic = np.load(traffic_src, mmap_mode='r')
    if traffic.ndim != 3: raise ValueError(f'{traffic_src}: expected [T,N,C], got {traffic.shape}')
    T,N,C = traffic.shape
    channels = load_channel_names(src, C, sf.get('channel_mapping_file','channel_mapping.json'))
    primary = str(sf['primary_channel']).lower()
    if primary not in channels: raise ValueError(f'Primary channel {primary} not in {channels}')
    speed_idx = channels.index(primary)

    link_or_copy(traffic_src, core/'traffic_2024.npy', sf.get('link_mode','symlink'))
    nodes = canonicalize_nodes(pd.read_csv(nodes_src))
    if len(nodes) != N: raise ValueError(f'{src.name}: nodes {len(nodes)} != traffic N {N}')
    nodes.to_csv(core/'nodes.csv', index=False)
    if (src/sf.get('channel_mapping_file','channel_mapping.json')).exists():
        link_or_copy(src/sf.get('channel_mapping_file','channel_mapping.json'), core/'channel_mapping.json', sf.get('link_mode','symlink'))
    # Optional source metadata is preserved when present; it is not required for the benchmark.
    for source_name, target_name in [('time_index_2024.csv','time_index.csv'), ('dataset_summary.json','source_dataset_summary.json')]:
        if (src/source_name).exists():
            link_or_copy(src/source_name, core/target_name, sf.get('link_mode','symlink'))

    adj_src = src/sf['adjacency_file']
    if not adj_src.exists(): raise FileNotFoundError(adj_src)
    link_or_copy(adj_src, graph/'adjacency.npy', sf.get('link_mode','symlink'))
    dist_name = sf.get('distance_file')
    dist_src = (src / dist_name) if dist_name else None
    distance = None
    if dist_src is not None and dist_src.exists():
        link_or_copy(dist_src, graph/'distance.npy', sf.get('link_mode','symlink')); distance=np.load(dist_src, mmap_mode='r')
    adj = np.load(adj_src, mmap_mode='r')
    if adj.shape != (N,N): raise ValueError(f'{adj_src}: shape {adj.shape} != {(N,N)}')

    valid_path = core/'raw_valid_mask.npy'
    build_valid_mask(traffic_src, valid_path, speed_idx, cfg)
    valid = np.load(valid_path, mmap_mode='r').astype(bool)
    splits = chronological_splits(T, float(cfg['forecast']['train_ratio']), float(cfg['forecast']['val_ratio']), float(cfg['forecast']['test_ratio']))
    write_json(splits_dir/'time_splits.json', splits)
    tr=splits['train']
    count,mean,std,vmin,vmax = streaming_stats(traffic, valid, int(tr['start']), int(tr['end_exclusive']))
    stats={'train_interval':[int(tr['start']),int(tr['end_exclusive'])], 'channels':{}}
    for i,name in enumerate(channels):
        stats['channels'][name]={'count':int(count[i]),'mean':float(mean[i]),'std':float(std[i]),'min':float(vmin[i]),'max':float(vmax[i])}
    write_json(core/'normalization.json', stats)

    d = build_directed_neighbors(nodes, np.asarray(adj), np.asarray(distance) if distance is not None else None, topk=4)
    np.savez_compressed(graph/'directed_neighbors.npz', **d)

    # Preserve the original SupportBench general structured-layout generation exactly.
    # Predictive/RAG layouts are intentionally not built in Base_SupportBench.
    scfg=cfg['sensing']
    build_layout_partitions(
        nodes, np.asarray(adj), graph/'layouts', src.name,
        list(map(float, scfg['severities'])),
        float(scfg['regional_seen_fraction']),
        int(scfg['regional_id_layouts_per_severity']),
        int(scfg['regional_ood_layouts_per_severity']),
        int(scfg['directional_id_layouts_per_severity']),
        int(scfg['directional_ood_layouts_per_severity']),
        int(scfg['global_seed']),
        ood_severities=list(map(float, scfg.get('ood_severities',[scfg['main_severity']]))),
        regional_min_id_layouts_per_severity=int(scfg.get('regional_min_id_layouts_per_severity',16)),
        regional_min_ood_layouts_per_severity=int(scfg.get('regional_min_ood_layouts_per_severity',8)),
        directional_min_id_layouts_per_severity=int(scfg.get('directional_min_id_layouts_per_severity',16)),
        directional_min_ood_layouts_per_severity=int(scfg.get('directional_min_ood_layouts_per_severity',8)),
        min_layout_overrides=scfg.get('layout_min_overrides',{}),
    )

    manifest={
        'benchmark':'Base-SupportBench','version':'1.2.1','subgraph':src.name,'source_dir':str(src.resolve()),
        'shape':[int(T),int(N),int(C)],'channels':channels,'primary_channel':primary,'primary_channel_index':speed_idx,
        'history_len':int(cfg['forecast']['history_len']),'horizon':int(cfg['forecast']['horizon']),'freq_minutes':int(cfg['forecast']['freq_minutes']),
        'controlled_complete_min_fraction':float(cfg['forecast']['controlled_complete_min_fraction']),
        'design':'canonical traffic facts + five general incomplete-sensing families for Backbone/Baseline training and evaluation only',
    }
    write_json(dst/'manifest.json', manifest)
    log(f'core built {src.name}: T={T} N={N} C={C}')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', default=None); ap.add_argument('--subgraph', action='append', default=[])
    args=ap.parse_args(); cfg=load_config(args.config); source=discover_source_root(cfg); items=discover_subgraphs(source,cfg)
    if args.subgraph: items=[p for p in items if p.name in set(args.subgraph)]
    ensure_dir(cfg['project']['benchmark_root']); ensure_dir(cfg['project']['diagnostics_root'])
    log(f'source_root={source}')
    for p in items: build_one(p,cfg)
    write_json(Path(cfg['project']['benchmark_root'])/'benchmark_manifest.json', {
        'benchmark':'Base-SupportBench','version':'1.2.1','subgraphs':[p.name for p in items],
        'history_len':int(cfg['forecast']['history_len']),'horizon':int(cfg['forecast']['horizon']),
        'split':'70/10/20 chronological','primary_channel':cfg['source']['primary_channel'],
        'canonical_data_rule':'Backbone/Baseline base data only; RAG/retrieval/LLM derived data are intentionally excluded',
    })

if __name__=='__main__': main()
