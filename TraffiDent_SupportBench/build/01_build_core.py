#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from supportbench.config import discover_source_root, load_config
from supportbench.graph import build_directed_neighbors, build_layout_partitions, build_predictive_layout_partitions
from supportbench.dependency import build_train_predictive_dependency
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

    dep_cfg=cfg['dependency']
    dep_meta=build_train_predictive_dependency(
        traffic, valid, speed_idx, int(tr['start']), int(tr['end_exclusive']),
        d['upstream_ids'], d['upstream_weights'], d['downstream_ids'], d['downstream_weights'],
        graph/'predictive_dependency.npz',
        max_neighbors=int(dep_cfg['max_neighbors']), max_lag_steps=int(dep_cfg['max_lag_steps']),
        sample_stride=int(dep_cfg['sample_stride']), min_pairs=int(dep_cfg['min_pairs']),
        corr_floor=float(dep_cfg['corr_floor']), graph_weight_mix=float(dep_cfg['graph_weight_mix']),
        freq_minutes=int(cfg['forecast']['freq_minutes']),
    )
    write_json(graph/'predictive_dependency.json', dep_meta)

    build_layout_partitions(
        nodes, np.asarray(adj), graph/'layouts', src.name,
        list(map(float, cfg['sensing']['severities'])),
        float(cfg['sensing']['regional_seen_fraction']),
        int(cfg['sensing']['regional_id_layouts_per_severity']),
        int(cfg['sensing']['regional_ood_layouts_per_severity']),
        int(cfg['sensing']['directional_id_layouts_per_severity']),
        int(cfg['sensing']['directional_ood_layouts_per_severity']),
        int(cfg['sensing']['global_seed']),
        ood_severities=list(map(float, cfg['sensing'].get('ood_severities', [cfg['sensing']['main_severity']]))),
        regional_min_id_layouts_per_severity=int(cfg['sensing'].get('regional_min_id_layouts_per_severity', 16)),
        regional_min_ood_layouts_per_severity=int(cfg['sensing'].get('regional_min_ood_layouts_per_severity', 8)),
        directional_min_id_layouts_per_severity=int(cfg['sensing'].get('directional_min_id_layouts_per_severity', 16)),
        directional_min_ood_layouts_per_severity=int(cfg['sensing'].get('directional_min_ood_layouts_per_severity', 8)),
        min_layout_overrides=cfg['sensing'].get('layout_min_overrides', {}),
    )

    dep_npz=np.load(graph/'predictive_dependency.npz')
    ps=cfg['sensing']
    build_predictive_layout_partitions(
        nodes, np.asarray(adj), dep_npz['ids'], dep_npz['weights'], graph/'layouts', src.name,
        list(map(float,ps.get('predictive_severities',[ps['main_severity']]))), int(ps['global_seed']),
        center_pool_fraction=float(ps.get('predictive_center_pool_fraction',0.35)),
        exploration_temperature=float(ps.get('predictive_layout_exploration_temperature',0.12)),
        directional_exploration_temperature=float(ps.get('predictive_directional_exploration_temperature',ps.get('predictive_layout_exploration_temperature',0.12))),
        directional_candidate_multiplier=int(ps.get('predictive_directional_candidate_multiplier',64)),
        directional_anchor_pool_fraction=float(ps.get('predictive_directional_anchor_pool_fraction',0.25)),
        directional_restart_penalty=float(ps.get('predictive_directional_restart_penalty',0.05)),
        directional_continuity_bonus=float(ps.get('predictive_directional_continuity_bonus',0.02)),
        directional_min_segment_nodes=int(ps.get('predictive_directional_min_segment_nodes',3)),
        directional_max_segments=int(ps.get('predictive_directional_max_segments',24)),
        directional_target_ratio=float(dep_cfg.get('predictive_context_min_ratio',1.10)),
        regional_id_count=int(ps.get('predictive_regional_id_layouts_per_severity',24)),
        regional_ood_count=int(ps.get('predictive_regional_ood_layouts_per_severity',12)),
        regional_min_id=int(ps.get('predictive_regional_min_id_layouts_per_severity',8)),
        regional_min_ood=int(ps.get('predictive_regional_min_ood_layouts_per_severity',4)),
        directional_id_count=int(ps.get('predictive_directional_id_layouts_per_severity',24)),
        directional_ood_count=int(ps.get('predictive_directional_ood_layouts_per_severity',12)),
        directional_min_id=int(ps.get('predictive_directional_min_id_layouts_per_severity',8)),
        directional_min_ood=int(ps.get('predictive_directional_min_ood_layouts_per_severity',4)),
        ood_severities=list(map(float,ps.get('ood_severities',[ps['main_severity']]))),
        min_layout_overrides=ps.get('layout_min_overrides', {}),
    )

    manifest={
        'benchmark':'TraffiDent-SupportBench','version':'1.2.6','subgraph':src.name,'source_dir':str(src.resolve()),
        'shape':[int(T),int(N),int(C)],'channels':channels,'primary_channel':primary,'primary_channel_index':speed_idx,
        'history_len':int(cfg['forecast']['history_len']),'horizon':int(cfg['forecast']['horizon']),'freq_minutes':int(cfg['forecast']['freq_minutes']),
        'controlled_complete_min_fraction':float(cfg['forecast']['controlled_complete_min_fraction']),
        'design':'canonical facts + general/support-oriented sensing views + Train-only predictive dependency + Train-only historical support memory; model-specific embeddings/utility labels are derived artifacts',
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
        'benchmark':'TraffiDent-SupportBench','version':'1.2.6','subgraphs':[p.name for p in items],
        'history_len':int(cfg['forecast']['history_len']),'horizon':int(cfg['forecast']['horizon']),
        'split':'70/10/20 chronological','primary_channel':cfg['source']['primary_channel'],
        'canonical_data_rule':'facts/protocols only; learned retrieval and LLM artifacts are external derived data',
    })

if __name__=='__main__': main()
