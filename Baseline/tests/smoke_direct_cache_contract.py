from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for base in [ROOT/'baseline',ROOT/'scripts',ROOT/'pbs']:
    for f in base.rglob('*'):
        if not f.is_file() or f.suffix not in {'.py','.sh','.pbs'}:
            continue
        if f.name == 'cleanup_legacy_cache_aliases.py':
            continue
        t=f.read_text(errors='ignore')
        if "'_cache'" in t or '"_cache"' in t or '/_cache/' in t or 'ensure_consumer_alias' in t:
            raise SystemExit(f'FAIL local cache alias reference remains in {f}')
helper=(ROOT/'baseline/shared_cache.py').read_text()
assert 'experiment_common.cache' in helper and 'cache_dir' in helper
print('BASELINE DIRECT EXPERIMENTCACHE CONTRACT SMOKE PASS')
