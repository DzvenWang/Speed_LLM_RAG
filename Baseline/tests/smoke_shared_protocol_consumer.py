from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=(root/'scripts/prepare_data.py').read_text()
assert 'experiment_common.cache' in p
assert 'ensure_consumer_alias' not in p
assert "'_cache'" not in p and '"_cache"' not in p
helper=(root/'baseline/shared_cache.py').read_text()
assert 'experiment_common.cache' in helper and 'cache_dir' in helper
assert 'local_sampling' not in p
print('SHARED PROTOCOL/DIRECT CACHE CONSUMER SMOKE PASS')
