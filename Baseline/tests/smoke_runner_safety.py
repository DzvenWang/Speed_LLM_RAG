from pathlib import Path
root=Path(__file__).resolve().parents[1]
for m in ['ginar','raft','saraf']:
    p=root/'baseline'/'runners'/f'{m}_runner.py'; s=p.read_text()
    assert "development" in s and "formal" in s
    assert "test_access" in s or 'no Test dataset' in s
for m in ['raft','saraf']:
    s=(root/'baseline'/'runners'/f'{m}_runner.py').read_text(); assert 'window_start' in s and 'overlap' in s and 'full' in s and 'obs' in s
assert (root/'baseline/runners/imputeformer_runner.py').exists()
assert (root/'baseline/runners/rast_runner.py').exists()
assert 'evaluation' not in (root/'baseline/runners/imputeformer_runner.py').read_text().split('choices=')[1].split(')')[0]
print('BASELINE RUNNER SAFETY SMOKE PASS')
