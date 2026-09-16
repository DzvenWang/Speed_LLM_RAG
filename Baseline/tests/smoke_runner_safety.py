from pathlib import Path
root=Path(__file__).resolve().parents[1]
for m in ['ginar','raft','saraf','sprint','mogernn','remft','lrtc_tmcp']:
    p=root/'baseline'/'runners'/f'{m}_runner.py'; s=p.read_text()
    assert "development" in s and "formal" in s
    assert "test_access" in s or 'no Test dataset' in s
for m in ['raft','saraf']:
    s=(root/'baseline'/'runners'/f'{m}_runner.py').read_text(); assert 'window_start' in s and 'overlap' in s and 'full' in s and 'obs' in s
for m in ['imputeformer','lrtc_tmcp','rast','sprint']:
    assert (root/'baseline'/'runners'/f'{m}_runner.py').exists()
assert 'evaluation' not in (root/'baseline/runners/imputeformer_runner.py').read_text().split('choices=')[1].split(')')[0]
assert "choices=['development', 'formal']" in (root/'baseline/runners/sprint_runner.py').read_text()
assert 'batch_size must remain 1' in (root/'baseline/runners/mogernn_runner.py').read_text()
assert "'# load data'" in (root/'baseline/runners/remft_runner.py').read_text()
print('BASELINE RUNNER SAFETY SMOKE PASS')
