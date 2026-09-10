from pathlib import Path
root=Path(__file__).resolve().parents[1]
for m in ['ginar','raft','saraf']:
    p=root/'baseline'/'runners'/f'{m}_runner.py'; s=p.read_text()
    assert "choices=['development','formal']" in s or "choices=['development', 'formal']" in s
    assert "'test'" not in s.lower().replace("'test_access':False",'').replace("'test_access': False",'') or 'no Test dataset' in s
for m in ['raft','saraf']:
    s=(root/'baseline'/'runners'/f'{m}_runner.py').read_text()
    assert 'window_start' in s and 'overlap' in s and 'full' in s and 'obs' in s
print('BASELINE RUNNER SAFETY SMOKE PASS')
