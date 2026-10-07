from pathlib import Path

def test_no_forbidden_runtime_imports():
    root=Path(__file__).resolve().parents[1]/'fare'
    forbidden=('from baseline','import baseline','STLA_DASR','Support_Oracle')
    bad=[]
    for p in root.rglob('*.py'):
        s=p.read_text(encoding='utf-8')
        for t in forbidden:
            if t in s: bad.append((str(p),t))
    assert not bad,bad
