from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def test_formal_and_smoke_pbs_use_audited_storage_python_and_pipefail():
    for f in ('formal_v7.pbs','smoke_v7.pbs'):
        p=ROOT/'pbs'/f
        data=p.read_text()
        assert '#PBS -l storage=scratch/mh20+gdata/mh20' in data
        assert '#PBS -P mh20' in data
        assert 'set -euo pipefail' in data
        assert 'TFP' in data and '/bin/python' in data
    assert 'gpu-smoke-only' in (ROOT/'pbs'/'smoke_v7.pbs').read_text()


def test_v7_uses_existing_v6_data_cache_without_short_smoke_collision():
    code=(ROOT/'pcft_stla/v7/pipeline.py').read_text()
    assert 'prepare(cfg,checkpoint,smoke=False)' in code
