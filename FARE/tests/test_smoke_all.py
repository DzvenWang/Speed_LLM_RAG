"""Pure-file validation tests for the combined speed/flow smoke controller."""
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'smoke_all.py'
spec = importlib.util.spec_from_file_location('fare_smoke_all', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_phase_contract():
    with tempfile.TemporaryDirectory() as temp:
        run = Path(temp)
        bad = mod.validate_smoke_run(run)
        assert not bad['ok']
        (run / 'phase_summary.json').write_text(json.dumps([{'phase': x} for x in mod.PHASES]), encoding='utf-8')
        for phase in mod.PHASES:
            (run / f'{phase}_best.pt').write_bytes(b'test-fixture')
            folder = run / 'stages' / phase
            folder.mkdir(parents=True)
            (folder / 'val_summary.json').write_text(json.dumps({'final': {
                'MAE': 1.2, 'RMSE': 2.4, 'directly_masked_MAE': 1.3
            }}), encoding='utf-8')
        (run / 'candidate_best.pt').write_bytes(b'test-fixture')
        good = mod.validate_smoke_run(run)
        assert good['ok'], good['problems']
        assert len(good['phase_metrics']) == len(mod.PHASES)
        (run / 'stages' / 'recovery' / 'val_summary.json').write_text(json.dumps({'final': {
            'MAE': math.nan, 'RMSE': 2.4, 'directly_masked_MAE': 1.3
        }}), encoding='utf-8')
        invalid = mod.validate_smoke_run(run)
        assert not invalid['ok']
        assert any('Nonfinite' in s for s in invalid['problems'])


def test_cli_defaults():
    args = mod.parse_args([])
    assert args.targets == ['speed', 'flow']
    assert args.dataset == 'TraffiDent_Net_06'
    assert 'TraffiDent_Net_01' in mod.SUPPORTED_DATASETS


if __name__ == '__main__':
    test_phase_contract()
    test_cli_defaults()
    print('FARE_SMOKE_ALL_TEST_PASS')
