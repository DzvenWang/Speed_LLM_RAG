"""Pilot val/test report and retrieval comparison tests; no GPU or dataset dependency."""
import json
import tempfile
from pathlib import Path
from importlib.util import spec_from_file_location,module_from_spec
ROOT=Path(__file__).resolve().parents[1]


def _load(name, path):
    spec=spec_from_file_location(name,ROOT/'scripts'/path)
    module=module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_reports():
    ev=_load('pilot_report','report_pilot_val_test.py')
    attr=_load('attr_report','report_retrieval_attribution.py')
    with tempfile.TemporaryDirectory() as t:
        run=Path(t);folder=run/'evaluation_candidate';folder.mkdir()
        def write(path,delta):
            path.parent.mkdir(parents=True,exist_ok=True)
            data={'rows':5000,'final':{'MAE':10+delta,'RMSE':15+delta},
                  'base':{'MAE':11+delta,'RMSE':16+delta},
                  'candidate':{'MAE':12+delta,'RMSE':17+delta}}
            path.write_text(json.dumps(data))
        write(folder/'val_full_summary.json',0)
        for severity in (20,40,60,90):
            write(folder/f's{severity:02d}'/f's{severity:02d}_full_summary.json',severity/100)
        md,results=ev.create_report(run)
        assert md.exists() and len(results)==5
        assert {x['severity'] for x in results if x['split']=='test'}=={'s20','s40','s60','s90'}
        assert all(x['gain_MAE_pct']>0 for x in results)
        modes=('full','random_memory','no_memory','utility_off','similarity_memory')
        rows=[]
        for split,severities in [('val',['']),('test',[20,40,60,90])]:
            for severity in severities:
                for idx,mode in enumerate(modes):
                    rows.append({'split':split,'severity':severity,
                                 'mode':mode,'MAE':10+idx+(.1*float(severity or 0)),
                                 'RMSE':15+idx})
        report=attr.build_report(rows)
        assert 's20' in report and 's90' in report
        assert 'Test severity macro-average' in report
        assert 'Test is exploratory' in report
        assert 'real' not in report.lower() or 'real retrieval' in report.lower()
        print('FARE_PILOT_VAL_TEST_REPORT_TEST_PASS')


if __name__=='__main__':test_reports()
