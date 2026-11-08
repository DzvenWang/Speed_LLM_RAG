"""Regression: Gadi v3 One-Shot jobs failed after s20 GPU inference
because protocol_queries.csv's per-severity parent was never created.
"""
from pathlib import Path


def test_severity_directory_created_before_protocol_csv():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "oneshot_v3.py").read_text()
    start = script.index("for sev in args.severities:")
    create = script.index("outdir.mkdir(parents=True, exist_ok=True)", start)
    write = script.index("pd.DataFrame(proto_rows).to_csv(outdir/'protocol_queries.csv'", start)
    assert start < create < write


def test_official_mismatch_logging_cannot_reference_undefined_meth():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "oneshot_v3.py").read_text()
    assert "f'{meth}_OFFICIAL_REPRO_MISMATCH" not in script
    assert "_OFFICIAL_REPRO_MISMATCH" in script
