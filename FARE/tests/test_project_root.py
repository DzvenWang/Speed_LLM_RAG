"""Project root must default to the parent of FARE, not the scratch home."""
from pathlib import Path
from fare.config import PACKAGE_ROOT, root

def test_default_root_is_workspace(monkeypatch):
    monkeypatch.delenv("SPEED_LLM_RAG_ROOT", raising=False)
    assert root() == PACKAGE_ROOT.parent

def test_explicit_workspace_root(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEED_LLM_RAG_ROOT", str(tmp_path))
    assert root() == tmp_path.resolve()
