"""Project root must default to the parent of FARE, not the scratch home."""
from pathlib import Path
from fare.config import PACKAGE_ROOT, root

def test_default_root_is_workspace(monkeypatch):
    monkeypatch.delenv("SPEED_LLM_RAG_ROOT", raising=False)
    assert root() == PACKAGE_ROOT.parent

def test_explicit_workspace_root(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEED_LLM_RAG_ROOT", str(tmp_path))
    assert root() == tmp_path.resolve()


def test_repo_relative_config_works_from_elsewhere(monkeypatch,tmp_path):
    from fare.config import read_config, project_path, PACKAGE_ROOT
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('SPEED_LLM_RAG_ROOT',raising=False)
    assert read_config('FARE/configs/pilot.yaml')['version']=='4.0.1'
    assert read_config('configs/pilot.yaml')['version']=='4.0.1'
    assert project_path('FARE/configs/pilot.yaml',must_exist=True) == (PACKAGE_ROOT/'configs/pilot.yaml').resolve()


def test_project_relative_checkpoint_resolution(monkeypatch,tmp_path):
    from fare.config import project_path
    (tmp_path/'Backbone_exps').mkdir()
    path=tmp_path/'Backbone_exps'/'best.pt'
    path.write_bytes(b'dummy')
    monkeypatch.setenv('SPEED_LLM_RAG_ROOT',str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)
    assert project_path('Backbone_exps/best.pt',must_exist=True)==path
