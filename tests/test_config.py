import pytest
from loopbench.config import load_config
def test_mock_config():
    c=load_config("configs/mock.yaml")
    assert c["backend"]=="mock"
def test_requires_explicit_values(tmp_path):
    p=tmp_path/"bad.yaml";p.write_text("backend: mock\n")
    with pytest.raises(ValueError):load_config(p)
