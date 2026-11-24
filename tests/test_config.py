import pytest
from loopbench.backends import create_backend
from loopbench.config import load_config, validate_runtime_ready


def test_mock_config():
    c = load_config("configs/mock.yaml")
    assert c["backend"] == "mock"
    assert c["image_mode"] == "correct"


def test_requires_explicit_values(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("backend: mock\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing explicit config fields"):
        load_config(p)


def test_smoke_config_blocks_unpinned_gpu_execution():
    c = load_config("configs/smoke.yaml")
    assert c["backend"] == "transformers"
    with pytest.raises(ValueError, match="model_revision"):
        validate_runtime_ready(c)
    c_pinned = {**c, "model_revision": "abcdef1234567890"}
    with pytest.raises(NotImplementedError, match="Phase 2"):
        create_backend(c_pinned)


def test_invalid_loop_passes_and_image_mode(tmp_path):
    c = load_config("configs/mock.yaml")
    bad_pass = tmp_path / "bad_pass.yaml"
    bad_pass.write_text(
        "\n".join(f"{k}: {v!r}" for k, v in {**c, "loop_num_passes": 4}.items()),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="loop_num_passes"):
        load_config(bad_pass)
