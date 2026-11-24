from types import SimpleNamespace
import pytest
from loopbench.backends.transformers_backend import (
    TransformersBackend,
    apply_loop_num_passes,
    clear_model_cache,
    parse_inkling_output,
)


class _DummyTokenizer:
    _MAP = {
        "<|content_thinking|>": 200008,
        "<|content_text|>": 200004,
        "<|end_message|>": 200010,
        "<|content_model_end_sampling|>": 200006,
        "<|unused_200054|>": 200054,
    }

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self._MAP.get(tok, 999)

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "<|message_model|><|content_thinking|>step1 step2<|end_message|><|message_model|><|content_text|>60<|end_message|><|content_model_end_sampling|>"


def test_parse_inkling_output_splits_thinking_and_answer():
    tok = _DummyTokenizer()
    gen_ids = [200001, 200008, 11, 12, 13, 200010, 200001, 200004, 42, 200010, 200006]
    raw = tok.decode(gen_ids)
    out = parse_inkling_output(gen_ids, tok, raw)
    assert out["reasoning_tokens"] == 3
    assert out["answer_tokens"] == 1
    assert out["text"] == "60"
    assert out["reasoning_text"] == "step1 step2"


def test_apply_loop_num_passes_updates_nested_configs():
    lm_cfg = SimpleNamespace(loop_num_passes=3)
    text_cfg = SimpleNamespace(loop_num_passes=3)
    model = SimpleNamespace(
        config=SimpleNamespace(text_config=text_cfg),
        model=SimpleNamespace(language_model=SimpleNamespace(config=lm_cfg)),
    )
    eff_1 = apply_loop_num_passes(model, 1)
    assert eff_1 == 24
    assert model.config.text_config.loop_num_passes == 1
    assert model.model.language_model.config.loop_num_passes == 1

    eff_5 = apply_loop_num_passes(model, 5)
    assert eff_5 == 72
    assert model.config.text_config.loop_num_passes == 5
    assert model.model.language_model.config.loop_num_passes == 5

    with pytest.raises(ValueError):
        apply_loop_num_passes(model, 0)
    clear_model_cache()
