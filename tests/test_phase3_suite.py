import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from loopbench.backends.base import Backend
from loopbench.backends.transformers_backend import (
    TransformersBackend,
    clear_model_cache,
    get_model_cache_stats,
)
from loopbench.data.generate import generate_dataset
from loopbench.metrics import paired_cluster_bootstrap_diff_ci
from loopbench.report import MOCK_STAMP, build_phase3_report
from loopbench.runner import run
from loopbench.storage import load_rows


class _Fake1D(list):
    def __eq__(self, other):
        return _Fake1D([1 if x == other else 0 for x in self])

    def sum(self):
        total = sum(self)
        return SimpleNamespace(item=lambda: total)

    def __getitem__(self, item):
        res = super().__getitem__(item)
        return _Fake1D(res) if isinstance(item, slice) else res

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self)


class _Fake2D(list):
    @property
    def shape(self):
        return (len(self), len(self[0]) if self else 0)

    def to(self, _dev):
        return self

    def __getitem__(self, idx):
        row = super().__getitem__(idx)
        return _Fake1D(row) if isinstance(row, list) else row


class _FakeLayer:
    def __init__(self):
        self._pre_hooks = []

    def register_forward_pre_hook(self, fn):
        self._pre_hooks.append(fn)

        def _rm():
            if fn in self._pre_hooks:
                self._pre_hooks.remove(fn)

        return SimpleNamespace(remove=_rm)

    def step(self):
        for fn in list(self._pre_hooks):
            fn(self, ())


class _FakeLangModel:
    def __init__(self):
        self.config = SimpleNamespace(loop_num_passes=3)
        self.layers = [_FakeLayer() for _ in range(24)]
        self._post_hooks = []

    def register_forward_hook(self, fn):
        self._post_hooks.append(fn)

        def _rm():
            if fn in self._post_hooks:
                self._post_hooks.remove(fn)

        return SimpleNamespace(remove=_rm)

    def run_prefill(self):
        passes = int(self.config.loop_num_passes)
        total_evals = 24 + 12 * (passes - 1)
        for i in range(total_evals):
            self.layers[i % len(self.layers)].step()
        for fn in list(self._post_hooks):
            fn(self, (), None)


class _FakeCompositeModel:
    def __init__(self):
        self.device = SimpleNamespace(type="cpu")
        self.model = SimpleNamespace(language_model=_FakeLangModel())
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(loop_num_passes=3),
            image_token_id=200054,
            eos_token_id=200006,
        )

    def generate(self, **kwargs):
        self.model.language_model.run_prefill()
        inp = list(kwargs["input_ids"][0])
        sc_list = kwargs.get("stopping_criteria") or []
        gen = [200008, 11, 12, 200010, 200004, 42, 200010, 200006]
        seq = _Fake2D([inp + gen])
        for sc in sc_list:
            sc(seq, None)
        return seq


class _FakeProcessor:
    image_token_id = 200054
    _MAP = {
        "<|content_thinking|>": 200008,
        "<|content_text|>": 200004,
        "<|end_message|>": 200010,
        "<|content_model_end_sampling|>": 200006,
    }

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self._MAP.get(tok, 999)

    def apply_chat_template(self, messages, **kwargs):
        return {
            "input_ids": _Fake2D([[101, 200054, 200054, 102]]),
            "image_grid_thw": _Fake2D([[1, 20, 20]]),
        }

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "<|content_thinking|>ok<|end_message|><|content_text|>42<|end_message|><|content_model_end_sampling|>"


def test_zero_reload_across_passes_1_2_3_5(tmp_path, monkeypatch):
    class _StoppingCriteria:
        pass

    class _StoppingCriteriaList(list):
        pass

    fake_torch = SimpleNamespace(
        device=lambda t: SimpleNamespace(type=t),
        cuda=SimpleNamespace(is_available=lambda: False),
        manual_seed=lambda _s: None,
        inference_mode=contextlib.nullcontext,
        LongTensor=object,
        FloatTensor=object,
    )
    fake_transformers = SimpleNamespace(
        StoppingCriteria=_StoppingCriteria,
        StoppingCriteriaList=_StoppingCriteriaList,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    clear_model_cache()
    load_counter = {"n": 0}

    def _loader(_cfg):
        load_counter["n"] += 1
        return _FakeCompositeModel(), _FakeProcessor()

    backend = TransformersBackend(loader=_loader)
    img = tmp_path / "img.png"
    from PIL import Image

    Image.new("RGB", (768, 768), (255, 255, 255)).save(img)
    q = {"resolved_image_path": str(img), "prompt": "What is X?"}
    base_cfg = {
        "model_id": "thinkingmachines/TML-d24-varloop-research",
        "model_revision": "c90921bd5f78df6c8168173b9f8a3a850320c594",
        "backend": "transformers",
        "reasoning_effort": "medium",
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 128,
        "stop": [],
        "seed": 20261123,
    }

    expected_layers = {1: 24, 2: 36, 3: 48, 5: 72}
    for idx, passes in enumerate((1, 2, 3, 5)):
        out = backend.generate(q, {**base_cfg, "loop_num_passes": passes})
        assert out["effective_layers_verified"] == expected_layers[passes]
        assert out["visual_tokens"] == 2
        assert out["image_grid_thw"] == [1, 20, 20]
        assert out["model_cache_hit"] is (idx > 0)
        if idx > 0:
            assert out["model_load_s"] == 0.0

    stats = get_model_cache_stats()
    assert load_counter["n"] == 1
    assert stats == {"loads": 1, "hits": 3}
    clear_model_cache()


class _PassScalingMockBackend(Backend):
    def generate(self, question, config):
        passes = int(config["loop_num_passes"])
        mode = config.get("image_mode", "correct")
        layers = 24 + 12 * (passes - 1)
        if mode == "blank":
            return {
                "text": "The provided image is completely blank white.",
                "finish_reason": "stop",
                "ttft_ms": 200.0,
                "tpot_ms": 80.0,
                "e2e_ms": 4000.0,
                "peak_vram_mb": 50700.0,
                "reasoning_tokens": 60,
                "answer_tokens": 9,
                "visual_tokens": 400,
                "effective_layers_verified": layers,
                "is_mock": True,
            }
        if mode == "mismatched":
            return {
                "text": f"Wrong chart/diagram value: {float(question['answer']) + 777:g}",
                "finish_reason": "stop",
                "ttft_ms": 210.0,
                "tpot_ms": 82.0,
                "e2e_ms": 4500.0,
                "peak_vram_mb": 50705.0,
                "reasoning_tokens": 75,
                "answer_tokens": 8,
                "visual_tokens": 400,
                "effective_layers_verified": layers,
                "is_mock": True,
            }
        # Simulate multi_step accuracy improving from loop=1 to loop>=2
        is_correct = True if (passes >= 2 or question["task_type"] == "direct") else False
        ans = float(question["answer"]) if is_correct else float(question["answer"]) + 10.0
        return {
            "text": f"Final answer: {ans:g}",
            "finish_reason": "stop",
            "ttft_ms": 110.0 * (layers / 24.0),
            "tpot_ms": 35.0 * (layers / 24.0),
            "e2e_ms": 5000.0 * (layers / 24.0),
            "peak_vram_mb": 50620.0 + 15.0 * passes,
            "reasoning_tokens": 110 + 5 * passes,
            "answer_tokens": 6,
            "visual_tokens": 400,
            "effective_layers_verified": layers,
            "is_mock": True,
        }


def test_phase3_report_and_paired_cluster_bootstrap(tmp_path):
    ds = tmp_path / "ds"
    generate_dataset(ds, seed=20261123, n_per_type=6)  # 24 questions across 12 images
    base_cfg = {
        "model_id": "mock://phase3",
        "model_revision": "mock-v1",
        "backend": "mock",
        "loop_num_passes": 1,
        "reasoning_effort": "medium",
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": 2048,
        "stop": [],
        "image_size": [768, 768],
        "color_mode": "RGB",
        "render_version": "matplotlib-3.8.4",
        "concurrency": 1,
        "seed": 20261123,
    }
    backend = _PassScalingMockBackend()
    pass_map: dict[int, Path] = {}
    for p in (1, 2, 3, 5):
        jp = tmp_path / f"smoke_loop{p}.jsonl"
        run({**base_cfg, "loop_num_passes": p}, ds, jp, backend=backend)
        pass_map[p] = jp

    leak_map: dict[str, Path] = {}
    for mode in ("blank", "mismatched"):
        lp = tmp_path / f"leak_{mode}.jsonl"
        run({**base_cfg, "loop_num_passes": 3, "image_mode": mode}, ds, lp, backend=backend)
        leak_map[mode] = lp

    r1 = load_rows(pass_map[1])
    r3 = load_rows(pass_map[3])
    assert r1[0]["loop_num_passes"] == 1
    assert r3[0]["loop_num_passes"] == 3
    diff = paired_cluster_bootstrap_diff_ci(r1, r3)
    assert diff["n_questions"] == 24
    assert diff["n_clusters"] == 12
    assert abs(diff["diff_accuracy"] - 0.5) < 1e-9
    assert diff["diff_cluster_bootstrap_95ci"][0] == 0.5

    rep_dir = tmp_path / "report"
    payload = build_phase3_report(
        pass_jsonl_map=pass_map,
        leakage_jsonl_map=leak_map,
        out_dir=rep_dir,
        cache_stats={"loads": 1, "hits": 143},
        leakage_passes=3,
    )
    assert payload["mock"] is True
    assert payload["label"] == MOCK_STAMP
    assert payload["leakage_accuracy_drops"]["correct_minus_blank"] == 1.0
    assert payload["leakage_accuracy_drops"]["correct_minus_mismatched"] == 1.0
    assert (rep_dir / "phase3_summary.json").is_file()
    assert (rep_dir / "phase3_pass_curve.csv").is_file()
    assert (rep_dir / "phase3_leakage_controls.csv").is_file()
    assert (rep_dir / "pareto_pass_scaling_mock.png").is_file()
    assert (rep_dir / "report.md").is_file()

    saved = json.loads((rep_dir / "phase3_summary.json").read_text(encoding="utf-8"))
    assert saved["passes_evaluated"] == [1, 2, 3, 5]
