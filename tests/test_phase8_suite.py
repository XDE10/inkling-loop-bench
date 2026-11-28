from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from loopbench.backends.transformers_backend import apply_loop_num_passes
from loopbench.config import load_config
from loopbench.probes import compute_pass_trajectory_metrics, summarize_token_trajectories
from loopbench.speculative import speculative_loop_generate, verify_exact_equivalence
from run_phase8_mini_eval import render_mini_comparison_figure
from run_phase8_probes import render_probes_figure
from run_phase8_speculative import render_speculative_figure


def test_fixed_varloop_false_guard_for_official_mini():
    cfg_mv = load_config(ROOT / "configs/phase8_mini_mathvista.yaml")
    assert cfg_mv["model_id"] == "thinkingmachines/Inkling-2-Mini"
    assert cfg_mv["loop_num_passes"] == 3

    cfg_cqa = load_config(ROOT / "configs/phase8_mini_chartqa.yaml")
    assert cfg_cqa["model_id"] == "thinkingmachines/Inkling-2-Mini"
    assert cfg_cqa["loop_num_passes"] == 3

    fixed_model = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(loop_config={"is_varloop": False, "num_passes": 3}, loop_num_passes=3)),
        model=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace(loop_config={"is_varloop": False, "num_passes": 3}, loop_num_passes=3))),
    )
    assert apply_loop_num_passes(fixed_model, 3) == 48
    with pytest.raises(ValueError, match="is_varloop=False"):
        apply_loop_num_passes(fixed_model, 1)


def test_verify_equivalence_and_summarize_trajectories_and_figures(tmp_path: Path):
    assert verify_exact_equivalence([1, 2, 3], [1, 2, 3])["exact_match"] is True
    mm = verify_exact_equivalence([1, 2, 9], [1, 2, 3])
    assert mm["exact_match"] is False and mm["first_mismatch_idx"] == 2

    step_metrics = [
        {
            "passes": 5,
            "cos_sim_prev": [None, 0.92, 0.97, 0.992, 0.997],
            "rel_l2_prev": [None, 0.35, 0.18, 0.09, 0.04],
            "adjacent_delta_ratio": [None, None, 0.52, 0.50, 0.45],
            "top1_ids": [5, 7, 7, 7, 7],
            "top1_probs": [0.55, 0.80, 0.91, 0.96, 0.98],
            "entropies": [0.95, 0.42, 0.18, 0.09, 0.04],
            "kl_to_final": [0.60, 0.15, 0.05, 0.01, 0.0],
            "kl_to_loop3": [0.45, 0.10, 0.0, 0.02, 0.04],
            "agree_final": [False, True, True, True, True],
            "agree_loop3": [False, True, True, True, True],
            "first_stable_pass": 2,
        }
        for _ in range(8)
    ]
    agg = summarize_token_trajectories(step_metrics, passes=5)
    assert len(agg["per_pass"]) == 5
    assert agg["per_pass"][-1]["agree_final_rate"] == 1.0
    assert agg["per_pass"][2]["mean_adjacent_delta_ratio"] == pytest.approx(0.52)
    assert agg["per_pass"][2]["delta_ratio_lt1_rate"] == 1.0
    assert agg["mean_first_stable_pass"] == 2.0
    assert len(agg["early_exit_sweep"]) >= 5
    assert len(agg["early_exit_vs_loop3"]) >= 5

    from loopbench.metrics import exact_mcnemar_pvalue
    assert exact_mcnemar_pvalue(0, 4) == pytest.approx(0.125)
    assert exact_mcnemar_pvalue(3, 5) == pytest.approx(0.7265625)
    assert exact_mcnemar_pvalue(7, 11) == pytest.approx(0.480682373046875)

    p_mini = tmp_path / "mini.png"
    render_mini_comparison_figure({
        "mathvista_mini": {"low_loop3": {"accuracy": 0.87, "reasoning_tokens_mean": 340, "truncated": 1}},
        "mathvista_varloop_ref": {"low_loop3": {"accuracy": 0.855, "reasoning_tokens_mean": 364, "truncated": 1}},
        "chartqa_mini": {"relaxed_accuracy": 0.91, "reasoning_tokens_mean": 210, "truncated": 0},
        "chartqa_varloop_ref": {"relaxed_accuracy": 0.888, "reasoning_tokens_mean": 247, "truncated": 0},
    }, p_mini)
    assert p_mini.is_file() and p_mini.stat().st_size > 10_000

    p_spec = tmp_path / "spec.png"
    render_speculative_figure({
        "sweep_summary": [
            {"draft_k": 2, "mean_speedup_vs_loop3": 1.25, "mean_tokens_per_verify_round": 2.4, "mean_acceptance_rate": 0.72, "thinking_acceptance_rate": 0.70, "answer_acceptance_rate": 0.85},
            {"draft_k": 4, "mean_speedup_vs_loop3": 1.52, "mean_tokens_per_verify_round": 3.5, "mean_acceptance_rate": 0.64, "thinking_acceptance_rate": 0.62, "answer_acceptance_rate": 0.80},
        ]
    }, p_spec)
    assert p_spec.is_file() and p_spec.stat().st_size > 10_000

    p_probe = tmp_path / "probe.png"
    render_probes_figure({"overall_summary": agg}, p_probe)
    assert p_probe.is_file() and p_probe.stat().st_size > 10_000


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("draft_k", [2, 4, 6])
def test_self_speculative_loop_decoding_bit_exact_torch(hybrid: bool, draft_k: int):
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class _ToyCache:
        def __init__(self):
            self.key_cache = []
            self.value_cache = []

        def get_seq_length(self) -> int:
            return int(self.key_cache[0].shape[-2]) if self.key_cache else 0

        def crop(self, max_length: int) -> None:
            self.key_cache = [k[..., :max_length, :] for k in self.key_cache]
            self.value_cache = [v[..., :max_length, :] for v in self.value_cache]

    class _ToyHybridCache(_ToyCache):
        def __init__(self, d_model: int):
            super().__init__()
            self.conv_states = [torch.zeros(1, d_model)]

    class _ToyLoopedLM(nn.Module):
        def __init__(self, vocab_size: int = 24, d_model: int = 12, eos_id: int = 23):
            super().__init__()
            torch.manual_seed(1234)
            self.d_model = d_model
            self.device = torch.device("cpu")
            self.config = SimpleNamespace(
                eos_token_id=eos_id,
                text_config=SimpleNamespace(loop_num_passes=3, loop_config={"is_varloop": True, "num_passes": 3}),
            )
            self.emb = nn.Embedding(vocab_size, d_model)
            self.core = nn.Linear(d_model, d_model, bias=False)
            nn.init.eye_(self.core.weight)
            with torch.no_grad():
                self.core.weight += 0.08 * torch.randn_like(self.core.weight)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.model = SimpleNamespace(
                language_model=SimpleNamespace(
                    config=SimpleNamespace(loop_num_passes=3, total_num_hidden_layers=48),
                    norm=nn.LayerNorm(d_model),
                )
            )

        def forward(self, input_ids, past_key_values=None, attention_mask=None, use_cache=True, cache_position=None):
            passes = int(self.model.language_model.config.loop_num_passes)
            x = self.emb(input_ids)
            if past_key_values is None:
                cache = _ToyHybridCache(self.d_model) if hybrid else _ToyCache()
                k_full = x.unsqueeze(1)
                cache.key_cache = [k_full]
                cache.value_cache = [k_full]
            else:
                cache = past_key_values
                k_full = torch.cat([cache.key_cache[0], x.unsqueeze(1)], dim=-2)
                cache.key_cache = [k_full]
                cache.value_cache = [k_full]

            T = input_ids.shape[1]
            start_pos = k_full.shape[-2] - T
            h_steps = []
            conv_s = cache.conv_states[0] if hybrid else None
            for t in range(T):
                pos = start_pos + t
                ctx = k_full[0, 0, : pos + 1, :].mean(dim=0, keepdim=True)
                cur = x[0, t : t + 1, :] + 0.5 * ctx
                if hybrid:
                    conv_s = 0.6 * conv_s + 0.4 * cur
                    cur = cur + 0.25 * conv_s
                h = cur
                for _ in range(passes):
                    h = torch.tanh(self.core(h))
                h_steps.append(h)
            if hybrid:
                cache.conv_states = [conv_s]

            h_seq = torch.stack(h_steps, dim=1)
            logits = self.lm_head(self.model.language_model.norm(h_seq))
            return SimpleNamespace(logits=logits, past_key_values=cache)

        def naive_greedy(self, input_ids, passes: int, max_new_tokens: int) -> list[int]:
            apply_loop_num_passes(self, passes)
            out = self.forward(input_ids, past_key_values=None, use_cache=True)
            pkv = out.past_key_values
            tok = int(out.logits[:, -1, :].argmax(dim=-1).item())
            seq = [tok]
            for _ in range(max_new_tokens - 1):
                if tok == self.config.eos_token_id:
                    break
                out = self.forward(torch.tensor([[tok]], dtype=torch.long), past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                tok = int(out.logits[:, -1, :].argmax(dim=-1).item())
                seq.append(tok)
            return seq

    model = _ToyLoopedLM()
    prompt = torch.tensor([[1, 5, 3, 9, 2]], dtype=torch.long)
    baseline = model.naive_greedy(prompt, passes=3, max_new_tokens=20)
    spec = speculative_loop_generate(
        model,
        {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)},
        draft_passes=1,
        target_passes=3,
        draft_k=draft_k,
        max_new_tokens=20,
        eos_token_id=23,
    )
    eq = verify_exact_equivalence(spec["generated_ids"], baseline)
    assert eq["exact_match"] is True, f"Mismatch for hybrid={hybrid}, K={draft_k}: {eq}"
    assert spec["hybrid_cache_detected"] is hybrid
    assert spec["verify_rounds"] > 0


def test_compute_pass_trajectory_metrics_torch():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    torch.manual_seed(7)
    lang_model = SimpleNamespace(norm=nn.LayerNorm(16))
    lm_head = nn.Linear(16, 20, bias=False)
    base_h = torch.randn(16)
    hs = [base_h + (0.4 / (p + 1)) * torch.randn(16) for p in range(5)]
    m = compute_pass_trajectory_metrics(hs, lang_model, lm_head)
    assert m["passes"] == 5
    assert m["agree_final"][-1] is True
    assert m["kl_to_final"][-1] == 0.0


def test_snapshot_cache_clones_nested_hybrid_and_sliding_states():
    from loopbench.speculative import _restore_recurrent_snapshot, _snapshot_cache

    class _FakeSlidingHybridLayer:
        def __init__(self):
            self.sliding_window = 1024
            self.keys = [1, 2, 3]
            self.values = [4, 5, 6]
            self.conv_states = [[10, 20]]
            self.recurrent_states = {"s": [30, 40]}
            self.cumulative_length = 3

    layer = _FakeSlidingHybridLayer()
    cache = SimpleNamespace(layers=[layer])
    snap = _snapshot_cache(cache)

    layer.keys.append(99)
    layer.conv_states[0].append(999)
    layer.recurrent_states["s"].append(888)
    layer.cumulative_length = 4

    _restore_recurrent_snapshot(cache, snap)
    assert layer.keys == [1, 2, 3]
    assert layer.conv_states == [[10, 20]]
    assert layer.recurrent_states == {"s": [30, 40]}
    assert layer.cumulative_length == 3
