from pathlib import Path
import re
import time
from typing import Any
from .base import Backend

THINK_START_TOKEN = "<|content_thinking|>"
TEXT_START_TOKEN = "<|content_text|>"
END_MSG_TOKEN = "<|end_message|>"
EOS_SAMPLING_TOKEN = "<|content_model_end_sampling|>"

_THINK_RE = re.compile(r"<\|content_thinking\|>(.*?)(?:<\|end_message\|>|$)", re.S)
_TEXT_RE = re.compile(r"<\|content_text\|>(.*?)(?:<\|end_message\|>|<\|content_model_end_sampling\|>|$)", re.S)

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, float]] = {}
_CACHE_STATS: dict[str, int] = {"loads": 0, "hits": 0}


def clear_model_cache() -> None:
    _MODEL_CACHE.clear()
    _CACHE_STATS["loads"] = 0
    _CACHE_STATS["hits"] = 0


def get_model_cache_stats() -> dict[str, int]:
    return dict(_CACHE_STATS)


def parse_inkling_output(gen_ids: list[int], tokenizer: Any, raw_text: str) -> dict[str, Any]:
    tok_think = tokenizer.convert_tokens_to_ids(THINK_START_TOKEN) if tokenizer is not None else 200008
    tok_text = tokenizer.convert_tokens_to_ids(TEXT_START_TOKEN) if tokenizer is not None else 200004
    tok_end = tokenizer.convert_tokens_to_ids(END_MSG_TOKEN) if tokenizer is not None else 200010
    tok_eos = tokenizer.convert_tokens_to_ids(EOS_SAMPLING_TOKEN) if tokenizer is not None else 200006

    reasoning_tokens = 0
    answer_tokens = 0
    mode = None
    for tid in gen_ids:
        if tid == tok_think:
            mode = "thinking"
            continue
        if tid == tok_text:
            mode = "answer"
            continue
        if tid in (tok_end, tok_eos):
            mode = None
            continue
        if mode == "thinking":
            reasoning_tokens += 1
        elif mode == "answer":
            answer_tokens += 1

    think_matches = _THINK_RE.findall(raw_text)
    text_matches = _TEXT_RE.findall(raw_text)
    reasoning_text = "\n".join(m.strip() for m in think_matches if m.strip())
    answer_text = "\n".join(m.strip() for m in text_matches if m.strip())

    # If model answered directly without special tags (e.g., unit-test stub), fall back cleanly.
    effective_text = answer_text if text_matches else raw_text
    if not text_matches and not think_matches and answer_tokens == 0:
        answer_tokens = len([t for t in gen_ids if t != tok_eos])

    return {
        "text": effective_text,
        "raw_text": raw_text,
        "reasoning_text": reasoning_text,
        "reasoning_tokens": reasoning_tokens,
        "answer_tokens": answer_tokens,
    }


def apply_loop_num_passes(model: Any, loop_num_passes: int) -> int:
    passes = int(loop_num_passes)
    if passes < 1:
        raise ValueError(f"loop_num_passes must be >= 1, got {passes}")

    # Check if model is fixed-loop (e.g., official Inkling-2-Mini with is_varloop=False)
    lang_model = getattr(getattr(model, "model", None), "language_model", None)
    lm_cfg = getattr(lang_model, "config", None) if lang_model is not None else None
    text_cfg = getattr(getattr(model, "config", None), "text_config", None)
    for cfg_obj in (lm_cfg, text_cfg, getattr(model, "config", None)):
        loop_cfg = getattr(cfg_obj, "loop_config", None)
        if isinstance(loop_cfg, dict) and loop_cfg.get("is_varloop") is False:
            fixed_passes = int(loop_cfg.get("num_passes", 3))
            if passes != fixed_passes:
                raise ValueError(
                    f"Checkpoint has is_varloop=False (fixed num_passes={fixed_passes}); "
                    f"cannot switch loop_num_passes to {passes}"
                )

    # Update both outer composite config and inner language_model config so __post_init__
    # defaults from config.json ('loop_config': {'num_passes': 3}) never shadow runtime overrides.
    if hasattr(model, "config"):
        if hasattr(model.config, "text_config") and model.config.text_config is not None:
            model.config.text_config.loop_num_passes = passes
        elif hasattr(model.config, "loop_num_passes"):
            model.config.loop_num_passes = passes

    if lang_model is not None and hasattr(lang_model, "config"):
        lang_model.config.loop_num_passes = passes
        return int(getattr(lang_model.config, "total_num_hidden_layers", 24 + 12 * (passes - 1)))

    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "config") and hasattr(inner.config, "loop_num_passes"):
        inner.config.loop_num_passes = passes
        return int(getattr(inner.config, "total_num_hidden_layers", 24 + 12 * (passes - 1)))

    return 24 + 12 * (passes - 1)


class TransformersBackend(Backend):
    name = "transformers"

    def __init__(self, loader=None):
        self._loader = loader

    def _get_model_and_processor(self, config: dict) -> tuple[Any, Any, float, bool]:
        key = (config["model_id"], str(config["model_revision"]))
        if key in _MODEL_CACHE:
            _CACHE_STATS["hits"] += 1
            model, processor, load_s = _MODEL_CACHE[key]
            return model, processor, load_s, True

        if self._loader is not None:
            t0 = time.perf_counter()
            model, processor = self._loader(config)
            load_s = time.perf_counter() - t0
            _MODEL_CACHE[key] = (model, processor, load_s)
            _CACHE_STATS["loads"] += 1
            return model, processor, load_s, False

        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        t0 = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            config["model_id"],
            revision=config["model_revision"],
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            config["model_id"],
            revision=config["model_revision"],
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()
        load_s = time.perf_counter() - t0
        _MODEL_CACHE[key] = (model, processor, load_s)
        _CACHE_STATS["loads"] += 1
        return model, processor, load_s, False

    def generate(self, question: dict, config: dict) -> dict:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        image_path = question.get("resolved_image_path") or question.get("image_path")
        if not image_path or not Path(image_path).is_file():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        model, processor, load_s, cache_hit = self._get_model_and_processor(config)
        passes = int(config["loop_num_passes"])
        expected_effective_layers = apply_loop_num_passes(model, passes)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": question["prompt"]},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            reasoning_effort=config["reasoning_effort"],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = getattr(model, "device", torch.device("cpu"))
        inputs = {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }

        input_ids = inputs["input_ids"]
        input_len = int(input_ids.shape[-1])
        tokenizer = getattr(processor, "tokenizer", processor)
        image_token_id = getattr(
            processor,
            "image_token_id",
            getattr(getattr(model, "config", None), "image_token_id", 200054),
        )
        visual_tokens = int((input_ids[0] == image_token_id).sum().item())
        grid_thw = (
            inputs["image_grid_thw"][0].detach().cpu().tolist()
            if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None
            else None
        )

        # Count decoder layer forward invocations on prefill (first forward pass) to verify loop depth causally.
        layer_call_counts = {"step_calls": 0, "prefill_layers": None, "total_layers": 0}
        hooks = []
        lang_model = getattr(getattr(model, "model", None), "language_model", None)
        if lang_model is not None and hasattr(lang_model, "layers"):
            def _layer_hook(_mod, _inp):
                layer_call_counts["step_calls"] += 1
                layer_call_counts["total_layers"] += 1

            def _lm_post_hook(_mod, _inp, _out):
                if layer_call_counts["prefill_layers"] is None:
                    layer_call_counts["prefill_layers"] = layer_call_counts["step_calls"]
                layer_call_counts["step_calls"] = 0

            for lyr in lang_model.layers:
                hooks.append(lyr.register_forward_pre_hook(_layer_hook))
            hooks.append(lang_model.register_forward_hook(_lm_post_hook))

        stop_strings: list[str] = list(config.get("stop") or [])
        use_cuda = torch.cuda.is_available() and getattr(device, "type", "cpu") == "cuda"

        class _TimingAndStopCriteria(StoppingCriteria):
            def __init__(self):
                self.first_token_perf: float | None = None
                self.stopped_by_string = False

            def __call__(self, cur_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                if self.first_token_perf is None:
                    if use_cuda:
                        torch.cuda.synchronize(device)
                    self.first_token_perf = time.perf_counter()
                if stop_strings:
                    tail_txt = tokenizer.decode(cur_ids[0][input_len:], skip_special_tokens=True)
                    for s in stop_strings:
                        if s and s in tail_txt:
                            self.stopped_by_string = True
                            return True
                return False

        timer_criteria = _TimingAndStopCriteria()
        temp = float(config["temperature"])
        max_new_tokens = int(config["max_new_tokens"])

        if int(config.get("seed", 0)):
            torch.manual_seed(int(config["seed"]))
            if use_cuda:
                torch.cuda.manual_seed_all(int(config["seed"]))

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": float(config["repetition_penalty"]),
            "stopping_criteria": StoppingCriteriaList([timer_criteria]),
            "use_cache": True,
        }
        if temp > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = float(config["top_p"])
        else:
            gen_kwargs["do_sample"] = False

        if use_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        start_perf = time.perf_counter()
        try:
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **gen_kwargs)
            if use_cuda:
                torch.cuda.synchronize(device)
            end_perf = time.perf_counter()
        finally:
            for h in hooks:
                h.remove()

        if (
            layer_call_counts["prefill_layers"] is not None
            and layer_call_counts["prefill_layers"] != expected_effective_layers
        ):
            raise RuntimeError(
                f"Loop pass verification failed: expected {expected_effective_layers} layer calls "
                f"for loop_num_passes={passes}, observed {layer_call_counts['prefill_layers']}"
            )

        gen_seq = output_ids[0][input_len:].detach().cpu().tolist()
        total_gen_tokens = len(gen_seq)
        eos_id = getattr(getattr(model, "config", None), "eos_token_id", 200006)

        if timer_criteria.stopped_by_string:
            finish_reason = "stop"
        elif total_gen_tokens >= max_new_tokens and (not gen_seq or gen_seq[-1] != eos_id):
            finish_reason = "length"
        else:
            finish_reason = "stop"

        raw_text = processor.decode(gen_seq, skip_special_tokens=False)
        parsed = parse_inkling_output(gen_seq, tokenizer, raw_text)
        for s in stop_strings:
            if s and s in parsed["text"]:
                parsed["text"] = parsed["text"].split(s)[0]

        e2e_ms = (end_perf - start_perf) * 1000.0
        ttft_ms = (
            (timer_criteria.first_token_perf - start_perf) * 1000.0
            if timer_criteria.first_token_perf is not None
            else e2e_ms
        )
        tpot_ms = ((e2e_ms - ttft_ms) / (total_gen_tokens - 1)) if total_gen_tokens > 1 else 0.0
        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            if use_cuda
            else None
        )

        return {
            "text": parsed["text"],
            "raw_text": parsed["raw_text"],
            "reasoning_text": parsed["reasoning_text"],
            "finish_reason": finish_reason,
            "reasoning_tokens": parsed["reasoning_tokens"],
            "answer_tokens": parsed["answer_tokens"],
            "total_generated_tokens": total_gen_tokens,
            "input_tokens": input_len,
            "visual_tokens": visual_tokens,
            "image_grid_thw": grid_thw,
            "effective_layers_verified": layer_call_counts["prefill_layers"],
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_ms": e2e_ms,
            "peak_vram_mb": peak_vram_mb,
            "model_load_s": 0.0 if cache_hit else load_s,
            "initial_model_load_s": load_s,
            "model_cache_hit": cache_hit,
            "speculative_decoding": False,
            "loop_share_cache": False,
            "is_mock": False,
        }

    def generate_batch(self, questions: list[dict], config: dict) -> list[dict]:
        """Throughput-oriented left-padded batched generation for concurrency > 1.

        For `len(questions) == 1`, delegates directly to `generate()` to preserve exact
        single-stream TTFT/TPOT semantics. For `len(questions) > 1`, records amortized
        `e2e_ms = batch_e2e_ms / B` and marks `batched = True` (leaving per-request
        `ttft_ms` / `tpot_ms` explicitly `None` to prevent mixing with `bs=1` telemetry).
        """
        if not questions:
            return []
        if len(questions) == 1:
            return [self.generate(questions[0], config)]

        import torch

        for q in questions:
            img_p = q.get("resolved_image_path") or q.get("image_path")
            if not img_p or not Path(img_p).is_file():
                raise FileNotFoundError(f"Missing image file: {img_p}")

        model, processor, load_s, cache_hit = self._get_model_and_processor(config)
        passes = int(config["loop_num_passes"])
        expected_effective_layers = apply_loop_num_passes(model, passes)

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(q.get("resolved_image_path") or q.get("image_path"))},
                        {"type": "text", "text": q["prompt"]},
                    ],
                }
            ]
            for q in questions
        ]

        tokenizer = getattr(processor, "tokenizer", processor)
        orig_padding_side = getattr(tokenizer, "padding_side", None)
        eos_id = getattr(getattr(model, "config", None), "eos_token_id", 200006)
        if isinstance(eos_id, (list, tuple)):
            eos_id = int(eos_id[0])
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_id
            if hasattr(tokenizer, "pad_token_id"):
                tokenizer.pad_token_id = pad_id

        try:
            if hasattr(tokenizer, "padding_side"):
                tokenizer.padding_side = "left"
            inputs = processor.apply_chat_template(
                conversations,
                reasoning_effort=config["reasoning_effort"],
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            )
        finally:
            if orig_padding_side is not None and hasattr(tokenizer, "padding_side"):
                tokenizer.padding_side = orig_padding_side

        device = getattr(model, "device", torch.device("cpu"))
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        attn_mask = inputs.get("attention_mask")
        padded_input_len = int(input_ids.shape[-1])
        batch_size = len(questions)

        image_token_id = getattr(
            processor,
            "image_token_id",
            getattr(getattr(model, "config", None), "image_token_id", 200054),
        )
        grid_thw_all = (
            inputs["image_grid_thw"].detach().cpu().tolist()
            if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None
            else None
        )

        layer_call_counts = {"step_calls": 0, "prefill_layers": None}
        hooks = []
        lang_model = getattr(getattr(model, "model", None), "language_model", None)
        if lang_model is not None and hasattr(lang_model, "layers"):
            def _layer_hook(_mod, _inp):
                layer_call_counts["step_calls"] += 1

            def _lm_post_hook(_mod, _inp, _out):
                if layer_call_counts["prefill_layers"] is None:
                    layer_call_counts["prefill_layers"] = layer_call_counts["step_calls"]
                layer_call_counts["step_calls"] = 0

            for lyr in lang_model.layers:
                hooks.append(lyr.register_forward_pre_hook(_layer_hook))
            hooks.append(lang_model.register_forward_hook(_lm_post_hook))

        use_cuda = torch.cuda.is_available() and getattr(device, "type", "cpu") == "cuda"
        temp = float(config["temperature"])
        max_new_tokens = int(config["max_new_tokens"])
        stop_strings: list[str] = list(config.get("stop") or [])

        if int(config.get("seed", 0)):
            torch.manual_seed(int(config["seed"]))
            if use_cuda:
                torch.cuda.manual_seed_all(int(config["seed"]))

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": float(config["repetition_penalty"]),
            "use_cache": True,
            "pad_token_id": pad_id,
        }
        if temp > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temp
            gen_kwargs["top_p"] = float(config["top_p"])
        else:
            gen_kwargs["do_sample"] = False

        if use_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **gen_kwargs)
            if use_cuda:
                torch.cuda.synchronize(device)
            batch_e2e_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            for h in hooks:
                h.remove()

        if (
            layer_call_counts["prefill_layers"] is not None
            and layer_call_counts["prefill_layers"] != expected_effective_layers
        ):
            raise RuntimeError(
                f"Batched loop pass verification failed: expected {expected_effective_layers} "
                f"for loop_num_passes={passes}, observed {layer_call_counts['prefill_layers']}"
            )

        peak_vram_mb = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            if use_cuda
            else None
        )

        results: list[dict] = []
        for idx in range(batch_size):
            raw_seq = output_ids[idx][padded_input_len:].detach().cpu().tolist()
            # Trim at first EOS or trailing pad tokens produced while waiting for longer batch siblings
            trimmed: list[int] = []
            hit_eos = False
            for tid in raw_seq:
                if tid == eos_id:
                    trimmed.append(tid)
                    hit_eos = True
                    break
                if tid == pad_id and pad_id != eos_id:
                    break
                trimmed.append(tid)

            total_gen_tokens = len(trimmed)
            finish_reason = "length" if (total_gen_tokens >= max_new_tokens and not hit_eos) else "stop"
            raw_text = processor.decode(trimmed, skip_special_tokens=False)
            parsed = parse_inkling_output(trimmed, tokenizer, raw_text)
            for s in stop_strings:
                if s and s in parsed["text"]:
                    parsed["text"] = parsed["text"].split(s)[0]

            in_len = (
                int(attn_mask[idx].sum().item())
                if attn_mask is not None
                else padded_input_len
            )
            vis_toks = int((input_ids[idx] == image_token_id).sum().item())
            grid_thw = grid_thw_all[idx] if (grid_thw_all and idx < len(grid_thw_all)) else None

            results.append({
                "text": parsed["text"],
                "raw_text": parsed["raw_text"],
                "reasoning_text": parsed["reasoning_text"],
                "finish_reason": finish_reason,
                "reasoning_tokens": parsed["reasoning_tokens"],
                "answer_tokens": parsed["answer_tokens"],
                "total_generated_tokens": total_gen_tokens,
                "input_tokens": in_len,
                "visual_tokens": vis_toks,
                "image_grid_thw": grid_thw,
                "effective_layers_verified": layer_call_counts["prefill_layers"],
                "ttft_ms": None,
                "tpot_ms": None,
                "e2e_ms": batch_e2e_ms / batch_size,
                "batch_e2e_ms": batch_e2e_ms,
                "batch_size": batch_size,
                "batched": True,
                "peak_vram_mb": peak_vram_mb,
                "model_load_s": 0.0 if (cache_hit or idx > 0) else load_s,
                "initial_model_load_s": load_s,
                "model_cache_hit": cache_hit or idx > 0,
                "speculative_decoding": False,
                "loop_share_cache": False,
                "is_mock": False,
            })
        return results
