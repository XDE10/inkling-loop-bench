import base64
import json
from pathlib import Path
import time
import urllib.request
from .base import Backend


class OpenAICompatBackend(Backend):
    """Streaming OpenAI-compatible multimodal client for vLLM / SGLang servers."""

    def __init__(self, name: str = "vllm", base_url: str = "http://127.0.0.1:8000/v1"):
        self.name = name
        self.base_url = base_url.rstrip("/")

    def generate(self, question: dict, config: dict) -> dict:
        image_path = Path(question.get("resolved_image_path") or question["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": config["model_id"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": question["prompt"]},
                    ],
                }
            ],
            "temperature": float(config["temperature"]),
            "top_p": float(config["top_p"]),
            "max_tokens": int(config["max_new_tokens"]),
            "stop": config["stop"] or None,
            "reasoning_effort": config["reasoning_effort"],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            method="POST",
        )

        start = time.perf_counter()
        first_tok: float | None = None
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason = "stop"
        usage = {}

        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                r_piece = delta.get("reasoning_content") or ""
                c_piece = delta.get("content") or ""
                if (r_piece or c_piece) and first_tok is None:
                    first_tok = time.perf_counter()
                if r_piece:
                    reasoning_parts.append(r_piece)
                if c_piece:
                    content_parts.append(c_piece)
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]

        end = time.perf_counter()
        e2e_ms = (end - start) * 1000.0
        ttft_ms = ((first_tok - start) * 1000.0) if first_tok is not None else e2e_ms
        comp_tokens = int(usage.get("completion_tokens") or 0)
        tpot_ms = ((e2e_ms - ttft_ms) / (comp_tokens - 1)) if comp_tokens > 1 else 0.0

        return {
            "text": "".join(content_parts),
            "reasoning_text": "".join(reasoning_parts),
            "finish_reason": finish_reason,
            "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
            "answer_tokens": comp_tokens or None,
            "visual_tokens": None,
            "image_grid_thw": None,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "e2e_ms": e2e_ms,
            "peak_vram_mb": None,
            "speculative_decoding": bool(config.get("speculative_decoding", False)),
            "loop_share_cache": config.get("loop_share_cache"),
            "is_mock": False,
        }
