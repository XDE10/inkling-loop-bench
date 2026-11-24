from pathlib import Path
import time
from .base import Backend


class MockBackend(Backend):
    name = "mock"

    def generate(self, question, config):
        start = time.perf_counter()
        image_path = question.get("resolved_image_path") or question.get("image_path")
        if not image_path or not Path(image_path).is_file():
            raise FileNotFoundError(f"Missing image file for mock inference: {image_path}")

        image_mode = config.get("image_mode", "correct")
        if image_mode == "blank":
            text = "TEST DATA ONLY (BLANK IMAGE). Visual input is blank; answer unavailable."
        elif image_mode == "mismatched":
            wrong_val = float(question["answer"]) + 1000.0
            text = f"TEST DATA ONLY (MISMATCHED IMAGE). Extracted distractor value {wrong_val:g}."
        else:
            # Deterministic test response; deliberately a pipeline fixture rather than a model prediction.
            answer = question["answer"]
            text = f"TEST DATA ONLY. The computed answer is {answer:g}."

        elapsed = (time.perf_counter() - start) * 1000.0
        return {
            "text": text,
            "finish_reason": "stop",
            "reasoning_tokens": 0,
            "answer_tokens": 8,
            "ttft_ms": elapsed,
            "tpot_ms": 0.0,
            "e2e_ms": elapsed,
            "peak_vram_mb": None,
            "is_mock": True,
        }
