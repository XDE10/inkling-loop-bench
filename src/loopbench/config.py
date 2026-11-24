from pathlib import Path
import yaml

REQUIRED = (
    "model_id",
    "model_revision",
    "backend",
    "loop_num_passes",
    "reasoning_effort",
    "temperature",
    "top_p",
    "repetition_penalty",
    "max_new_tokens",
    "stop",
    "image_size",
    "color_mode",
    "render_version",
    "concurrency",
    "seed",
)

VALID_PASSES = (1, 2, 3, 5)
VALID_BACKENDS = ("mock", "transformers", "vllm", "sglang")
VALID_EFFORTS = ("none", "low", "medium", "high")
VALID_IMAGE_MODES = ("correct", "blank", "mismatched")
PLACEHOLDER_REVISIONS = {"", "TO_BE_PINNED_BEFORE_GPU", "TBD", "TODO"}


def validate_config_dict(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Config payload must be a mapping")
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise ValueError("Missing explicit config fields: " + ", ".join(missing))
    if data["loop_num_passes"] not in VALID_PASSES:
        raise ValueError("loop_num_passes must be one of 1, 2, 3, 5")
    if data["backend"] not in VALID_BACKENDS:
        raise ValueError(f"Unsupported backend: {data['backend']}")
    if data["reasoning_effort"] not in VALID_EFFORTS:
        raise ValueError(f"Unsupported reasoning_effort: {data['reasoning_effort']}")
    if float(data["temperature"]) < 0.0:
        raise ValueError("temperature must be >= 0")
    if not (0.0 < float(data["top_p"]) <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    if float(data["repetition_penalty"]) <= 0.0:
        raise ValueError("repetition_penalty must be positive")
    if int(data["max_new_tokens"]) <= 0 or int(data["concurrency"]) <= 0:
        raise ValueError("max_new_tokens and concurrency must be positive")
    if list(data["image_size"]) != [768, 768]:
        raise ValueError("Phase 1 requires fixed 768x768 images")
    if data["color_mode"] != "RGB":
        raise ValueError("color_mode must be RGB")
    if not isinstance(data["stop"], list):
        raise ValueError("stop must be an explicit list")
    image_mode = data.get("image_mode", "correct")
    if image_mode not in VALID_IMAGE_MODES:
        raise ValueError(f"image_mode must be one of {VALID_IMAGE_MODES}")
    normalized = dict(data)
    normalized["image_size"] = [768, 768]
    normalized["image_mode"] = image_mode
    return normalized


def validate_runtime_ready(config: dict) -> None:
    if config["backend"] != "mock":
        rev = str(config.get("model_revision") or "").strip()
        if rev in PLACEHOLDER_REVISIONS:
            raise ValueError(
                "Non-mock execution requires pinning an exact model_revision commit SHA"
            )


def load_config(path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return validate_config_dict(raw)
