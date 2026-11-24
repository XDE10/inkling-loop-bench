from .base import Backend
from .mock import MockBackend
from .openai_compat import OpenAICompatBackend
from .transformers_backend import TransformersBackend
from ..config import validate_runtime_ready


def create_backend(config: dict) -> Backend:
    validate_runtime_ready(config)
    backend_name = config["backend"]
    if backend_name == "mock":
        return MockBackend()
    if backend_name == "transformers":
        return TransformersBackend()
    if backend_name in ("vllm", "sglang"):
        base_url = config.get("api_base_url", "http://127.0.0.1:8000/v1")
        return OpenAICompatBackend(name=backend_name, base_url=base_url)
    raise ValueError(f"Unsupported backend: {backend_name}")


__all__ = [
    "Backend",
    "MockBackend",
    "OpenAICompatBackend",
    "TransformersBackend",
    "create_backend",
]
