from .base import Backend
from .mock import MockBackend
from ..config import validate_runtime_ready


def create_backend(config: dict) -> Backend:
    validate_runtime_ready(config)
    backend_name = config["backend"]
    if backend_name == "mock":
        return MockBackend()
    if backend_name in ("transformers", "vllm", "sglang"):
        raise NotImplementedError(
            f"Backend '{backend_name}' is reserved for Phase 2+ GPU execution; "
            "Phase 1 CPU baseline supports 'mock' only."
        )
    raise ValueError(f"Unsupported backend: {backend_name}")


__all__ = ["Backend", "MockBackend", "create_backend"]
