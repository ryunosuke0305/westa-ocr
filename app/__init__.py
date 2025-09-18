"""Core application package for the relay service."""

from typing import Any

__all__ = ["app"]

_lazy_app: Any | None = None


def __getattr__(name: str) -> Any:
    if name != "app":
        raise AttributeError(name)

    global _lazy_app
    if _lazy_app is None:
        from .main import app as fastapi_app  # pragma: no cover - exercised indirectly

        _lazy_app = fastapi_app
    return _lazy_app
