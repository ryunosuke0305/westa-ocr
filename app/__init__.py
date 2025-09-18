"""Core application package for the relay service."""

from .main import app  # re-export for convenience when running via `uvicorn app:app`

__all__ = ["app"]
