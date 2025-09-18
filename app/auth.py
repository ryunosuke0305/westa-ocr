"""Request authentication utilities."""

from __future__ import annotations

import hmac
from fastapi import Depends, HTTPException, Request, status

from .logging_config import get_logger
from .settings import Settings, get_settings

LOGGER = get_logger(__name__)


async def verify_request(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """Validate incoming requests using Bearer tokens."""

    auth_header = request.headers.get("Authorization")
    if auth_header:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer" and hmac.compare_digest(parts[1], settings.relay_token):
            return

    LOGGER.warning("Authentication failed")
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


__all__ = ["verify_request"]
