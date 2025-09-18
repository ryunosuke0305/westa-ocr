"""Request authentication utilities."""

from __future__ import annotations

import hmac
import time
from hashlib import sha256
from fastapi import Depends, HTTPException, Request, status

from .logging_config import get_logger
from .settings import Settings, get_settings

LOGGER = get_logger(__name__)


async def verify_request(request: Request, settings: Settings = Depends(get_settings)) -> None:
    """Validate incoming requests using Bearer tokens or HMAC signatures."""

    raw_body = await request.body()
    request._body = raw_body  # type: ignore[attr-defined]
    request.state.raw_body = raw_body

    if not settings.requires_authentication:
        return

    auth_header = request.headers.get("Authorization")
    if auth_header and settings.relay_token:
        parts = auth_header.split()
        if len(parts) == 2 and parts[0].lower() == "bearer" and hmac.compare_digest(parts[1], settings.relay_token):
            return

    signature = request.headers.get("X-Signature")
    timestamp = request.headers.get("X-Timestamp")
    if signature and timestamp and settings.hmac_secret:
        try:
            ts_int = int(timestamp)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid timestamp") from exc
        now = int(time.time())
        if abs(now - ts_int) > settings.hmac_ttl_seconds:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Signature expired")
        expected = hmac.new(settings.hmac_secret.encode("utf-8"), raw_body, sha256).hexdigest()
        if hmac.compare_digest(expected, signature):
            return

    LOGGER.warning("Authentication failed")
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


__all__ = ["verify_request"]
