"""v0.2 security primitives — verify-only.

The coordinator component is never the trust root. The watchdog generates
the per-restart HMAC signing key and distributes it via env var
(`EUGENE_PLEXUS_CRD_AUTH_SIGNING_KEY`). This module exposes just the
decode side so route dependencies can validate inbound bearer tokens.

Identical shape to the other platform components' security
modules — keeping these nearly-identical makes the cross-cutting
contract easy to audit at a glance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jwt

_JWT_ALG = "HS256"

AUDIENCE_OPERATOR = "operator"
SERVICE_AUDIENCE_PREFIX = "service:"


@dataclass(frozen=True)
class TokenPayload:
    sub: str
    aud: str
    iat: int
    exp: int


def decode_token(
    *,
    token: str,
    signing_key: bytes,
    accept_operator: bool = True,
    accept_any_service: bool = True,
) -> TokenPayload:
    """Verify a bearer token's signature + expiry and return its claims.

    Raises `jwt.InvalidTokenError` (or its `InvalidAudienceError`
    subclass) on any failure so the dependency can collapse all auth
    rejection paths into one except branch.
    """
    if not (accept_operator or accept_any_service):
        raise ValueError("must accept at least one audience class")

    options: Any = {
        "require": ["sub", "aud", "iat", "exp"],
        "verify_aud": False,
    }
    claims = jwt.decode(token, key=signing_key, algorithms=[_JWT_ALG], options=options)

    aud = str(claims["aud"])
    is_operator = accept_operator and aud == AUDIENCE_OPERATOR
    is_service = accept_any_service and aud.startswith(SERVICE_AUDIENCE_PREFIX)
    if not (is_operator or is_service):
        raise jwt.InvalidAudienceError(f"audience {aud!r} not accepted")

    return TokenPayload(
        sub=str(claims["sub"]),
        aud=aud,
        iat=int(claims["iat"]),
        exp=int(claims["exp"]),
    )
