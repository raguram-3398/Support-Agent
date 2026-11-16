import asyncio
import json
import os
import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from mcp.server import MCPServer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from support_agent.auth import ISSUER

SMALL_ORDER, LARGE_ORDER, FAILING_ORDER, BOB_ORDER = 1, 2, 3, 4
TOTALS = {SMALL_ORDER: 4999, LARGE_ORDER: 15000, FAILING_ORDER: 2500, BOB_ORDER: 8900}
SLOW_S = 1
KID = "test-key"
TOKEN_LIFETIME_S = 300

SIGNING_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PUBLIC_JWK = {
    **json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(SIGNING_KEY.public_key())),
    "kid": KID,
    "use": "sig",
    "alg": "RS256",
}


def make_token(audience: str, headers: dict | None = None, **claims) -> str:
    now = int(time.time())
    payload = {
        "iss": ISSUER,
        "aud": audience,
        "sub": "user-1",
        "iat": now,
        "exp": now + TOKEN_LIFETIME_S,
        **claims,
    }
    return jwt.encode(
        payload, SIGNING_KEY, algorithm="RS256", headers={"kid": KID, **(headers or {})}
    )


def serve_test_jwks(monkeypatch, verifier) -> None:
    monkeypatch.setattr(verifier.jwks, "fetch_data", lambda: {"keys": [PUBLIC_JWK]})


@pytest.fixture(scope="session")
async def admin_engine():
    engine = create_async_engine(os.environ["DATABASE_URL_ADMIN"])
    yield engine
    await engine.dispose()


@pytest.fixture
async def no_refunds(admin_engine):
    async with admin_engine.begin() as conn:
        await conn.execute(text("TRUNCATE refunds"))
    yield
    async with admin_engine.begin() as conn:
        await conn.execute(text("TRUNCATE refunds"))


@pytest.fixture
def fake_mcp():
    server = MCPServer("fake")
    lookups, refunds = [], []

    @server.tool()
    async def get_order(customer_id: int, order_id: int) -> dict[str, Any] | None:
        lookups.append((customer_id, order_id))
        return {
            "order_id": order_id,
            "status": "delivered",
            "total_cents": TOTALS[order_id],
        }

    @server.tool()
    async def list_orders(customer_id: int) -> list[dict[str, Any]]:
        await asyncio.sleep(SLOW_S)
        return []

    @server.tool()
    async def issue_refund(
        customer_id: int, order_id: int, reason: str
    ) -> dict[str, Any]:
        if order_id == FAILING_ORDER:
            raise RuntimeError("database unavailable")
        refunds.append((customer_id, order_id))
        return {"status": "refunded", "order_id": order_id}

    return server, lookups, refunds
