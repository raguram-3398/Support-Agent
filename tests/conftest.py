import asyncio
import os
from typing import Any

import pytest
from mcp.server import MCPServer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

SMALL_ORDER, LARGE_ORDER, FAILING_ORDER, BOB_ORDER = 1, 2, 3, 4
TOTALS = {SMALL_ORDER: 4999, LARGE_ORDER: 15000, FAILING_ORDER: 2500, BOB_ORDER: 8900}
SLOW_S = 1


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
