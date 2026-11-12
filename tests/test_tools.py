import asyncio
import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from support_agent.db import read_session, write_session
from support_agent.tools import get_order, issue_refund, list_orders

pytestmark = pytest.mark.skipif(
    os.environ.get("CI_SKIP_DB") == "1", reason="no Postgres in CI"
)

ALICE, BOB = 1, 2
ALICE_ORDER, ALICE_BIG_ORDER, BOB_ORDER = 1, 2, 4
INSUFFICIENT_PRIVILEGE = "42501"


async def refund_count(admin_engine, order_id: int) -> int:
    async with admin_engine.connect() as conn:
        return await conn.scalar(
            text("SELECT count(*) FROM refunds WHERE order_id = :id"), {"id": order_id}
        )


async def test_get_order_returns_own_order(no_refunds):
    order = await get_order(ALICE, ALICE_BIG_ORDER)
    assert order["total_cents"] == 15000
    assert order["refunded"] is False


async def test_get_order_hides_other_customers_order():
    assert await get_order(ALICE, BOB_ORDER) is None


async def test_list_orders_returns_only_own_orders_newest_first():
    orders = await list_orders(ALICE)
    assert [o["order_id"] for o in orders] == [3, 2, 1]


async def test_issue_refund_writes_one_row(admin_engine, no_refunds):
    result = await issue_refund(ALICE, ALICE_BIG_ORDER, "damaged")
    assert result["status"] == "refunded"
    assert result["amount_cents"] == 15000
    assert await refund_count(admin_engine, ALICE_BIG_ORDER) == 1
    assert (await get_order(ALICE, ALICE_BIG_ORDER))["refunded"] is True


async def test_second_refund_is_rejected(admin_engine, no_refunds):
    await issue_refund(ALICE, ALICE_ORDER, "damaged")
    result = await issue_refund(ALICE, ALICE_ORDER, "damaged")
    assert result["status"] == "already_refunded"
    assert await refund_count(admin_engine, ALICE_ORDER) == 1


async def test_concurrent_refunds_write_one_row(admin_engine, no_refunds):
    results = await asyncio.gather(
        issue_refund(ALICE, ALICE_ORDER, "damaged"),
        issue_refund(ALICE, ALICE_ORDER, "damaged"),
    )
    assert sorted(r["status"] for r in results) == ["already_refunded", "refunded"]
    assert await refund_count(admin_engine, ALICE_ORDER) == 1


async def test_refund_of_other_customers_order_is_not_found(admin_engine, no_refunds):
    result = await issue_refund(ALICE, BOB_ORDER, "damaged")
    assert result["status"] == "not_found"
    assert await refund_count(admin_engine, BOB_ORDER) == 0


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO refunds (order_id, amount_cents, reason) VALUES (1, 4999, 'x')",
        "UPDATE orders SET total_cents = 1 WHERE id = 1",
    ],
)
async def test_read_role_cannot_write(no_refunds, statement):
    with pytest.raises(ProgrammingError) as exc:
        async with read_session()() as session, session.begin():
            await session.execute(text(statement))
    assert exc.value.orig.sqlstate == INSUFFICIENT_PRIVILEGE


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE orders SET total_cents = 1 WHERE id = 1",
        "DELETE FROM refunds",
        "SELECT email FROM customers",
    ],
)
async def test_write_role_limited_to_refund_insert(statement):
    with pytest.raises(ProgrammingError) as exc:
        async with write_session()() as session, session.begin():
            await session.execute(text(statement))
    assert exc.value.orig.sqlstate == INSUFFICIENT_PRIVILEGE
