from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from support_agent.db import Order, Refund, read_session, write_session


def _order_dict(order: Order, refund_id: int | None) -> dict:
    return {
        "order_id": order.id,
        "status": order.status,
        "total_cents": order.total_cents,
        "placed_at": order.placed_at.isoformat(),
        "note": order.note,
        "refunded": refund_id is not None,
    }


def _orders_query(customer_id: int):
    return (
        select(Order, Refund.id)
        .outerjoin(Refund, Refund.order_id == Order.id)
        .where(Order.customer_id == customer_id)
    )


async def get_order(customer_id: int, order_id: int) -> dict | None:
    async with read_session()() as session:
        row = (
            await session.execute(
                _orders_query(customer_id).where(Order.id == order_id)
            )
        ).one_or_none()
    return _order_dict(*row) if row else None


async def list_orders(customer_id: int) -> list[dict]:
    async with read_session()() as session:
        rows = await session.execute(
            _orders_query(customer_id).order_by(Order.placed_at.desc())
        )
        return [_order_dict(*row) for row in rows]


async def issue_refund(customer_id: int, order_id: int, reason: str) -> dict:
    try:
        async with write_session()() as session, session.begin():
            order = await session.scalar(
                select(Order).where(
                    Order.id == order_id, Order.customer_id == customer_id
                )
            )
            if order is None:
                return {"status": "not_found", "order_id": order_id}
            refund = Refund(
                order_id=order.id, amount_cents=order.total_cents, reason=reason
            )
            session.add(refund)
    except IntegrityError:
        return {"status": "already_refunded", "order_id": order_id}
    return {
        "status": "refunded",
        "order_id": order_id,
        "refund_id": refund.id,
        "amount_cents": refund.amount_cents,
    }
