import os
from datetime import datetime
from functools import cache

from sqlalchemy import ForeignKey
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

CONNECT_TIMEOUT_S = 5
QUERY_TIMEOUT_S = 5


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    customer_id: Mapped[int]
    status: Mapped[str]
    total_cents: Mapped[int]
    placed_at: Mapped[datetime]
    note: Mapped[str | None]


class Refund(Base):
    __tablename__ = "refunds"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"))
    amount_cents: Mapped[int]
    reason: Mapped[str]


def _sessionmaker(url_env: str) -> async_sessionmaker[AsyncSession]:
    engine = create_async_engine(
        os.environ[url_env],
        connect_args={"timeout": CONNECT_TIMEOUT_S, "command_timeout": QUERY_TIMEOUT_S},
    )
    return async_sessionmaker(engine, expire_on_commit=False)


@cache
def read_session() -> async_sessionmaker[AsyncSession]:
    return _sessionmaker("DATABASE_URL_READ")


@cache
def write_session() -> async_sessionmaker[AsyncSession]:
    return _sessionmaker("DATABASE_URL_WRITE")
