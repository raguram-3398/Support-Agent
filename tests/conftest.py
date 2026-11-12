import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


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
