from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from typing import AsyncGenerator

DATABASE_URL = "sqlite+aiosqlite:///./debug_eval.db"
engine = create_async_engine(DATABASE_URL, echo=True)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    # Disable attribute expiration on commit so objects remain accessible
    # after committing within long-running background tasks.
    async with AsyncSession(engine, expire_on_commit=False) as session:
        yield session
