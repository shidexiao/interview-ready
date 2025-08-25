from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import event
from sqlalchemy.pool import Pool
from .base import Base
from app.core.config import settings

engine = create_async_engine(settings.database_url, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# 在获取连接时设置 search_path，实现 schema 级多租户
async def set_search_path(conn, tenant: str):
    await conn.exec_driver_sql(f"SET search_path TO {tenant}")

async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)