from fastapi import Header
from app.core.config import settings

TENANT_HEADER = "X-Tenant-ID"

async def resolve_tenant(x_tenant_id: str | None = Header(default=None)) -> str:
    # 可扩展为基于域名、JWT、Path 的解析
    return x_tenant_id or settings.default_tenant