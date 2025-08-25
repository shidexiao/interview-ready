from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import AsyncSessionLocal
from app.tenancy.utils import resolve_tenant

class DBSessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tenant = await resolve_tenant(request.headers.get("X-Tenant-ID"))
        async with AsyncSessionLocal() as session:
            # 将 session 注入到 request.state，路由依赖里读取
            request.state.db = session
            request.state.tenant = tenant
            # 每次事务开始时设置 search_path
            async with session.begin():
                await session.execute(f"SET search_path TO {tenant}")
            response: Response = await call_next(request)
        return response