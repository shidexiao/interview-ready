from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

def get_db(request: Request) -> AsyncSession:
    return request.state.db

def get_tenant(request: Request) -> str:
    return request.state.tenant