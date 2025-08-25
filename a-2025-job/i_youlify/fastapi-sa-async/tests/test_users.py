# 测试

## tests/test_users.py

import asyncio
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_create_and_get_user():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.post("/api/v1/users/", json={"email": "a@b.com", "full_name": "Alice"}, headers={"X-Tenant-ID": "public"})
        assert r.status_code == 200
        uid = r.json()["id"]
        r2 = await ac.get(f"/api/v1/users/{uid}", headers={"X-Tenant-ID": "public"})
        assert r2.status_code == 200
        assert r2.json()["email"] == "a@b.com"
