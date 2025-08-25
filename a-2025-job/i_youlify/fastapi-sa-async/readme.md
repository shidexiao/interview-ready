# README 摘要
```md
## 快速开始
1. `cp .env.example .env` 并修改连接串
2. `docker-compose up -d` 启动 Postgres
3. 创建表：`alembic upgrade head`
4. 运行：`uvicorn app.main:app --reload`
5. 多租户：请求头携带 `X-Tenant-ID: clinic_a`，需先在 DB 中创建 `clinic_a` schema 并迁移。

## 多租户策略
- **Schema per tenant**：通过 `SET search_path TO <schema>` 实现。每个租户独立 schema，共享同一套表结构。
- 创建新租户：
  ```sql
  CREATE SCHEMA clinic_a;
  -- 迁移结构（以同步驱动执行）：
  ALEMBIC_CONFIG=. alembic -x schema=clinic_a upgrade head
  ```