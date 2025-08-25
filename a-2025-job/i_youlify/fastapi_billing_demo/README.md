# FastAPI + SQLAlchemy Async (SQLite) Demo

Features:
- Layered architecture: router → service → repository
- JWT auth (register/login), password hashing
- Pagination, error handling
- Claims CRUD (minimal), denial rate & revenue trend stats
- Async SQLAlchemy + SQLite (aiosqlite)

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Then open: http://127.0.0.1:8000/docs

## Seed data
Call `POST /admin/seed` once to populate sample users and claims.
