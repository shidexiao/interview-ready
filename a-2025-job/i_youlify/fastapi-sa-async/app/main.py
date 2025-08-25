from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.endpoints import users
from app.tenancy.middleware import DBSessionMiddleware

app = FastAPI(title="FastAPI SA Async MT")
app.add_middleware(DBSessionMiddleware)

api = FastAPI()
app.include_router(users.router, prefix=f"{settings.api_prefix}/users", tags=["users"])