from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime
from app.models import ClaimStatus

# Auth
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenPayload(BaseModel):
    sub: str
    exp: int

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    email: EmailStr
    is_active: bool
    class Config:
        from_attributes = True

# Claims
class ClaimCreate(BaseModel):
    payer: str = Field(..., examples=["Aetna", "United", "BCBS"])
    amount: float

class ClaimOut(BaseModel):
    id: int
    user_id: int
    payer: str
    amount: float
    paid_amount: float
    status: ClaimStatus
    created_at: datetime
    class Config:
        from_attributes = True

# Pagination
class Page(BaseModel):
    items: list
    total: int
    page: int
    size: int
