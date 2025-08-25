from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from app.db import get_session, engine, Base
from app.core.security import decode_token
from app.schemas import UserCreate, UserOut, Token, ClaimCreate, ClaimOut, Page
from app.services import auth_service, claim_service, stats_service
from app.models import ClaimStatus, User
from typing import List

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

router = APIRouter()

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_session)) -> User:
    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub"))
    except (JWTError, ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication")
    from sqlalchemy import select
    res = await db.execute(select(User).where(User.id == user_id))
    user = res.scalars().first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# Auth
@router.post("/auth/register", response_model=UserOut)
async def register(body: UserCreate, db: AsyncSession = Depends(get_session)):
    return await auth_service.register(db, email=body.email, password=body.password)

@router.post("/auth/login", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_session)):
    token = await auth_service.login(db, email=form.username, password=form.password)
    return {"access_token": token, "token_type": "bearer"}

# Claims
@router.post("/claims", response_model=ClaimOut)
async def create_claim(body: ClaimCreate, current=Depends(get_current_user), db: AsyncSession = Depends(get_session)):
    claim = await claim_service.create(db, user_id=current.id, payer=body.payer, amount=body.amount)
    return claim

@router.get("/claims", response_model=Page)
async def list_claims(page: int = 1, size: int = 10, current=Depends(get_current_user), db: AsyncSession = Depends(get_session)):
    items, total = await claim_service.list_by_user(db, user_id=current.id, page=page, size=size)
    return Page(items=[ClaimOut.model_validate(i) for i in items], total=total, page=page, size=size)

@router.post("/claims/{claim_id}/approve", response_model=ClaimOut)
async def approve_claim(claim_id: int, paid_amount: float, current=Depends(get_current_user), db: AsyncSession = Depends(get_session)):
    return await claim_service.update_status(db, claim_id=claim_id, status=ClaimStatus.APPROVED, paid_amount=paid_amount)

@router.post("/claims/{claim_id}/deny", response_model=ClaimOut)
async def deny_claim(claim_id: int, current=Depends(get_current_user), db: AsyncSession = Depends(get_session)):
    return await claim_service.update_status(db, claim_id=claim_id, status=ClaimStatus.DENIED)

# Stats
@router.get("/stats/denial_rate")
async def denial_rate(db: AsyncSession = Depends(get_session)):
    return {"denial_rate": await stats_service.denial_rate(db)}

@router.get("/stats/revenue_trend")
async def revenue_trend(db: AsyncSession = Depends(get_session)):
    return await stats_service.revenue_trend_monthly(db)

# Admin seed (for demo only)
@router.post("/admin/seed")
async def seed(db: AsyncSession = Depends(get_session)):
    # create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # create user and some claims
    from app.services import auth_service, claim_service
    try:
        user = await auth_service.register(db, email="demo@clinic.com", password="demo123")
    except Exception:
        from sqlalchemy import select
        from app.models import User
        res = await db.execute(select(User).where(User.email=="demo@clinic.com"))
        user = res.scalars().first()

    import random, datetime
    payers = ["Aetna","United","BCBS","Cigna"]
    for i in range(20):
        amount = random.choice([120.0, 260.0, 540.0, 980.0])
        claim = await claim_service.create(db, user_id=user.id, payer=random.choice(payers), amount=amount)
        # randomly approve/deny
        if random.random() < 0.6:
            paid = round(amount * random.uniform(0.5, 0.9), 2)
            await claim_service.update_status(db, claim_id=claim.id, status=ClaimStatus.APPROVED, paid_amount=paid)
        elif random.random() < 0.5:
            await claim_service.update_status(db, claim_id=claim.id, status=ClaimStatus.DENIED)
    return {"ok": True}
