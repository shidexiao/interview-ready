from typing import Optional, Tuple, Sequence
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException, status
from app.core.security import hash_password, verify_password, create_access_token
from app.models import User, Claim, ClaimStatus
from app.repositories import UserRepo, ClaimRepo

user_repo = UserRepo()
claim_repo = ClaimRepo()

class AuthService:
    async def register(self, db: AsyncSession, *, email: str, password: str) -> User:
        existing = await user_repo.get_by_email(db, email)
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")
        user = await user_repo.create(db, email=email, hashed_password=hash_password(password))
        return user

    async def login(self, db: AsyncSession, *, email: str, password: str) -> str:
        user = await user_repo.get_by_email(db, email)
        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        return create_access_token(sub=str(user.id))

class ClaimService:
    async def create(self, db: AsyncSession, *, user_id: int, payer: str, amount: float) -> Claim:
        return await claim_repo.create(db, user_id=user_id, payer=payer, amount=amount)

    async def list_by_user(self, db: AsyncSession, *, user_id: int, page: int, size: int) -> tuple[list[Claim], int]:
        items, total = await claim_repo.list_by_user(db, user_id, page, size)
        return list(items), total

    async def update_status(self, db: AsyncSession, *, claim_id: int, status: ClaimStatus, paid_amount: float = 0.0) -> Claim:
        claim = await claim_repo.update_status(db, claim_id, status, paid_amount)
        if not claim:
            raise HTTPException(status_code=404, detail="Claim not found")
        return claim

class StatsService:
    async def denial_rate(self, db: AsyncSession) -> float:
        return await claim_repo.denial_rate(db)

    async def revenue_trend_monthly(self, db: AsyncSession):
        return await claim_repo.revenue_trend_monthly(db)

auth_service = AuthService()
claim_service = ClaimService()
stats_service = StatsService()
