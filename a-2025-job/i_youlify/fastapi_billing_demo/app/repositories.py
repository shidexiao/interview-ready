from typing import Optional, Sequence, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from app.models import User, Claim, ClaimStatus

class UserRepo:
    async def get_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        res = await db.execute(select(User).where(User.email == email))
        return res.scalars().first()

    async def create(self, db: AsyncSession, email: str, hashed_password: str) -> User:
        user = User(email=email, hashed_password=hashed_password)
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user

class ClaimRepo:
    async def create(self, db: AsyncSession, *, user_id: int, payer: str, amount: float) -> Claim:
        claim = Claim(user_id=user_id, payer=payer, amount=amount, status=ClaimStatus.PENDING, paid_amount=0.0)
        db.add(claim)
        await db.commit()
        await db.refresh(claim)
        return claim

    async def list_by_user(self, db: AsyncSession, user_id: int, page: int, size: int) -> Tuple[Sequence[Claim], int]:
        q = select(Claim).where(Claim.user_id == user_id).order_by(desc(Claim.id))
        total = (await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one()
        items = (await db.execute(q.limit(size).offset((page-1)*size))).scalars().all()
        return items, total

    async def update_status(self, db: AsyncSession, claim_id: int, status: ClaimStatus, paid_amount: float = 0.0) -> Optional[Claim]:
        res = await db.execute(select(Claim).where(Claim.id == claim_id))
        claim = res.scalars().first()
        if not claim:
            return None
        claim.status = status
        claim.paid_amount = paid_amount
        await db.commit()
        await db.refresh(claim)
        return claim

    async def denial_rate(self, db: AsyncSession) -> float:
        # denied / (approved + denied)
        totals = await db.execute(select(
            func.sum(func.case((Claim.status == ClaimStatus.DENIED, 1), else_=0)),
            func.sum(func.case((Claim.status == ClaimStatus.APPROVED, 1), else_=0))
        ))
        denied, approved = totals.first() or (0, 0)
        denom = (approved or 0) + (denied or 0)
        return float(denied or 0) / denom if denom else 0.0

    async def revenue_trend_monthly(self, db: AsyncSession):
        # Sum of paid_amount by YYYY-MM
        res = await db.execute(
            select(func.strftime("%Y-%m", Claim.created_at).label("month"),
                   func.sum(Claim.paid_amount).label("revenue"))
            .where(Claim.status == ClaimStatus.APPROVED)
            .group_by("month")
            .order_by("month")
        )
        return [{"month": m, "revenue": float(r or 0)} for m, r in res.all()]
