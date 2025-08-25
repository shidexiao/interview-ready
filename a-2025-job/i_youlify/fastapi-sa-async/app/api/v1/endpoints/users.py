from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db
from app.schemas.user import UserCreate, UserOut
from app.crud.crud_user import user

router = APIRouter()

@router.post("/", response_model=UserOut)
async def create_user(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    return await user.create(db, payload)

@router.get("/{user_id}", response_model=UserOut | None)
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    return await user.get(db, user_id)