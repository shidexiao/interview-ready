from .base import CRUDBase
from app.models.user import User
from app.schemas.user import UserCreate

user = CRUDBase[User, UserCreate](User)