from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from db.database import get_db
from db.crud import create_user, get_user_by_username, verify_password
from db.schemas import UserCreate, UserResponse, TokenResponse, LoginRequest
from db.models import User, UserRole
from .deps import create_access_token, get_current_user

router = APIRouter()


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != UserRole.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


class UpdateRoleRequest(BaseModel):
    role: str


class UpdateActiveRequest(BaseModel):
    is_active: bool


class UserListResponse(BaseModel):
    users: list[UserResponse]
    total: int


@router.get("/health")
async def health():
    return {"status": "ok", "module": "auth"}


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    existing = get_user_by_username(db, user_data.username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        )
    try:
        user = create_user(db, user_data)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        )
    return user


@router.post("/login", response_model=TokenResponse)
async def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    user = get_user_by_username(db, login_data.username)
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={"sub": user.username})
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user


# --- Admin endpoints ---

@router.get("/users", response_model=UserListResponse)
async def list_users(
    _admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """List all users (admin only)."""
    users = db.query(User).order_by(User.id.asc()).all()
    return UserListResponse(users=users, total=len(users))


@router.patch("/users/{user_id}/role", response_model=UserResponse)
async def update_user_role(
    user_id: int,
    req: UpdateRoleRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Change a user's role (admin only)."""
    valid_roles = {r.value for r in UserRole}
    if req.role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {sorted(valid_roles)}")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Don't let the last admin demote themselves
    if user.id == admin.id and user.role == UserRole.admin and req.role != UserRole.admin.value:
        admin_count = db.query(User).filter(User.role == UserRole.admin).count()
        if admin_count <= 1:
            raise HTTPException(status_code=400, detail="Cannot demote the last remaining admin")

    user.role = UserRole(req.role)
    db.commit()
    db.refresh(user)
    return user


@router.patch("/users/{user_id}/active", response_model=UserResponse)
async def update_user_active(
    user_id: int,
    req: UpdateActiveRequest,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Activate or deactivate a user (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if user.id == admin.id and not req.is_active:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")

    user.is_active = req.is_active
    db.commit()
    db.refresh(user)
    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Delete a user (admin only). Cannot delete yourself."""
    if user_id == admin.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return None
