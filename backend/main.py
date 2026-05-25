from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from api import api_router
from db.database import engine, Base, SessionLocal
from db.models import User, UserRole

# Create database tables on startup
Base.metadata.create_all(bind=engine)


def _bootstrap_admin() -> None:
    """If no admin exists, promote the earliest-registered active user to admin."""
    db = SessionLocal()
    try:
        has_admin = db.query(User).filter(User.role == UserRole.admin).first() is not None
        if has_admin:
            return
        first_user = db.query(User).order_by(User.id.asc()).first()
        if first_user:
            first_user.role = UserRole.admin
            db.commit()
    finally:
        db.close()


_bootstrap_admin()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_origin_regex=settings.CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "project": settings.PROJECT_NAME}
