from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    PROJECT_NAME: str = "Secure Medical Imaging"
    API_V1_PREFIX: str = "/api"

    # Database
    DATABASE_URL: str = "sqlite:///./secure_medical.db"

    # JWT
    JWT_SECRET_KEY: str = "dev-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 30

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = Path(__file__).resolve().parent.parent / "uploads"
    MODEL_DIR: Path = Path(__file__).resolve().parent.parent / "ml" / "weights"

    # CORS — accept comma-separated list from env (e.g. "https://a.com,https://b.com")
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    # Optional regex for matching dynamic preview URLs (Vercel previews etc.)
    CORS_ORIGIN_REGEX: Optional[str] = None

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
