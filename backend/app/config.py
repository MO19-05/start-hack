from pydantic import BaseSettings, AnyHttpUrl
from typing import List, Optional


class Settings(BaseSettings):
    # SMTP settings
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: Optional[int] = None
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM: Optional[str] = None
    SMTP_USE_TLS: bool = True
    ALERT_RECIPIENTS: Optional[str] = None  # comma-separated

    BACKEND_URL: Optional[AnyHttpUrl] = None

    class Config:
        env_file = ".env"


settings = Settings()
