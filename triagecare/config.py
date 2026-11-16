from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime settings. Every field can be overridden with an environment
    variable of the same name (case-insensitive) or a line in `.env`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "TriageCare"
    environment: str = "development"
    secret_key: str = "dev-only-change-me"
    database_url: str = f"sqlite:///{ROOT_DIR / 'triagecare.db'}"
    timezone: str = "Asia/Karachi"
    hospital_name: str = "City General Hospital"

    model_path: Path = ROOT_DIR / "models" / "triage_pipeline.joblib"
    model_card_path: Path = ROOT_DIR / "models" / "model_card.json"

    # Demo mode shows one-click demo accounts on the login page.
    # `python -m triagecare.cli seed` creates those accounts with demo_password.
    demo_mode: bool = True
    demo_password: str = "triage123"

    @property
    def sqlalchemy_url(self) -> str:
        url = self.database_url
        # Render and Heroku hand out postgres:// URLs; SQLAlchemy wants an explicit driver.
        if url.startswith("postgres://"):
            url = "postgresql+psycopg://" + url[len("postgres://") :]
        elif url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://") :]
        return url

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
