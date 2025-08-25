from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_env: str = "dev"
    api_prefix: str = "/api/v1"
    database_url: str
    sync_database_url: str
    default_tenant: str = "public"

    model_config = {
        "env_file": ".env",
        "env_prefix": "",
        "case_sensitive": False,
    }

settings = Settings()