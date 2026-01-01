from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for Milvus MCP Server."""

    milvus_uri: str = Field("http://localhost:19530", description="Milvus server URI")
    milvus_token: Optional[str] = Field(None, description="Milvus server authentication token")
    milvus_db: str = Field("default", description="Milvus database name")
    sse: bool = Field(False, description="Enable Server-Sent Events")

    model_config = SettingsConfigDict(env_file=".env")

    @field_validator("milvus_uri")
    @classmethod
    def validate_uri(cls, value: str) -> str:
        """Validate that the URI is not empty."""
        if not value or not value.strip():
            raise ValueError("Milvus URI cannot be empty")
        return value.strip()

    @field_validator("milvus_db")
    @classmethod
    def validate_db_name(cls, value: str) -> str:
        """Validate that the database name is not empty."""
        if not value or not value.strip():
            raise ValueError("Database name cannot be empty")
        return value.strip()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources to prioritize CLI arguments."""
        return (
            init_settings,
            CliSettingsSource(settings_cls, cli_parse_args=True, cli_ignore_unknown_args=True),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()  # type: ignore
