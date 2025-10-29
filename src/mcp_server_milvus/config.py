from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, CliSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict


class Settings(BaseSettings):
    milvus_uri: str = Field('http://localhost:19530', description='Milvus server URI')
    milvus_token: str | None = Field(None, description='Milvus server authentication token')
    milvus_db: str = Field('default', description='Milvus database name')
    sse: bool = Field(False, description='Enable Server-Sent Events')

    model_config = SettingsConfigDict(env_file='.env')

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            CliSettingsSource(settings_cls, cli_parse_args=True, cli_ignore_unknown_args=True),
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore
