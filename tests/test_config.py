"""Tests for config module covering Settings validation and get_settings."""

import pytest

from mcp_server_milvus.config import Settings, get_settings


def test_settings_validate_uri_empty():
    """Test that empty URI raises ValueError."""
    with pytest.raises(ValueError, match="Milvus URI cannot be empty"):
        Settings(milvus_uri="")


def test_settings_validate_uri_whitespace():
    """Test that whitespace-only URI raises ValueError."""
    with pytest.raises(ValueError, match="Milvus URI cannot be empty"):
        Settings(milvus_uri="   ")


def test_settings_validate_uri_strips_whitespace():
    """Test that URI whitespace is stripped."""
    settings = Settings(milvus_uri="  http://localhost:19530  ")
    assert settings.milvus_uri == "http://localhost:19530"


def test_settings_validate_db_name_empty():
    """Test that empty database name raises ValueError."""
    with pytest.raises(ValueError, match="Database name cannot be empty"):
        Settings(milvus_db="")


def test_settings_validate_db_name_whitespace():
    """Test that whitespace-only database name raises ValueError."""
    with pytest.raises(ValueError, match="Database name cannot be empty"):
        Settings(milvus_db="   ")


def test_settings_validate_db_name_strips_whitespace():
    """Test that database name whitespace is stripped."""
    settings = Settings(milvus_db="  my_db  ")
    assert settings.milvus_db == "my_db"


def test_get_settings_returns_instance():
    """Test that get_settings returns a Settings instance."""
    settings = get_settings()
    assert isinstance(settings, Settings)
    # Test that it's cached
    settings2 = get_settings()
    assert settings is settings2


def test_settings_customise_sources():
    """Test that settings_customise_sources is defined and returns tuple."""
    # This tests the method exists and can be called
    from pydantic_settings import CliSettingsSource, InitSettingsSource

    # Create mock sources
    init_settings = InitSettingsSource(Settings, init_kwargs={})
    env_settings = InitSettingsSource(Settings, init_kwargs={})
    dotenv_settings = InitSettingsSource(Settings, init_kwargs={})
    file_secret_settings = InitSettingsSource(Settings, init_kwargs={})

    result = Settings.settings_customise_sources(
        Settings,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    )

    assert isinstance(result, tuple)
    assert len(result) == 5
    # First should be init_settings
    assert result[0] is init_settings
    # Second should be CliSettingsSource
    assert isinstance(result[1], CliSettingsSource)
