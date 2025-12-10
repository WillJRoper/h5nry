"""Tests for configuration management."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from h5nry.config import AppConfig, ConfigManager, SafetyLevel


def test_app_config_defaults():
    """Test that AppConfig has sensible defaults."""
    config = AppConfig()

    assert config.provider == "openai"
    assert config.model == "gpt-4-turbo-preview"
    assert config.temperature == 0.1
    assert config.max_tokens is None
    assert config.stream is True
    assert config.safety_level == SafetyLevel.TOOLS_ONLY
    assert config.max_data_gb == 0.5
    assert config.recent_code_limit == 20


def test_app_config_validation():
    """Test that AppConfig validates constraints."""
    # Temperature out of range
    with pytest.raises(ValueError):
        AppConfig(temperature=3.0)

    with pytest.raises(ValueError):
        AppConfig(temperature=-0.1)

    # Invalid max_data_gb
    with pytest.raises(ValueError):
        AppConfig(max_data_gb=0)

    with pytest.raises(ValueError):
        AppConfig(max_data_gb=-1)

    # Invalid recent_code_limit
    with pytest.raises(ValueError):
        AppConfig(recent_code_limit=0)


def test_config_manager_with_temp_dir():
    """Test ConfigManager with a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / ".h5nry"
        config_manager = ConfigManager(config_dir=config_dir)

        # Get default config (should work even without files)
        config = config_manager.get_config()
        assert isinstance(config, AppConfig)
        assert config.provider == "openai"


def test_config_manager_env_overrides():
    """Test that environment variables override config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / ".h5nry"
        config_manager = ConfigManager(config_dir=config_dir)

        # Set environment variables
        os.environ["H5NRY_PROVIDER"] = "anthropic"
        os.environ["H5NRY_MODEL"] = "claude-3-opus-20240229"
        os.environ["H5NRY_TEMPERATURE"] = "0.5"
        os.environ["H5NRY_MAX_DATA_GB"] = "1.0"

        try:
            config = config_manager.get_config()

            assert config.provider == "anthropic"
            assert config.model == "claude-3-opus-20240229"
            assert config.temperature == 0.5
            assert config.max_data_gb == 1.0

        finally:
            # Clean up environment
            for key in [
                "H5NRY_PROVIDER",
                "H5NRY_MODEL",
                "H5NRY_TEMPERATURE",
                "H5NRY_MAX_DATA_GB",
            ]:
                os.environ.pop(key, None)


def test_config_manager_save_and_load():
    """Test saving and loading configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / ".h5nry"
        config_manager = ConfigManager(config_dir=config_dir)

        # Create custom config
        config = AppConfig(
            provider="anthropic",
            model="claude-3-sonnet-20240229",
            temperature=0.3,
            max_data_gb=1.5,
        )

        # Save it
        config_manager.save_config(config)

        # Load it back
        loaded_config = config_manager.get_config()

        assert loaded_config.provider == "anthropic"
        assert loaded_config.model == "claude-3-sonnet-20240229"
        assert loaded_config.temperature == 0.3
        assert loaded_config.max_data_gb == 1.5


def test_api_key_storage():
    """Test API key storage and retrieval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir) / ".h5nry"
        config_manager = ConfigManager(config_dir=config_dir)

        # Set API key
        config_manager.set_api_key("openai", "test-key-12345")

        # Retrieve it
        api_key = config_manager.get_api_key("openai")
        assert api_key == "test-key-12345"

        # Check that secrets file has correct permissions (Unix-like systems)
        secrets_file = config_dir / "secrets.yaml"
        if os.name != "nt":  # Skip on Windows
            stat_info = secrets_file.stat()
            assert oct(stat_info.st_mode)[-3:] == "600"
