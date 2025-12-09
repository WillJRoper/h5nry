"""Configuration management for H5nry."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SafetyLevel(str, Enum):
    """Safety level for tool execution."""

    TOOLS_ONLY = "tools_only"
    TOOLS_PLUS_PYTHON = "tools_plus_python"


class AppConfig(BaseModel):
    """Application configuration."""

    # LLM Provider settings
    provider: Literal["openai", "anthropic", "gemini"] = Field(
        default="openai",
        description="LLM provider to use",
    )
    model: str = Field(
        default="gpt-4-turbo-preview",
        description="Model name/identifier",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM sampling",
    )
    max_tokens: int | None = Field(
        default=None,
        description="Maximum tokens to generate (None for provider default)",
    )
    stream: bool = Field(
        default=True,
        description="Enable streaming responses",
    )

    # Safety settings
    safety_level: SafetyLevel = Field(
        default=SafetyLevel.TOOLS_PLUS_PYTHON,
        description="Safety level for tool execution",
    )

    # Memory limits
    max_data_gb: float = Field(
        default=0.5,
        gt=0.0,
        description="Maximum data to load into memory at once (GB)",
    )

    # Code history
    recent_code_limit: int = Field(
        default=20,
        ge=1,
        description="Maximum number of code snippets to keep in memory",
    )

    # API keys (optional, can also use env vars)
    openai_api_key: str | None = Field(default=None, exclude=True)
    anthropic_api_key: str | None = Field(default=None, exclude=True)
    gemini_api_key: str | None = Field(default=None, exclude=True)


class ConfigManager:
    """Manages application configuration."""

    DEFAULT_CONFIG_DIR = Path.home() / ".h5nry"
    CONFIG_FILE = "config.yaml"
    PACKAGE_DEFAULT_CONFIG = Path(__file__).parent / "data" / "default_config.yaml"

    def __init__(self, config_dir: Path | None = None):
        """Initialize configuration manager.

        Args:
            config_dir: Directory for config files (default: ~/.h5nry)
        """
        self.config_dir = config_dir or self.DEFAULT_CONFIG_DIR
        self.config_path = self.config_dir / self.CONFIG_FILE

    def _load_defaults(self) -> dict:
        """Load default configuration from package data.

        Returns:
            Default configuration dictionary
        """
        if self.PACKAGE_DEFAULT_CONFIG.exists():
            with open(self.PACKAGE_DEFAULT_CONFIG) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _load_user_config(self) -> dict:
        """Load user configuration file.

        Returns:
            User configuration dictionary
        """
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _apply_env_overrides(self, config: dict) -> dict:
        """Apply environment variable overrides.

        Args:
            config: Base configuration dictionary

        Returns:
            Configuration with environment overrides
        """
        env_mappings = {
            "H5NRY_PROVIDER": "provider",
            "H5NRY_MODEL": "model",
            "H5NRY_TEMPERATURE": ("temperature", float),
            "H5NRY_MAX_TOKENS": ("max_tokens", int),
            "H5NRY_STREAM": ("stream", lambda x: x.lower() == "true"),
            "H5NRY_SAFETY_LEVEL": "safety_level",
            "H5NRY_MAX_DATA_GB": ("max_data_gb", float),
            "H5NRY_RECENT_CODE_LIMIT": ("recent_code_limit", int),
        }

        for env_var, mapping in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if isinstance(mapping, tuple):
                    key, converter = mapping
                    config[key] = converter(value)
                else:
                    config[mapping] = value

        # API keys from environment
        if key := os.getenv("OPENAI_API_KEY"):
            config["openai_api_key"] = key
        if key := os.getenv("ANTHROPIC_API_KEY"):
            config["anthropic_api_key"] = key
        if key := os.getenv("GEMINI_API_KEY"):
            config["gemini_api_key"] = key

        return config

    def get_config(self) -> AppConfig:
        """Get application configuration.

        Loads defaults, merges user config, and applies environment overrides.

        Returns:
            Application configuration object
        """
        # Start with defaults
        config = self._load_defaults()

        # Merge user config
        user_config = self._load_user_config()
        config.update(user_config)

        # Apply environment overrides
        config = self._apply_env_overrides(config)

        return AppConfig(**config)

    def save_config(self, config: AppConfig) -> None:
        """Save configuration to user config file.

        Args:
            config: Configuration to save
        """
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Convert to dict and save
        config_dict = config.model_dump(
            mode="json",
            exclude_none=True,
            exclude={"openai_api_key", "anthropic_api_key", "gemini_api_key"}
        )

        with open(self.config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for a provider.

        Args:
            provider: Provider name (openai, anthropic, gemini)
            api_key: API key to store
        """
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load existing config
        config = self.get_config()

        # Set the appropriate API key
        if provider == "openai":
            config.openai_api_key = api_key
        elif provider == "anthropic":
            config.anthropic_api_key = api_key
        elif provider == "gemini":
            config.gemini_api_key = api_key
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Save with API keys (store separately in a secrets file)
        secrets_path = self.config_dir / "secrets.yaml"
        secrets = {}
        if secrets_path.exists():
            with open(secrets_path) as f:
                secrets = yaml.safe_load(f) or {}

        secrets[f"{provider}_api_key"] = api_key

        with open(secrets_path, "w") as f:
            yaml.dump(secrets, f, default_flow_style=False)

        # Set restrictive permissions
        secrets_path.chmod(0o600)

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for a provider.

        Args:
            provider: Provider name (openai, anthropic, gemini)

        Returns:
            API key or None if not found
        """
        # Check environment first
        env_var = f"{provider.upper()}_API_KEY"
        if key := os.getenv(env_var):
            return key

        # Check secrets file
        secrets_path = self.config_dir / "secrets.yaml"
        if secrets_path.exists():
            with open(secrets_path) as f:
                secrets = yaml.safe_load(f) or {}
                return secrets.get(f"{provider}_api_key")

        return None
