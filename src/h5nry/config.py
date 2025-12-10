"""Configuration management for H5nry.

This module provides a ConfigManager that loads and manages user configuration
from ~/.h5nry/config.yaml. The initial configuration is seeded from the packaged
YAML template and then written to the user's config directory on first run.
Subsequent runs will round-trip the user's file (preserving comments and ordering)
and merge in any new options shipped with h5nry.
"""

from __future__ import annotations

import copy
import os
import warnings
from enum import Enum
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from ruamel.yaml import YAML

from h5nry import __version__


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
    """Singleton class to manage h5nry configuration.

    This class loads configuration from ~/.h5nry/config.yaml and provides
    helpers to access values. It ensures only one instance exists throughout
    the application lifecycle.

    Behavior:
      - On first run, copies the packaged default_config.yaml to the user's
        config path, stamping version with the running __version__.
      - On subsequent runs, loads and round-trips the user's config, merging
        any new options from the packaged template while preserving comments,
        ordering, and user values.
      - If the user's config version differs from the package template version,
        automatically updates the stored version and writes the file.
    """

    # Singleton instance and initialization flag
    _instance: ConfigManager | None = None
    _initialized: bool = False

    def __new__(cls) -> ConfigManager:
        """Ensure only one instance of ConfigManager exists.

        Returns:
            ConfigManager: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the ConfigManager (only happens once)."""
        if self._initialized:
            return

        # Paths and YAML loader configured for round-trip behavior
        self._yaml = YAML(typ="rt")
        self._yaml.preserve_quotes = True
        self._yaml.indent(mapping=2, sequence=4, offset=2)

        self._config_dir = Path.home() / ".h5nry"
        self._config_path = self._config_dir / "config.yaml"
        self._secrets_path = self._config_dir / "secrets.yaml"
        self._template_path = importlib_resources.files("h5nry.data").joinpath(
            "default_config.yaml"
        )

        # Round-trip document and plain dict mirrors
        self._config_doc: Any = None
        self._config: dict[str, Any] = {}

        self._load_config()
        ConfigManager._initialized = True

    def _ensure_config_dir_exists(self) -> None:
        """Create the config directory if it does not exist."""
        self._config_dir.mkdir(parents=True, exist_ok=True)

    def _load_default_template(self) -> Any:
        """Load the packaged default YAML template as a round-trip document.

        The template's version is overwritten with the current package version.

        Returns:
            Any: A ruamel CommentedMap representing the template.
        """
        with (
            importlib_resources.as_file(self._template_path) as p,
            p.open("r", encoding="utf-8") as f,
        ):
            doc = self._yaml.load(f)
        # Stamp the runtime version into the template
        doc["version"] = str(__version__)
        return doc

    def _create_default_config(self) -> None:
        """Create the initial user config file from the packaged template.

        Raises:
            OSError: If the file cannot be written.
        """
        self._ensure_config_dir_exists()
        tmpl = self._load_default_template()
        with self._config_path.open("w", encoding="utf-8") as f:
            self._yaml.dump(tmpl, f)

    def _load_config(self) -> None:
        """Load configuration from disk, creating or migrating as needed.

        The algorithm is:
          1. If the user config does not exist, create it from the template.
          2. Load the user config and the packaged template as round-trip docs.
          3. Recursively add any keys missing from the user config using the
             template as reference (preserving comments).
          4. If version differs, update it and mark the document as changed.
          5. If changed, write the updated doc back to disk.
          6. Cache a plain-dict mirror for fast lookups.

        On YAML parse errors, a warning is emitted and a fresh default config
        is written to the user's config path.

        Raises:
            OSError: If reading or writing the config file fails.
        """
        # Does the user's config file exist?
        if not self._config_path.exists():
            self._create_default_config()

        # Load user config, recreating on parse errors
        try:
            with self._config_path.open("r", encoding="utf-8") as f:
                user_doc = self._yaml.load(f) or {}
        except Exception as exc:
            warnings.warn(
                f"Failed to read config at '{self._config_path}': {exc}. "
                "Using default configuration.",
                stacklevel=2,
            )
            self._create_default_config()
            with self._config_path.open("r", encoding="utf-8") as f:
                user_doc = self._yaml.load(f) or {}

        # Load default template
        default_doc = self._load_default_template()

        # Merge missing keys from default into user config
        changed = self._add_missing_keys(user_doc, default_doc)

        # Update version if it differs
        if str(user_doc.get("version")) != str(default_doc.get("version")):
            user_doc["version"] = default_doc["version"]
            changed = True

        # If we made any changes, write back to disk
        if changed:
            try:
                with self._config_path.open("w", encoding="utf-8") as f:
                    self._yaml.dump(user_doc, f)
            except Exception as exc:
                warnings.warn(f"Failed to write updated config: {exc}", stacklevel=2)

        # Cache both round-trip and plain forms
        self._config_doc = user_doc
        self._config = self._to_plain(user_doc)

    @staticmethod
    def _add_missing_keys(target: Any, reference: Any) -> bool:
        """Recursively add keys from reference to target if absent.

        Args:
            target: The document to modify in place.
            reference: The document providing the reference structure.

        Returns:
            bool: True if target was modified; False otherwise.
        """
        changed = False
        if not isinstance(reference, dict) or not isinstance(target, dict):
            return changed

        for key, ref_val in reference.items():
            if key not in target:
                target[key] = copy.deepcopy(ref_val)
                changed = True
            else:
                tgt_val = target[key]
                if (
                    isinstance(ref_val, dict)
                    and isinstance(tgt_val, dict)
                    and ConfigManager._add_missing_keys(tgt_val, ref_val)
                ):
                    changed = True
        return changed

    @staticmethod
    def _to_plain(node: Any) -> Any:
        """Convert a ruamel node tree to standard Python objects.

        Args:
            node: A value that may contain ruamel CommentedMap/CommentedSeq.

        Returns:
            Any: The same data represented as dict, list, and scalars.
        """
        if isinstance(node, dict):
            return {k: ConfigManager._to_plain(v) for k, v in node.items()}
        if isinstance(node, list):
            return [ConfigManager._to_plain(x) for x in node]
        return node

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

        Loads config, applies environment overrides, and returns validated object.

        Returns:
            Application configuration object
        """
        # Start with loaded config
        config = self._config.copy()

        # Apply environment overrides
        config = self._apply_env_overrides(config)

        return AppConfig(**config)

    def save_config(self, config: AppConfig) -> None:
        """Save configuration to user config file.

        Args:
            config: Configuration to save
        """
        self._ensure_config_dir_exists()

        # Convert to dict and update the round-trip document
        config_dict = config.model_dump(
            mode="json",
            exclude_none=True,
            exclude={"openai_api_key", "anthropic_api_key", "gemini_api_key"},
        )

        # Update the existing doc to preserve comments
        for key, value in config_dict.items():
            if key in self._config_doc:
                self._config_doc[key] = value

        with self._config_path.open("w", encoding="utf-8") as f:
            self._yaml.dump(self._config_doc, f)

    def set_api_key(self, provider: str, api_key: str) -> None:
        """Set API key for a provider.

        Args:
            provider: Provider name (openai, anthropic, gemini)
            api_key: API key to store
        """
        self._ensure_config_dir_exists()

        # Load existing secrets
        secrets = {}
        if self._secrets_path.exists():
            with self._secrets_path.open("r", encoding="utf-8") as f:
                secrets = self._yaml.load(f) or {}

        secrets[f"{provider}_api_key"] = api_key

        with self._secrets_path.open("w", encoding="utf-8") as f:
            self._yaml.dump(secrets, f)

        # Set restrictive permissions
        self._secrets_path.chmod(0o600)

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
        if self._secrets_path.exists():
            with self._secrets_path.open("r", encoding="utf-8") as f:
                secrets = self._yaml.load(f) or {}
                return secrets.get(f"{provider}_api_key")

        return None

    def reload(self) -> None:
        """Reload configuration from disk.

        Useful for testing or if the config file is modified while the
        application is running.
        """
        self._load_config()

    @property
    def config_path(self) -> Path:
        """Path to the user's config.yaml."""
        return self._config_path

    @property
    def config(self) -> dict[str, Any]:
        """A copy of the full configuration dictionary.

        Returns:
            dict: The loaded configuration.
        """
        return self._config.copy()
