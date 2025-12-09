"""Main application orchestrator for H5nry."""

from __future__ import annotations

from pathlib import Path

from h5nry.config import AppConfig, ConfigManager
from h5nry.llm.anthropic_client import AnthropicClient
from h5nry.llm.base import LLMClient
from h5nry.llm.gemini_client import GeminiClient
from h5nry.llm.openai_client import OpenAIClient
from h5nry.session import H5nrySession
from h5nry.tools.hdf5_tree import build_tree


class H5nryApp:
    """Main application orchestrator."""

    def __init__(self, config_manager: ConfigManager | None = None):
        """Initialize application.

        Args:
            config_manager: Configuration manager (creates default if None)
        """
        self.config_manager = config_manager or ConfigManager()

    def _create_llm_client(self, config: AppConfig) -> LLMClient:
        """Create LLM client based on configuration.

        Args:
            config: Application configuration

        Returns:
            LLM client instance

        Raises:
            ValueError: If provider is unknown or API key is missing
        """
        # Get API key
        api_key = self.config_manager.get_api_key(config.provider)
        if not api_key:
            raise ValueError(
                f"No API key found for provider '{config.provider}'. "
                f"Use 'h5nry login {config.provider}' or set {config.provider.upper()}_API_KEY environment variable."
            )

        # Create appropriate client
        if config.provider == "openai":
            return OpenAIClient(
                api_key=api_key,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        elif config.provider == "anthropic":
            return AnthropicClient(
                api_key=api_key,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        elif config.provider == "gemini":
            return GeminiClient(
                api_key=api_key,
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        else:
            raise ValueError(f"Unknown provider: {config.provider}")

    async def create_session(self, file_path: str | Path) -> H5nrySession:
        """Create a new session for a file.

        Args:
            file_path: Path to HDF5 file

        Returns:
            Initialized session

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file can't be opened as HDF5
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load configuration
        config = self.config_manager.get_config()

        # Create LLM client
        llm_client = self._create_llm_client(config)

        # Parse HDF5 file
        tree = build_tree(file_path)

        # Create session
        return H5nrySession(
            llm_client=llm_client,
            config=config,
            file_path=file_path,
            tree=tree,
        )

    async def ask_once(self, file_path: str | Path, question: str) -> str:
        """Ask a single question about a file.

        Args:
            file_path: Path to HDF5 file
            question: Question to ask

        Returns:
            Answer from the LLM
        """
        session = await self.create_session(file_path)
        return await session.ask(question)

    def run_tui(self, file_path: str | Path) -> None:
        """Launch the TUI for a file.

        Args:
            file_path: Path to HDF5 file
        """
        from h5nry.tui import H5nryTUI

        app = H5nryTUI(file_path, self.config_manager)
        app.run()
