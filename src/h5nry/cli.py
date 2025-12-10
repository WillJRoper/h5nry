"""Command-line interface for H5nry."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console

from h5nry.app import H5nryApp
from h5nry.config import ConfigManager

app = typer.Typer(
    name="h5nry",
    help="An AI assistant for investigating HDF5 files",
    add_completion=False,
)
console = Console()


@app.command()
def main_command(
    file_path: Path = typer.Argument(
        ...,
        help="Path to HDF5 file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
):
    """Launch the interactive TUI for exploring an HDF5 file."""
    try:
        config_manager = ConfigManager()
        h5nry_app = H5nryApp(config_manager)
        h5nry_app.run_tui(file_path)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", style="bold")
        sys.exit(1)


@app.command()
def ask(
    file_path: Path = typer.Argument(
        ...,
        help="Path to HDF5 file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    question: str = typer.Argument(..., help="Question to ask about the file"),
):
    """Ask a single question about an HDF5 file (non-interactive)."""
    try:
        config_manager = ConfigManager()
        h5nry_app = H5nryApp(config_manager)

        # Run async ask
        answer = asyncio.run(h5nry_app.ask_once(file_path, question))

        console.print("\n[bold green]H5nry:[/bold green]")
        console.print(answer)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}", style="bold")
        sys.exit(1)


@app.command()
def login(
    provider: str = typer.Argument(
        ...,
        help="Provider name (openai, anthropic, or gemini)",
    ),
    api_key: str = typer.Option(
        None,
        "--api-key",
        "-k",
        help="API key (will prompt if not provided)",
    ),
):
    """Configure API key for a provider."""
    # Validate provider
    valid_providers = ["openai", "anthropic", "gemini"]
    if provider.lower() not in valid_providers:
        console.print(
            f"[red]Error:[/red] Invalid provider '{provider}'. "
            f"Valid options: {', '.join(valid_providers)}"
        )
        sys.exit(1)

    provider = provider.lower()

    # Get API key
    if api_key is None:
        api_key = typer.prompt(
            f"Enter your {provider.capitalize()} API key",
            hide_input=True,
        )

    if not api_key:
        console.print("[red]Error:[/red] API key cannot be empty")
        sys.exit(1)

    # Save API key
    try:
        config_manager = ConfigManager()
        config_manager.set_api_key(provider, api_key)

        console.print(f"[green]Success![/green] API key for {provider} has been saved.")
        console.print(
            f"[dim]Stored in: {config_manager.config_dir / 'secrets.yaml'}[/dim]"
        )

    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to save API key: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        app()
    # If first argument is a file path (not a subcommand), use main_command
    elif len(sys.argv) == 2 and Path(sys.argv[1]).exists():
        main_command(Path(sys.argv[1]))
    else:
        app()


if __name__ == "__main__":
    main()
