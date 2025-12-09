"""Textual TUI for H5nry."""

from __future__ import annotations

import asyncio
from pathlib import Path

from rich.panel import Panel
from rich.syntax import Syntax
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Static

from h5nry.app import H5nryApp
from h5nry.config import ConfigManager
from h5nry.session import H5nrySession


class MessageWidget(Static):
    """Widget for displaying a single message."""

    def __init__(self, role: str, content: str, **kwargs):
        """Initialize message widget.

        Args:
            role: Message role (user, assistant, etc.)
            content: Message content
            **kwargs: Additional widget arguments
        """
        self.role = role
        self.message_content = content

        # Create styled panel
        if role == "user":
            title = "You"
            border_style = "blue"
        elif role == "assistant":
            title = "H5nry"
            border_style = "green"
        elif role == "tool":
            title = "Tool Result"
            border_style = "yellow"
        else:
            title = role.capitalize()
            border_style = "white"

        panel = Panel(
            content,
            title=title,
            border_style=border_style,
            padding=(0, 1),
        )

        super().__init__(panel, **kwargs)


class CodeHistoryWidget(Static):
    """Widget for displaying code history."""

    def __init__(self, code_snippets: list[str], **kwargs):
        """Initialize code history widget.

        Args:
            code_snippets: List of code snippets
            **kwargs: Additional widget arguments
        """
        if not code_snippets:
            content = "No code has been executed yet."
        else:
            lines = []
            for i, code in enumerate(code_snippets, 1):
                lines.append(f"[bold cyan]Snippet {i}:[/bold cyan]")
                lines.append(f"```python\n{code}\n```")
                lines.append("")
            content = "\n".join(lines)

        panel = Panel(
            content,
            title="Code History",
            border_style="magenta",
            padding=(0, 1),
        )

        super().__init__(panel, **kwargs)


class CodeSnippetWidget(Static):
    """Widget for displaying a specific code snippet."""

    def __init__(self, index: int, code: str, **kwargs):
        """Initialize code snippet widget.

        Args:
            index: Snippet index
            code: Code content
            **kwargs: Additional widget arguments
        """
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        panel = Panel(
            syntax,
            title=f"Code Snippet {index}",
            border_style="magenta",
            padding=(0, 1),
        )

        super().__init__(panel, **kwargs)


class H5nryTUI(App):
    """Textual TUI application for H5nry."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #status-bar {
        dock: top;
        height: 3;
        background: $surface;
        padding: 0 1;
    }

    #messages-container {
        height: 1fr;
        padding: 1;
    }

    #input-container {
        dock: bottom;
        height: auto;
        background: $surface;
        padding: 0 1;
    }

    MessageWidget {
        margin: 0 0 1 0;
    }

    CodeHistoryWidget {
        margin: 0 0 1 0;
    }

    CodeSnippetWidget {
        margin: 0 0 1 0;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, file_path: str | Path, config_manager: ConfigManager):
        """Initialize TUI.

        Args:
            file_path: Path to HDF5 file
            config_manager: Configuration manager
        """
        super().__init__()
        self.file_path = Path(file_path)
        self.config_manager = config_manager
        self.session: H5nrySession | None = None
        self.app_instance = H5nryApp(config_manager)

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header()

        # Status bar
        config = self.config_manager.get_config()
        status_text = (
            f"File: {self.file_path.name} | "
            f"Provider: {config.provider} | "
            f"Model: {config.model} | "
            f"Safety: {config.safety_level.value}"
        )
        yield Static(status_text, id="status-bar")

        # Messages container
        yield VerticalScroll(id="messages-container")

        # Input container
        with Container(id="input-container"):
            yield Input(placeholder="Ask a question or type /history, /show N...")

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize session when app mounts."""
        try:
            self.session = await self.app_instance.create_session(self.file_path)

            # Add welcome message
            messages_container = self.query_one("#messages-container", VerticalScroll)
            welcome = MessageWidget(
                "assistant",
                f"Hello! I'm H5nry, ready to help you explore {self.file_path.name}.\n\n"
                "You can ask me questions about the file structure, compute statistics, "
                "create plots, or analyze datasets. Type /history to see executed code snippets."
            )
            await messages_container.mount(welcome)

            # Focus input
            self.query_one(Input).focus()

        except Exception as e:
            messages_container = self.query_one("#messages-container", VerticalScroll)
            error_msg = MessageWidget("assistant", f"Error initializing session: {str(e)}")
            await messages_container.mount(error_msg)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission.

        Args:
            event: Input submitted event
        """
        user_input = event.value.strip()
        if not user_input:
            return

        # Clear input
        event.input.value = ""

        messages_container = self.query_one("#messages-container", VerticalScroll)

        # Handle special commands
        if user_input.startswith("/"):
            await self._handle_command(user_input, messages_container)
            return

        # Add user message
        user_msg = MessageWidget("user", user_input)
        await messages_container.mount(user_msg)

        # Scroll to bottom
        messages_container.scroll_end(animate=False)

        # Show thinking indicator
        thinking_msg = MessageWidget("assistant", "Thinking...")
        await messages_container.mount(thinking_msg)
        messages_container.scroll_end(animate=False)

        try:
            # Get response from session
            response = await self.session.ask(user_input)

            # Remove thinking indicator
            await thinking_msg.remove()

            # Add assistant response
            assistant_msg = MessageWidget("assistant", response)
            await messages_container.mount(assistant_msg)

        except Exception as e:
            # Remove thinking indicator
            await thinking_msg.remove()

            # Show error
            error_msg = MessageWidget("assistant", f"Error: {str(e)}")
            await messages_container.mount(error_msg)

        # Scroll to bottom
        messages_container.scroll_end(animate=False)

    async def _handle_command(self, command: str, container: VerticalScroll) -> None:
        """Handle special commands.

        Args:
            command: Command string
            container: Messages container
        """
        if command == "/history":
            # Show code history
            code_history = self.session.list_code_history()
            history_widget = CodeHistoryWidget(code_history)
            await container.mount(history_widget)

        elif command.startswith("/show "):
            # Show specific code snippet
            try:
                index = int(command.split()[1])
                code_history = self.session.list_code_history()

                if 1 <= index <= len(code_history):
                    snippet = code_history[index - 1]
                    snippet_widget = CodeSnippetWidget(index, snippet)
                    await container.mount(snippet_widget)
                else:
                    error_msg = MessageWidget("assistant", f"Invalid index. Valid range: 1-{len(code_history)}")
                    await container.mount(error_msg)

            except (IndexError, ValueError):
                error_msg = MessageWidget("assistant", "Usage: /show N (where N is a number)")
                await container.mount(error_msg)

        else:
            error_msg = MessageWidget("assistant", f"Unknown command: {command}")
            await container.mount(error_msg)

        container.scroll_end(animate=False)
