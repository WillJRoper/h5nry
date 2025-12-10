"""Textual TUI for H5nry."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Footer, Header, Input, Static

from h5nry.app import H5nryApp
from h5nry.config import ConfigManager
from h5nry.session import H5nrySession


def load_css() -> str:
    """Load CSS from data file.

    Returns:
        CSS content as string
    """
    try:
        # Python 3.11+ style
        css_file = importlib.resources.files("h5nry.data").joinpath("tui.css")
        return css_file.read_text()
    except AttributeError:
        # Python 3.10 fallback
        with importlib.resources.open_text("h5nry.data", "tui.css") as f:
            return f.read()


class ThinkingWidget(Static):
    """Widget showing AI thinking/working status with animated spinner."""

    # Braille spinner frames for smooth animation
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    # Reactive properties
    status_message: reactive[str] = reactive("Thinking...")
    frame_index: reactive[int] = reactive(0)
    status_history: reactive[list[str]] = reactive(list)

    def __init__(self, **kwargs):
        """Initialize thinking widget."""
        super().__init__(**kwargs)
        self.status_history = []

    def on_mount(self) -> None:
        """Set up spinner animation timer when mounted."""
        # Update spinner every 0.1 seconds
        self.set_interval(0.1, self._advance_spinner)
        self._update_display()

    def _advance_spinner(self) -> None:
        """Advance the spinner frame."""
        self.frame_index = (self.frame_index + 1) % len(self.SPINNER_FRAMES)

    def watch_status_message(self, message: str) -> None:  # noqa: ARG002
        """React to status message changes."""
        self._update_display()

    def watch_frame_index(self, index: int) -> None:  # noqa: ARG002
        """React to frame index changes."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the displayed text."""
        spinner = self.SPINNER_FRAMES[self.frame_index]

        # Build content showing current status and recent history
        content_parts = []

        # Current status with spinner
        current_status = Text()
        current_status.append(spinner, style="bold cyan")
        current_status.append(" ", style="")
        current_status.append(self.status_message, style="cyan")

        content_parts.append(current_status)

        # Show recent history (last 5 items for more context)
        if self.status_history:
            content_parts.append(Text(""))  # Empty line
            for item in self.status_history[-5:]:
                history_line = Text()
                history_line.append("  → ", style="dim")
                history_line.append(item, style="dim")
                content_parts.append(history_line)

        # Combine all parts
        content = Group(*content_parts)

        panel = Panel(
            content,
            title="H5nry",
            border_style="cyan",
            padding=(0, 1),
            expand=False,  # Don't force expansion
        )

        self.update(panel)

    def update_status(self, message: str, add_to_history: bool = True) -> None:
        """Update the status message.

        Args:
            message: New status message
            add_to_history: Whether to add to history
        """
        if add_to_history and self.status_message:
            self.status_history.append(self.status_message)
        self.status_message = message


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


class H5nryTUI(App):
    """Textual TUI application for H5nry."""

    CSS = load_css()

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, file_path: str | Path, config_manager: ConfigManager):
        """Initialize TUI.

        Args:
            file_path: Path to HDF5 file
            config_manager: Configuration manager
        """
        # Set ansi_color to True for terminal transparency
        # Disable mouse to allow terminal text selection
        super().__init__(ansi_color=True)
        self.file_path = Path(file_path)
        self.config_manager = config_manager
        self.session: H5nrySession | None = None
        self.app_instance = H5nryApp(config_manager)
        self.current_thinking_widget: ThinkingWidget | None = None
        self.input_widget: Input | None = None
        self.messages_container: VerticalScroll | None = None

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header()

        # Info bar (file, provider, model, safety)
        config = self.config_manager.get_config()
        info_text = (
            f"File: {self.file_path.name} | "
            f"Provider: {config.provider} | "
            f"Model: {config.model} | "
            f"Safety: {config.safety_level.value}"
        )
        yield Static(info_text, id="info-bar")

        # Messages container
        with VerticalScroll(id="messages-container"):
            yield Input(
                placeholder="Ask a question or type /history, /exit...",
                id="chat-input",
            )

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize session when app mounts."""
        try:
            self.messages_container = self.query_one(
                "#messages-container", VerticalScroll
            )
            self.input_widget = self.query_one("#chat-input", Input)
            self.session = await self.app_instance.create_session(self.file_path)

            # Add welcome message with ASCII art
            ascii_art = """
 __    __   _______
|\\ \\  |\\ \\ |\\      \\
| HH  | HH | 5555555   _______     ______    __    __
| HH__| HH | 55____   |\\      \\   /\\     \\  |\\ \\  |\\ \\
| HH   \\HH | 55    \\  | NNNNNNN\\ |\\ RRRRRR\\ | YY  | YY
| HHHHHHHH  \\5555555\\ | NN  | NN | RR   \\RR | YY  | YY
| HH  | HH |  \\__| 55 | NN  | NN | RR       | YY__/ YY
| HH  | HH  \\55   \\55 | NN  | NN | RR        \\YY   \\YY
 \\HH   \\HH   \\555555   \\NN   \\NN  \\RR        _\\YYYYYYY
                                            |\\ \\__| YY
                                             \\YY   \\YY
                                              \\YYYYYY


Hello! I'm H5nry, your AI assistant for HDF5 file exploration.

**File:** {self.file_path.name}

I can help you:
• Explore file structure and metadata
• Compute statistics on datasets
• Create visualizations (histograms, scatter plots, line plots)
• Analyze data with custom Python code
• Search for specific datasets or attributes

**Commands:**
• **/history** - See executed code snippets
• **/exit** - Exit the application

Type your question to get started!"""
            welcome = MessageWidget("assistant", ascii_art)
            await self._add_to_conversation(welcome)

            # Focus input
            self.input_widget.focus()

        except Exception as e:
            error_msg = MessageWidget(
                "assistant", f"Error initializing session: {str(e)}"
            )
            await self._add_to_conversation(error_msg)

    def _status_callback(self, phase: str, details: dict[str, Any] | None) -> None:
        """Handle status updates from the session.

        Args:
            phase: Current phase (planning, tool_call, tool_result, tool_progress, answer)
            details: Optional details dict
        """
        if not self.current_thinking_widget:
            return

        # Debug: log callback invocation
        self.log(f"Status callback: phase={phase}, details={details}")

        if phase == "planning":
            self.current_thinking_widget.update_status("Planning tool calls…")

        elif phase == "tool_call":
            tool_name = details.get("name", "unknown") if details else "unknown"
            # Extract a relevant path/dataset for display
            if details and "args" in details:
                args = details["args"]
                path_key = None
                for key in ["dataset_path", "path", "x_dataset_path", "y_dataset_path"]:
                    if key in args:
                        path_key = args[key]
                        break
                if path_key:
                    self.current_thinking_widget.update_status(
                        f"Calling {tool_name} on {path_key}…"
                    )
                else:
                    self.current_thinking_widget.update_status(f"Calling {tool_name}…")
            else:
                self.current_thinking_widget.update_status(f"Calling {tool_name}…")

        elif phase == "tool_result":
            tool_name = details.get("name", "unknown") if details else "unknown"
            self.current_thinking_widget.update_status(
                f"Received result from {tool_name}"
            )

        elif phase == "tool_progress":
            # Finer-grained progress from within a tool
            tool_name = details.get("name", "unknown") if details else "unknown"
            tool_phase = details.get("phase", "working") if details else "working"
            self.current_thinking_widget.update_status(
                f"Calling {tool_name} ({tool_phase}…)"
            )

        elif phase == "answer":
            self.current_thinking_widget.update_status("Generating final answer…")

        else:
            self.current_thinking_widget.update_status("Working…")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission.

        Args:
            event: Input submitted event
        """
        user_input = event.value.strip()
        self.log(f"Input submitted: '{user_input}'")

        if not user_input:
            return

        # Clear input
        event.input.value = ""

        # Handle special commands
        if user_input.startswith("/"):
            self.log(f"Handling command: {user_input}")
            await self._handle_command(user_input)
            return

        # Add user message
        self.log("Adding user message widget")
        user_msg = MessageWidget("user", user_input)
        await self._add_to_conversation(user_msg)

        # Add thinking widget
        self.log("Creating thinking widget")
        thinking_widget = ThinkingWidget()
        self.current_thinking_widget = thinking_widget
        await self._add_to_conversation(thinking_widget)

        # Scroll to bottom
        if self.messages_container:
            self.messages_container.scroll_end(animate=False)

        try:
            # Get response from session with status callback
            self.log("Calling session.ask()")
            response = await self.session.ask(
                user_input, status_callback=self._status_callback
            )
            self.log(f"Got response: {response[:100]}...")

            # Remove thinking widget
            await thinking_widget.remove()
            self.current_thinking_widget = None

            # Add assistant response
            assistant_msg = MessageWidget("assistant", response)
            await self._add_to_conversation(assistant_msg)

        except Exception as e:
            # Remove thinking widget
            if self.current_thinking_widget:
                await self.current_thinking_widget.remove()
                self.current_thinking_widget = None

            # Show error
            error_msg = MessageWidget("assistant", f"Error: {str(e)}")
            await self._add_to_conversation(error_msg)

        # Scroll to bottom
        if self.messages_container:
            self.messages_container.scroll_end(animate=False)

    async def _handle_command(self, command: str) -> None:
        """Handle special commands.

        Args:
            command: Command string
        """
        if command == "/exit":
            # Exit the application
            self.exit()

        elif command == "/history":
            # Show code history
            code_history = self.session.list_code_history()
            history_widget = CodeHistoryWidget(code_history)
            await self._add_to_conversation(history_widget)

        else:
            error_msg = MessageWidget("assistant", f"Unknown command: {command}")
            await self._add_to_conversation(error_msg)

        # Only scroll if we didn't exit
        if command != "/exit" and self.messages_container:
            self.messages_container.scroll_end(animate=False)

    async def _add_to_conversation(self, widget: Widget) -> None:
        """Insert a widget above the input so the prompt flows with history."""
        if not self.messages_container or not self.input_widget:
            return
        await self.messages_container.mount(widget, before=self.input_widget)
        self.messages_container.scroll_end(animate=False)
