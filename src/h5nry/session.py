"""H5nry session management."""

from __future__ import annotations

import importlib.resources
import json
from collections import deque
from pathlib import Path
from typing import Any

from h5nry.config import AppConfig
from h5nry.llm.base import LLMClient, Message
from h5nry.tools import get_all_tools
from h5nry.tools.hdf5_tree import (
    H5Node,
    get_node_info,
    list_children,
    summarize_tree,
)
from h5nry.tools.plotting import plot_hexbin, plot_histogram, plot_line, plot_scatter
from h5nry.tools.python_exec import run_python
from h5nry.tools.stats import dataset_histogram, dataset_stats


def load_system_prompt_template() -> str:
    """Load system prompt template from data file.

    Returns:
        System prompt template as string
    """
    try:
        # Python 3.11+ style
        prompt_file = importlib.resources.files("h5nry.data").joinpath(
            "system_prompt.txt"
        )
        return prompt_file.read_text()
    except AttributeError:
        # Python 3.10 fallback
        with importlib.resources.open_text("h5nry.data", "system_prompt.txt") as f:
            return f.read()


class H5nrySession:
    """Manages a conversation session with HDF5 file context."""

    def __init__(
        self,
        llm_client: LLMClient,
        config: AppConfig,
        file_path: str | Path,
        tree: H5Node,
    ):
        """Initialize session.

        Args:
            llm_client: LLM client to use
            config: Application configuration
            file_path: Path to HDF5 file
            tree: Parsed HDF5 tree
        """
        self.llm_client = llm_client
        self.config = config
        self.file_path = Path(file_path)
        self.tree = tree
        self.messages: list[Message] = []
        self.code_history: deque[str] = deque(maxlen=config.recent_code_limit)

        # Initialize with system prompt
        self._initialize_system_prompt()

    def _initialize_system_prompt(self) -> None:
        """Initialize the system prompt with file context."""
        # Generate file summary
        tree_summary = summarize_tree(self.tree, max_depth=3, max_children=20)

        # Load prompt template
        template = load_system_prompt_template()

        # Build Python section if enabled
        python_section = ""
        if self.config.safety_level.value == "tools_plus_python":
            python_section = """
4. **Python Execution (REPL)**:
   - `run_python(code)`: Execute Python code to investigate the file directly
   - **The HDF5 file is already open** as `f` or `h5_file` (read-only)
   - Pre-imported: `numpy as np`, `h5py`, `math`
   - Example: `run_python("print(list(f.keys()))")` to list all groups/datasets
   - Example: `run_python("data = f['dataset_path'][:1000]; print(np.mean(data))")` to read and analyze
   - The file automatically opens/closes for each execution
   - If you try to read too much data at once, you'll get a clear error with suggestions

   **When to use Python execution:**
   - When you need to explore the file structure dynamically (list keys, navigate groups)
   - When you need to read actual dataset values that aren't covered by the stats tools
   - When you need custom calculations or data transformations
   - When the user asks to "run code" or wants specific computations

   **Important:** This gives you full programmatic access to the HDF5 file - use it liberally!
"""

        # Fill in template
        system_prompt = template.format(
            file_name=self.file_path.name,
            tree_summary=tree_summary,
            python_section=python_section,
            max_data_gb=self.config.max_data_gb,
        )

        self.messages.append(Message(role="system", content=system_prompt))

    def _get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions based on safety level.

        Returns:
            List of tool definitions
        """
        return get_all_tools(self.config.safety_level.value)

    def _execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            JSON string with tool result
        """
        max_bytes = int(self.config.max_data_gb * 1024 * 1024 * 1024)

        try:
            # HDF5 Tree Tools
            if tool_name == "get_node_info":
                result = get_node_info(self.tree, arguments["path"])
            elif tool_name == "list_children":
                result = list_children(self.tree, arguments.get("path", "/"))

            # Statistics Tools
            elif tool_name == "dataset_stats":
                result = dataset_stats(
                    self.file_path,
                    arguments["dataset_path"],
                    max_bytes,
                    arguments.get("axis"),
                )
            elif tool_name == "dataset_histogram":
                result = dataset_histogram(
                    self.file_path,
                    arguments["dataset_path"],
                    max_bytes,
                    arguments.get("bins", 50),
                    arguments.get("range_min"),
                    arguments.get("range_max"),
                    arguments.get("log_scale", False),
                )

            # Plotting Tools
            elif tool_name == "plot_histogram":
                result = plot_histogram(
                    self.file_path,
                    arguments["dataset_path"],
                    max_bytes,
                    arguments.get("bins", 50),
                    arguments.get("log_scale", False),
                    arguments.get("range_min"),
                    arguments.get("range_max"),
                )
            elif tool_name == "plot_scatter":
                result = plot_scatter(
                    self.file_path,
                    arguments["x_dataset_path"],
                    arguments["y_dataset_path"],
                    max_bytes,
                    arguments.get("log_x", False),
                    arguments.get("log_y", False),
                    arguments.get("alpha", 0.5),
                )
            elif tool_name == "plot_line":
                result = plot_line(
                    self.file_path,
                    arguments["x_dataset_path"],
                    arguments["y_dataset_path"],
                    max_bytes,
                    arguments.get("log_x", False),
                    arguments.get("log_y", False),
                )
            elif tool_name == "plot_hexbin":
                result = plot_hexbin(
                    self.file_path,
                    arguments["x_dataset_path"],
                    arguments["y_dataset_path"],
                    max_bytes,
                    arguments.get("gridsize", 50),
                    arguments.get("log_x", False),
                    arguments.get("log_y", False),
                    arguments.get("log_color", False),
                )

            # Python Execution
            elif tool_name == "run_python":
                if self.config.safety_level.value != "tools_plus_python":
                    result = {
                        "error": "Python execution is disabled. Set safety_level to 'tools_plus_python' to enable."
                    }
                else:
                    result = run_python(
                        arguments["code"],
                        self.file_path,
                        max_bytes,
                    )
                    # Record code in history
                    if "error" not in result:
                        self.record_code_snippet(arguments["code"])

            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {str(e)}"}, indent=2)

    def record_code_snippet(self, code: str) -> None:
        """Record a code snippet in history.

        Args:
            code: Code to record
        """
        self.code_history.append(code)

    def list_code_history(self) -> list[str]:
        """Get list of recorded code snippets.

        Returns:
            List of code snippets
        """
        return list(self.code_history)

    async def ask(self, user_text: str) -> str:
        """Process a user question and return the answer.

        Args:
            user_text: User's question

        Returns:
            Assistant's response
        """
        # Add user message
        self.messages.append(Message(role="user", content=user_text))

        # Get tool definitions
        tools = self._get_tool_definitions()

        # Loop to handle tool calls
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Call LLM
            response = await self.llm_client.chat(self.messages, tools)

            # If there are tool calls, execute them
            if response.tool_calls:
                # Add assistant message with tool calls
                assistant_msg = Message(
                    role="assistant",
                    content=response.content or "",
                )
                # Store tool calls in OpenAI format for compatibility
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                self.messages.append(assistant_msg)

                # Execute each tool call
                for tool_call in response.tool_calls:
                    result = self._execute_tool(tool_call.name, tool_call.arguments)

                    # Add tool result message
                    tool_msg = Message(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    )
                    self.messages.append(tool_msg)

                # Continue loop to get next response
                continue

            # No tool calls - we have a final answer
            if response.content:
                # Add assistant message
                self.messages.append(
                    Message(role="assistant", content=response.content)
                )
                return response.content
            else:
                # Shouldn't happen, but handle gracefully
                return "I'm sorry, I couldn't generate a response."

        # Max iterations reached
        return "I apologize, but I've reached the maximum number of tool calls. Please try rephrasing your question."
