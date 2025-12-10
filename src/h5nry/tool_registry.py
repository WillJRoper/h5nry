"""Central registry for all LLM tool definitions.

This module is the canonical source of truth for tool schemas passed to the LLM.
The system prompt describes behavior; this module defines the actual function signatures.

Tool schemas are loaded from tool_schemas.json, which is the single source of truth.
"""

from __future__ import annotations

import json
from importlib import resources as importlib_resources
from typing import Any

from h5nry.config import AppConfig


def _load_tool_schemas() -> dict[str, Any]:
    """Load tool schemas from JSON file.

    Returns:
        Dictionary containing base_tools and python_execution_tool
    """
    try:
        # Python 3.11+ style
        schema_file = importlib_resources.files("h5nry.data").joinpath(
            "tool_schemas.json"
        )
        return json.loads(schema_file.read_text())
    except AttributeError:
        # Python 3.10 fallback
        with importlib_resources.open_text("h5nry.data", "tool_schemas.json") as f:
            return json.load(f)


def get_tool_schemas(config: AppConfig) -> list[dict[str, Any]]:
    """Return the list of tool schemas to pass to the LLM.

    This function is the single source of truth for what tools are available
    to the LLM. Tool schemas are loaded from tool_schemas.json in OpenAI
    function-calling format (which can be mapped to Anthropic/Gemini).

    Args:
        config: Application configuration

    Returns:
        List of tool schemas in OpenAI format
    """
    # Load schemas from JSON file (canonical source of truth)
    schemas = _load_tool_schemas()

    # Start with base tools (always included)
    tools = schemas["base_tools"].copy()

    # Conditionally add Python execution tool based on safety level
    if config.safety_level.value == "tools_plus_python":
        tools.append(schemas["python_execution_tool"])

    return tools
