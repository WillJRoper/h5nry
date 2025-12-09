"""Tools for HDF5 file investigation."""

from __future__ import annotations

from h5nry.tools import hdf5_tree, plotting, python_exec, stats

__all__ = ["hdf5_tree", "stats", "plotting", "python_exec"]


def get_all_tools(safety_level: str = "tools_only") -> list[dict]:
    """Get all available tools based on safety level.

    Args:
        safety_level: Safety level (tools_only or tools_plus_python)

    Returns:
        List of tool definitions for LLM
    """
    tools = []

    # Always include HDF5 and analysis tools
    tools.extend(hdf5_tree.TOOLS)
    tools.extend(stats.TOOLS)
    tools.extend(plotting.TOOLS)

    # Conditionally include Python execution
    if safety_level == "tools_plus_python":
        tools.extend(python_exec.TOOLS)

    return tools
