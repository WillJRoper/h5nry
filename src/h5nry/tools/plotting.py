"""Plotting tools with memory-safe chunking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

from h5nry.tools.stats import dataset_histogram

# Use non-interactive backend
matplotlib.use("Agg")


def plot_histogram(
    file_path: str | Path,
    dataset_path: str,
    output_dir: Path,
    max_bytes: int,
    bins: int = 50,
    log_scale: bool = False,
    range_min: float | None = None,
    range_max: float | None = None,
) -> dict[str, Any]:
    """Create a histogram plot for a dataset.

    Args:
        file_path: Path to HDF5 file
        dataset_path: Path to dataset within file
        output_dir: Directory to save plot
        max_bytes: Maximum bytes to load at once
        bins: Number of histogram bins
        log_scale: Use logarithmic scale
        range_min: Minimum value for range
        range_max: Maximum value for range

    Returns:
        Dictionary with plot information
    """
    # Compute histogram using chunked reading
    hist_data = dataset_histogram(
        file_path=file_path,
        dataset_path=dataset_path,
        max_bytes=max_bytes,
        bins=bins,
        range_min=range_min,
        range_max=range_max,
        log_scale=log_scale,
    )

    if "error" in hist_data:
        return hist_data

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bin_edges = hist_data["bin_edges"]
    counts = hist_data["counts"]

    # Plot histogram
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    ax.bar(bin_centers, counts, width=[(bin_edges[i + 1] - bin_edges[i]) for i in range(len(bin_edges) - 1)], edgecolor="black", alpha=0.7)

    # Formatting
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_title(f"Histogram of {dataset_path}")
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_xscale("log")

    # Save plot
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create safe filename from dataset path
    safe_name = dataset_path.replace("/", "_").strip("_")
    output_file = output_dir / f"{safe_name}_histogram.png"

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "dataset_path": dataset_path,
        "plot_file": str(output_file),
        "bins": bins,
        "log_scale": log_scale,
        "total_count": hist_data["total_count"],
    }


# Tool definitions for LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "plot_histogram",
            "description": "Create a histogram plot for a dataset and save it to disk. Automatically chunks large datasets to respect memory limits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset in the HDF5 file",
                    },
                    "bins": {
                        "type": "integer",
                        "description": "Number of histogram bins (default: 50)",
                        "default": 50,
                    },
                    "log_scale": {
                        "type": "boolean",
                        "description": "Use logarithmic scale (default: false)",
                        "default": False,
                    },
                    "range_min": {
                        "type": "number",
                        "description": "Minimum value for histogram range (optional)",
                    },
                    "range_max": {
                        "type": "number",
                        "description": "Maximum value for histogram range (optional)",
                    },
                },
                "required": ["dataset_path"],
            },
        },
    },
]
