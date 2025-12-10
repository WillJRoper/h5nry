"""Plotting tools with memory-safe chunking."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from h5nry.tools.stats import dataset_histogram


def _check_log_scale_needed(
    data: np.ndarray, axis_name: str, log_enabled: bool
) -> dict[str, Any]:
    """Check if log scale should be used for data based on its range.

    Args:
        data: Data array to check
        axis_name: Name of axis (for messages)
        log_enabled: Whether log scale is currently enabled

    Returns:
        Dictionary with recommendation and ratio info
    """
    # Remove zeros and negatives for ratio calculation
    positive_data = data[data > 0]

    if len(positive_data) == 0:
        return {
            "recommendation": "linear",
            "reason": f"{axis_name}: No positive values (log scale not possible)",
            "ratio": None,
        }

    data_min = float(np.min(positive_data))
    data_max = float(np.max(positive_data))
    ratio = data_max / data_min if data_min > 0 else None

    if ratio is None or ratio < 100:
        recommendation = "linear"
        reason = f"{axis_name}: Range ratio {ratio:.1f}x (< 100x, linear OK)"
    elif ratio < 1000:
        recommendation = "log_suggested"
        reason = (
            f"{axis_name}: Range ratio {ratio:.1f}x (100-1000x, log scale recommended)"
        )
    else:
        recommendation = "log_required"
        reason = f"{axis_name}: Range ratio {ratio:.1f}x (> 1000x, log scale strongly recommended)"

    return {
        "recommendation": recommendation,
        "reason": reason,
        "ratio": ratio,
        "should_use_log": recommendation in ["log_suggested", "log_required"],
        "currently_log": log_enabled,
    }


def _open_file(file_path: Path) -> None:
    """Open a file with the default system application.

    Args:
        file_path: Path to the file to open
    """
    if sys.platform == "darwin":  # macOS
        subprocess.run(["open", str(file_path)], check=False)
    elif sys.platform == "win32":  # Windows
        subprocess.run(["start", str(file_path)], shell=True, check=False)
    else:  # Linux and other Unix-like systems
        subprocess.run(["xdg-open", str(file_path)], check=False)


def _load_dataset_safely(
    file_path: str | Path,
    dataset_path: str,
    max_bytes: int,
) -> dict[str, Any]:
    """Safely load a dataset with automatic downsampling for large datasets.

    Args:
        file_path: Path to HDF5 file
        dataset_path: Path to dataset within file
        max_bytes: Maximum bytes to load

    Returns:
        Dictionary with data, info about downsampling, or error
    """
    import h5py
    import numpy as np

    try:
        with h5py.File(file_path, "r") as f:
            if dataset_path not in f:
                return {"error": f"Dataset not found: {dataset_path}"}

            dataset = f[dataset_path]
            size_bytes = dataset.size * dataset.dtype.itemsize
            downsampled = False
            stride = 1

            if size_bytes > max_bytes:
                # Calculate stride to downsample data to fit in memory
                # Target using about 80% of max_bytes to leave some headroom
                target_bytes = int(max_bytes * 0.8)
                stride = max(1, int(np.ceil(size_bytes / target_bytes)))

                # Load every Nth element to fit in memory
                if len(dataset.shape) == 1:
                    data = dataset[::stride]
                else:
                    # For multidimensional, stride along first dimension
                    data = dataset[::stride].flatten()

                downsampled = True
                actual_bytes = data.size * data.dtype.itemsize
            else:
                data = (
                    dataset[...].flatten() if len(dataset.shape) > 1 else dataset[...]
                )
                actual_bytes = size_bytes

            return {
                "data": data,
                "shape": dataset.shape,
                "dtype": str(dataset.dtype),
                "downsampled": downsampled,
                "stride": stride if downsampled else 1,
                "original_size": dataset.size,
                "loaded_size": data.size,
                "size_mb": actual_bytes / (1024**2),
            }

    except Exception as e:
        return {"error": f"Failed to load dataset: {e}"}


def plot_histogram(
    file_path: str | Path,
    dataset_path: str,
    max_bytes: int,
    bins: int = 50,
    log_scale: bool = False,
    range_min: float | None = None,
    range_max: float | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Create and save a histogram plot for a dataset.

    Args:
        file_path: Path to HDF5 file
        dataset_path: Path to dataset within file
        max_bytes: Maximum bytes to load at once
        bins: Number of histogram bins
        log_scale: Use logarithmic scale
        range_min: Minimum value for range
        range_max: Maximum value for range
        title: Custom plot title (optional)

    Returns:
        Dictionary with plot information including the saved file path
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
    bin_centers = [
        (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)
    ]
    ax.bar(
        bin_centers,
        counts,
        width=[(bin_edges[i + 1] - bin_edges[i]) for i in range(len(bin_edges) - 1)],
        edgecolor="black",
        alpha=0.7,
    )

    # Formatting
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_title(title if title else f"Histogram of {dataset_path}")
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_xscale("log")

    # Save plot to current working directory
    # Create safe filename from dataset path
    safe_name = dataset_path.replace("/", "_").strip("_")
    output_file = Path.cwd() / f"{safe_name}_histogram.png"

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Open the plot in the default image viewer
    _open_file(output_file)

    return {
        "success": True,
        "message": f"Histogram saved to: {output_file} and opened in default viewer",
        "dataset_path": dataset_path,
        "plot_file": str(output_file),
        "absolute_path": str(output_file.absolute()),
        "bins": bins,
        "log_scale": log_scale,
        "total_count": hist_data["total_count"],
    }


def plot_scatter(
    file_path: str | Path,
    x_dataset_path: str,
    y_dataset_path: str,
    max_bytes: int,
    log_x: bool = False,
    log_y: bool = False,
    alpha: float = 0.5,
    xlim_min: float | None = None,
    xlim_max: float | None = None,
    ylim_min: float | None = None,
    ylim_max: float | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Create and save a scatter plot of two datasets.

    Args:
        file_path: Path to HDF5 file
        x_dataset_path: Path to dataset for x-axis
        y_dataset_path: Path to dataset for y-axis
        max_bytes: Maximum bytes to load at once
        log_x: Use logarithmic scale for x-axis
        log_y: Use logarithmic scale for y-axis
        alpha: Point transparency (0-1)
        xlim_min: Minimum value for x-axis (optional)
        xlim_max: Maximum value for x-axis (optional)
        ylim_min: Minimum value for y-axis (optional)
        ylim_max: Maximum value for y-axis (optional)
        title: Custom plot title (optional)

    Returns:
        Dictionary with plot information
    """

    # Load x data
    x_result = _load_dataset_safely(file_path, x_dataset_path, max_bytes // 2)
    if "error" in x_result:
        return x_result
    x_data = x_result["data"].flatten()

    # Load y data
    y_result = _load_dataset_safely(file_path, y_dataset_path, max_bytes // 2)
    if "error" in y_result:
        return y_result
    y_data = y_result["data"].flatten()

    # Check compatible sizes
    if len(x_data) != len(y_data):
        return {"error": f"Dataset sizes don't match: {len(x_data)} vs {len(y_data)}"}

    # Check if log scales are appropriate
    x_scale_check = _check_log_scale_needed(x_data, "X-axis", log_x)
    y_scale_check = _check_log_scale_needed(y_data, "Y-axis", log_y)

    # Build scaling warnings
    warnings = []
    if x_scale_check["should_use_log"] and not log_x:
        warnings.append(f"⚠️  {x_scale_check['reason']} - Consider setting log_x=true")
    if y_scale_check["should_use_log"] and not log_y:
        warnings.append(f"⚠️  {y_scale_check['reason']} - Consider setting log_y=true")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(x_data, y_data, alpha=alpha, s=20)

    # Formatting
    ax.set_xlabel(x_dataset_path.split("/")[-1])
    ax.set_ylabel(y_dataset_path.split("/")[-1])

    plot_title = title
    if not plot_title:
        plot_title = (
            f"{y_dataset_path.split('/')[-1]} vs {x_dataset_path.split('/')[-1]}"
        )
        if x_result["downsampled"] or y_result["downsampled"]:
            max_stride = max(x_result["stride"], y_result["stride"])
            plot_title += f" (downsampled 1:{max_stride})"
    ax.set_title(plot_title)

    ax.grid(True, alpha=0.3)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    # Apply axis limits if specified
    if xlim_min is not None or xlim_max is not None:
        ax.set_xlim(xlim_min, xlim_max)
    if ylim_min is not None or ylim_max is not None:
        ax.set_ylim(ylim_min, ylim_max)

    # Save plot
    safe_x = x_dataset_path.replace("/", "_").strip("_")
    safe_y = y_dataset_path.replace("/", "_").strip("_")
    output_file = Path.cwd() / f"{safe_y}_vs_{safe_x}_scatter.png"

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Open the plot
    _open_file(output_file)

    # Build message
    message = f"Scatter plot saved to: {output_file} and opened in default viewer"
    if x_result["downsampled"] or y_result["downsampled"]:
        message += f" (plotted {len(x_data):,} of {x_result['original_size']:,} points)"

    result = {
        "success": True,
        "message": message,
        "x_dataset": x_dataset_path,
        "y_dataset": y_dataset_path,
        "plot_file": str(output_file),
        "absolute_path": str(output_file.absolute()),
        "num_points": len(x_data),
        "downsampled": x_result["downsampled"] or y_result["downsampled"],
        "stride": max(x_result["stride"], y_result["stride"]),
        "x_range_ratio": x_scale_check["ratio"],
        "y_range_ratio": y_scale_check["ratio"],
    }

    if warnings:
        result["scaling_warnings"] = warnings
        result["message"] += "\n\n" + "\n".join(warnings)

    return result


def plot_line(
    file_path: str | Path,
    x_dataset_path: str,
    y_dataset_path: str,
    max_bytes: int,
    log_x: bool = False,
    log_y: bool = False,
    xlim_min: float | None = None,
    xlim_max: float | None = None,
    ylim_min: float | None = None,
    ylim_max: float | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Create and save a line plot of two datasets.

    Args:
        file_path: Path to HDF5 file
        x_dataset_path: Path to dataset for x-axis
        y_dataset_path: Path to dataset for y-axis
        max_bytes: Maximum bytes to load at once
        log_x: Use logarithmic scale for x-axis
        log_y: Use logarithmic scale for y-axis
        xlim_min: Minimum value for x-axis (optional)
        xlim_max: Maximum value for x-axis (optional)
        ylim_min: Minimum value for y-axis (optional)
        ylim_max: Maximum value for y-axis (optional)
        title: Custom plot title (optional)

    Returns:
        Dictionary with plot information
    """

    # Load x data
    x_result = _load_dataset_safely(file_path, x_dataset_path, max_bytes // 2)
    if "error" in x_result:
        return x_result
    x_data = x_result["data"].flatten()

    # Load y data
    y_result = _load_dataset_safely(file_path, y_dataset_path, max_bytes // 2)
    if "error" in y_result:
        return y_result
    y_data = y_result["data"].flatten()

    # Check compatible sizes
    if len(x_data) != len(y_data):
        return {"error": f"Dataset sizes don't match: {len(x_data)} vs {len(y_data)}"}

    # Check if log scales are appropriate
    x_scale_check = _check_log_scale_needed(x_data, "X-axis", log_x)
    y_scale_check = _check_log_scale_needed(y_data, "Y-axis", log_y)

    # Build scaling warnings
    warnings = []
    if x_scale_check["should_use_log"] and not log_x:
        warnings.append(f"⚠️  {x_scale_check['reason']} - Consider setting log_x=true")
    if y_scale_check["should_use_log"] and not log_y:
        warnings.append(f"⚠️  {y_scale_check['reason']} - Consider setting log_y=true")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_data, y_data, linewidth=1.5)

    # Formatting
    ax.set_xlabel(x_dataset_path.split("/")[-1])
    ax.set_ylabel(y_dataset_path.split("/")[-1])

    plot_title = title
    if not plot_title:
        plot_title = (
            f"{y_dataset_path.split('/')[-1]} vs {x_dataset_path.split('/')[-1]}"
        )
        if x_result["downsampled"] or y_result["downsampled"]:
            max_stride = max(x_result["stride"], y_result["stride"])
            plot_title += f" (downsampled 1:{max_stride})"
    ax.set_title(plot_title)

    ax.grid(True, alpha=0.3)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    # Apply axis limits if specified
    if xlim_min is not None or xlim_max is not None:
        ax.set_xlim(xlim_min, xlim_max)
    if ylim_min is not None or ylim_max is not None:
        ax.set_ylim(ylim_min, ylim_max)

    # Save plot
    safe_x = x_dataset_path.replace("/", "_").strip("_")
    safe_y = y_dataset_path.replace("/", "_").strip("_")
    output_file = Path.cwd() / f"{safe_y}_vs_{safe_x}_line.png"

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Open the plot
    _open_file(output_file)

    # Build message
    message = f"Line plot saved to: {output_file} and opened in default viewer"
    if x_result["downsampled"] or y_result["downsampled"]:
        message += f" (plotted {len(x_data):,} of {x_result['original_size']:,} points)"

    result = {
        "success": True,
        "message": message,
        "x_dataset": x_dataset_path,
        "y_dataset": y_dataset_path,
        "plot_file": str(output_file),
        "absolute_path": str(output_file.absolute()),
        "num_points": len(x_data),
        "downsampled": x_result["downsampled"] or y_result["downsampled"],
        "stride": max(x_result["stride"], y_result["stride"]),
        "x_range_ratio": x_scale_check["ratio"],
        "y_range_ratio": y_scale_check["ratio"],
    }

    if warnings:
        result["scaling_warnings"] = warnings
        result["message"] += "\n\n" + "\n".join(warnings)

    return result


def plot_hexbin(
    file_path: str | Path,
    x_dataset_path: str,
    y_dataset_path: str,
    max_bytes: int,
    gridsize: int = 50,
    log_x: bool = False,
    log_y: bool = False,
    log_color: bool = False,
    xlim_min: float | None = None,
    xlim_max: float | None = None,
    ylim_min: float | None = None,
    ylim_max: float | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Create and save a hexbin plot (2D histogram) of two datasets.

    Args:
        file_path: Path to HDF5 file
        x_dataset_path: Path to dataset for x-axis
        y_dataset_path: Path to dataset for y-axis
        max_bytes: Maximum bytes to load at once
        gridsize: Number of hexagons in x-direction
        log_x: Use logarithmic scale for x-axis
        log_y: Use logarithmic scale for y-axis
        log_color: Use logarithmic scale for colormap
        xlim_min: Minimum value for x-axis (optional)
        xlim_max: Maximum value for x-axis (optional)
        ylim_min: Minimum value for y-axis (optional)
        ylim_max: Maximum value for y-axis (optional)
        title: Custom plot title (optional)

    Returns:
        Dictionary with plot information
    """

    # Load x data
    x_result = _load_dataset_safely(file_path, x_dataset_path, max_bytes // 2)
    if "error" in x_result:
        return x_result
    x_data = x_result["data"].flatten()

    # Load y data
    y_result = _load_dataset_safely(file_path, y_dataset_path, max_bytes // 2)
    if "error" in y_result:
        return y_result
    y_data = y_result["data"].flatten()

    # Check compatible sizes
    if len(x_data) != len(y_data):
        return {"error": f"Dataset sizes don't match: {len(x_data)} vs {len(y_data)}"}

    # Check if log scales are appropriate
    x_scale_check = _check_log_scale_needed(x_data, "X-axis", log_x)
    y_scale_check = _check_log_scale_needed(y_data, "Y-axis", log_y)

    # Build scaling warnings
    warnings = []
    if x_scale_check["should_use_log"] and not log_x:
        warnings.append(f"⚠️  {x_scale_check['reason']} - Consider setting log_x=true")
    if y_scale_check["should_use_log"] and not log_y:
        warnings.append(f"⚠️  {y_scale_check['reason']} - Consider setting log_y=true")

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Handle log scales (hexbin requires positive values for log scales)
    xscale = "log" if log_x else "linear"
    yscale = "log" if log_y else "linear"

    hexbin = ax.hexbin(
        x_data,
        y_data,
        gridsize=gridsize,
        cmap="viridis",
        mincnt=1,
        xscale=xscale,
        yscale=yscale,
        bins="log" if log_color else None,
    )

    # Formatting
    ax.set_xlabel(x_dataset_path.split("/")[-1])
    ax.set_ylabel(y_dataset_path.split("/")[-1])

    plot_title = title
    if not plot_title:
        plot_title = f"{y_dataset_path.split('/')[-1]} vs {x_dataset_path.split('/')[-1]} (Hexbin)"
        if x_result["downsampled"] or y_result["downsampled"]:
            max_stride = max(x_result["stride"], y_result["stride"])
            plot_title += f" (downsampled 1:{max_stride})"
    ax.set_title(plot_title)

    # Apply axis limits if specified
    if xlim_min is not None or xlim_max is not None:
        ax.set_xlim(xlim_min, xlim_max)
    if ylim_min is not None or ylim_max is not None:
        ax.set_ylim(ylim_min, ylim_max)

    # Add colorbar
    plt.colorbar(hexbin, ax=ax, label="Count")

    # Save plot
    safe_x = x_dataset_path.replace("/", "_").strip("_")
    safe_y = y_dataset_path.replace("/", "_").strip("_")
    output_file = Path.cwd() / f"{safe_y}_vs_{safe_x}_hexbin.png"

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Open the plot
    _open_file(output_file)

    # Build message
    message = f"Hexbin plot saved to: {output_file} and opened in default viewer"
    if x_result["downsampled"] or y_result["downsampled"]:
        message += f" (plotted {len(x_data):,} of {x_result['original_size']:,} points)"

    result = {
        "success": True,
        "message": message,
        "x_dataset": x_dataset_path,
        "y_dataset": y_dataset_path,
        "plot_file": str(output_file),
        "absolute_path": str(output_file.absolute()),
        "num_points": len(x_data),
        "downsampled": x_result["downsampled"] or y_result["downsampled"],
        "stride": max(x_result["stride"], y_result["stride"]),
        "x_range_ratio": x_scale_check["ratio"],
        "y_range_ratio": y_scale_check["ratio"],
    }

    if warnings:
        result["scaling_warnings"] = warnings
        result["message"] += "\n\n" + "\n".join(warnings)

    return result


# Tool definitions for LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "plot_histogram",
            "description": """Create a histogram showing the distribution of values in a single dataset.

**Use when:** Exploring the distribution of a single variable (e.g., "show me the distribution of temperatures").

**After plotting:** View the image with the Read tool, analyze quality, and refine if needed.""",
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
                        "description": "Use logarithmic scale for x-axis (default: false)",
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
    {
        "type": "function",
        "function": {
            "name": "plot_scatter",
            "description": """Create a scatter plot showing the relationship between two datasets. Automatically downsamples large datasets to fit memory.

**Use when:** Comparing two variables to see correlations or patterns (e.g., "plot temperature vs density").

**After plotting:** View the image with the Read tool, analyze quality, and refine if needed.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_dataset_path": {
                        "type": "string",
                        "description": "Path to dataset for x-axis",
                    },
                    "y_dataset_path": {
                        "type": "string",
                        "description": "Path to dataset for y-axis",
                    },
                    "log_x": {
                        "type": "boolean",
                        "description": "Use logarithmic scale for x-axis (default: false)",
                        "default": False,
                    },
                    "log_y": {
                        "type": "boolean",
                        "description": "Use logarithmic scale for y-axis (default: false)",
                        "default": False,
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Point transparency 0-1 (default: 0.5)",
                        "default": 0.5,
                    },
                },
                "required": ["x_dataset_path", "y_dataset_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_line",
            "description": """Create a line plot connecting points between two datasets. Automatically downsamples large datasets to fit memory.

**Use when:** Showing trends or sequential data (e.g., "plot radius vs time", time series, profiles).

**After plotting:** View the image with the Read tool, analyze quality, and refine if needed.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_dataset_path": {
                        "type": "string",
                        "description": "Path to dataset for x-axis",
                    },
                    "y_dataset_path": {
                        "type": "string",
                        "description": "Path to dataset for y-axis",
                    },
                    "log_x": {
                        "type": "boolean",
                        "description": "Use logarithmic scale for x-axis (default: false)",
                        "default": False,
                    },
                    "log_y": {
                        "type": "boolean",
                        "description": "Use logarithmic scale for y-axis (default: false)",
                        "default": False,
                    },
                },
                "required": ["x_dataset_path", "y_dataset_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_hexbin",
            "description": """Create a hexbin plot (2D histogram) for datasets with many points. Automatically downsamples large datasets to fit memory.

**Use when:** You have large datasets (>10k points) and want to see density patterns. Better than scatter for crowded data.

**After plotting:** View the image with the Read tool, analyze quality, and refine if needed.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "x_dataset_path": {
                        "type": "string",
                        "description": "Path to dataset for x-axis",
                    },
                    "y_dataset_path": {
                        "type": "string",
                        "description": "Path to dataset for y-axis",
                    },
                    "gridsize": {
                        "type": "integer",
                        "description": "Number of hexagons in x-direction (default: 50)",
                        "default": 50,
                    },
                    "log_x": {
                        "type": "boolean",
                        "description": "Use logarithmic scale for x-axis (default: false)",
                        "default": False,
                    },
                    "log_y": {
                        "type": "boolean",
                        "description": "Use logarithmic scale for y-axis (default: false)",
                        "default": False,
                    },
                    "log_color": {
                        "type": "boolean",
                        "description": "Use logarithmic scale for color/count (default: false)",
                        "default": False,
                    },
                },
                "required": ["x_dataset_path", "y_dataset_path"],
            },
        },
    },
]
