"""Statistical analysis tools with memory-safe chunking."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def compute_chunk_size(
    total_elements: int,
    element_size_bytes: int,
    max_bytes: int,
) -> int:
    """Compute safe chunk size for reading.

    Args:
        total_elements: Total number of elements
        element_size_bytes: Size of each element in bytes
        max_bytes: Maximum bytes to read at once

    Returns:
        Number of elements to read per chunk
    """
    max_elements = max_bytes // element_size_bytes
    # Ensure at least 1 element per chunk
    return max(1, min(max_elements, total_elements))


def dataset_stats(
    file_path: str | Path,
    dataset_path: str,
    max_bytes: int,
    axis: int | None = None,
) -> dict[str, Any]:
    """Compute statistics for a dataset using chunked reading.

    Args:
        file_path: Path to HDF5 file
        dataset_path: Path to dataset within file
        max_bytes: Maximum bytes to load at once
        axis: Axis along which to compute stats (None = all data)

    Returns:
        Dictionary with statistics
    """
    try:
        with h5py.File(file_path, "r") as f:
            if dataset_path not in f:
                return {"error": f"Dataset not found: {dataset_path}"}

            dataset = f[dataset_path]

            if not isinstance(dataset, h5py.Dataset):
                return {"error": f"Path is not a dataset: {dataset_path}"}

            # Check if dataset is numeric
            if not np.issubdtype(dataset.dtype, np.number):
                return {"error": f"Dataset is not numeric (dtype: {dataset.dtype})"}

            total_size_bytes = dataset.size * dataset.dtype.itemsize

            # If dataset fits in memory, read it all
            if total_size_bytes <= max_bytes:
                data = dataset[...]
                return {
                    "path": dataset_path,
                    "shape": dataset.shape,
                    "dtype": str(dataset.dtype),
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "median": float(np.median(data)),
                    "chunked": False,
                }

            # Need to chunk - read along first axis
            chunk_size = compute_chunk_size(
                dataset.shape[0],
                dataset.dtype.itemsize * math.prod(dataset.shape[1:])
                if len(dataset.shape) > 1
                else dataset.dtype.itemsize,
                max_bytes,
            )

            # Compute statistics in streaming fashion
            min_val = float("inf")
            max_val = float("-inf")
            sum_val = 0.0
            sum_sq = 0.0
            count = 0
            all_values = []  # For median (may be memory-intensive)

            for i in range(0, dataset.shape[0], chunk_size):
                end = min(i + chunk_size, dataset.shape[0])
                chunk = dataset[i:end]

                if axis is not None:
                    chunk = np.take(chunk, indices=range(chunk.shape[0]), axis=0)

                chunk_flat = chunk.flatten()
                min_val = min(min_val, float(np.min(chunk_flat)))
                max_val = max(max_val, float(np.max(chunk_flat)))
                sum_val += float(np.sum(chunk_flat))
                sum_sq += float(np.sum(chunk_flat**2))
                count += chunk_flat.size

                # Collect values for median (warning: memory intensive for large datasets)
                if len(all_values) < 1_000_000:  # Cap at 1M elements for median
                    all_values.extend(chunk_flat.tolist())

            mean_val = sum_val / count
            variance = (sum_sq / count) - (mean_val**2)
            std_val = math.sqrt(max(0, variance))

            result = {
                "path": dataset_path,
                "shape": dataset.shape,
                "dtype": str(dataset.dtype),
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
                "chunked": True,
                "num_chunks": math.ceil(dataset.shape[0] / chunk_size),
            }

            # Add median if we collected enough samples
            if all_values:
                result["median"] = float(np.median(all_values))
                if len(all_values) < count:
                    result[
                        "median_note"
                    ] = f"Computed from {len(all_values)} samples (not all data)"

            return result

    except Exception as e:
        return {"error": f"Failed to compute statistics: {str(e)}"}


def dataset_histogram(
    file_path: str | Path,
    dataset_path: str,
    max_bytes: int,
    bins: int = 50,
    range_min: float | None = None,
    range_max: float | None = None,
    log_scale: bool = False,
) -> dict[str, Any]:
    """Compute histogram for a dataset using chunked reading.

    Args:
        file_path: Path to HDF5 file
        dataset_path: Path to dataset within file
        max_bytes: Maximum bytes to load at once
        bins: Number of histogram bins
        range_min: Minimum value for histogram range
        range_max: Maximum value for histogram range
        log_scale: Use logarithmic binning

    Returns:
        Dictionary with histogram data
    """
    try:
        with h5py.File(file_path, "r") as f:
            if dataset_path not in f:
                return {"error": f"Dataset not found: {dataset_path}"}

            dataset = f[dataset_path]

            if not isinstance(dataset, h5py.Dataset):
                return {"error": f"Path is not a dataset: {dataset_path}"}

            # Check if dataset is numeric
            if not np.issubdtype(dataset.dtype, np.number):
                return {"error": f"Dataset is not numeric (dtype: {dataset.dtype})"}

            total_size_bytes = dataset.size * dataset.dtype.itemsize

            # If no range specified, need to compute min/max first
            if range_min is None or range_max is None:
                stats = dataset_stats(file_path, dataset_path, max_bytes)
                if "error" in stats:
                    return stats
                range_min = range_min or stats["min"]
                range_max = range_max or stats["max"]

            # Prepare bins
            if log_scale:
                if range_min <= 0:
                    return {"error": "Log scale requires positive values"}
                bin_edges = np.logspace(
                    np.log10(range_min), np.log10(range_max), bins + 1
                )
            else:
                bin_edges = np.linspace(range_min, range_max, bins + 1)

            hist_counts = np.zeros(bins, dtype=np.int64)

            # If dataset fits in memory, read it all
            if total_size_bytes <= max_bytes:
                data = dataset[...].flatten()
                hist_counts, _ = np.histogram(data, bins=bin_edges)
            else:
                # Chunk the data
                chunk_size = compute_chunk_size(
                    dataset.shape[0],
                    dataset.dtype.itemsize * math.prod(dataset.shape[1:])
                    if len(dataset.shape) > 1
                    else dataset.dtype.itemsize,
                    max_bytes,
                )

                for i in range(0, dataset.shape[0], chunk_size):
                    end = min(i + chunk_size, dataset.shape[0])
                    chunk = dataset[i:end].flatten()
                    chunk_hist, _ = np.histogram(chunk, bins=bin_edges)
                    hist_counts += chunk_hist

            return {
                "path": dataset_path,
                "bins": bins,
                "bin_edges": bin_edges.tolist(),
                "counts": hist_counts.tolist(),
                "log_scale": log_scale,
                "range": [float(range_min), float(range_max)],
                "total_count": int(np.sum(hist_counts)),
            }

    except Exception as e:
        return {"error": f"Failed to compute histogram: {str(e)}"}


# Tool definitions for LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "dataset_stats",
            "description": "Compute statistics (min, max, mean, std, median) for a dataset. Automatically chunks large datasets to respect memory limits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to the dataset in the HDF5 file",
                    },
                    "axis": {
                        "type": "integer",
                        "description": "Axis along which to compute stats (optional)",
                    },
                },
                "required": ["dataset_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dataset_histogram",
            "description": "Compute histogram for a dataset. Automatically chunks large datasets to respect memory limits.",
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
                    "range_min": {
                        "type": "number",
                        "description": "Minimum value for histogram range (optional, auto-detected if not provided)",
                    },
                    "range_max": {
                        "type": "number",
                        "description": "Maximum value for histogram range (optional, auto-detected if not provided)",
                    },
                    "log_scale": {
                        "type": "boolean",
                        "description": "Use logarithmic binning (default: false)",
                        "default": False,
                    },
                },
                "required": ["dataset_path"],
            },
        },
    },
]
