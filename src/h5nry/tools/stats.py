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


def dataset_preview(
    file_path: str | Path,
    dataset_path: str,
    max_bytes: int,
    max_elements: int = 100,
    axis: int | None = None,
) -> dict[str, Any]:
    """Provide a preview of dataset values.

    Args:
        file_path: Path to HDF5 file
        dataset_path: Path to dataset within file
        max_bytes: Maximum bytes to load
        max_elements: Maximum elements to return (default: 100)
        axis: Optional axis to preview along

    Returns:
        Dictionary with preview data
    """
    try:
        with h5py.File(file_path, "r") as f:
            if dataset_path not in f:
                return {"error": f"Dataset not found: {dataset_path}"}

            dataset = f[dataset_path]

            if not isinstance(dataset, h5py.Dataset):
                return {"error": f"Path is not a dataset: {dataset_path}"}

            shape = dataset.shape
            dtype = dataset.dtype

            # Determine slice to read
            if axis is not None:
                # Preview along specific axis
                if axis >= len(shape):
                    return {"error": f"Axis {axis} out of range for shape {shape}"}

                # Take first max_elements along specified axis
                slice_size = min(max_elements, shape[axis])
                slices = [slice(None)] * len(shape)
                slices[axis] = slice(0, slice_size)
                slice_tuple = tuple(slices)

                # Check size
                total_elements = 1
                for i, s in enumerate(shape):
                    if i == axis:
                        total_elements *= slice_size
                    else:
                        total_elements *= s

                size_bytes = total_elements * dtype.itemsize
                if size_bytes > max_bytes:
                    return {
                        "error": f"Slice would be {size_bytes / (1024*1024):.2f} MB, exceeds limit of {max_bytes / (1024*1024):.2f} MB"
                    }

                data = dataset[slice_tuple]
                preview_values = data.tolist()

            else:
                # Flatten and take first max_elements
                total_elements = math.prod(shape)
                actual_elements = min(max_elements, total_elements)
                size_bytes = actual_elements * dtype.itemsize

                if size_bytes > max_bytes:
                    # Reduce number of elements
                    actual_elements = max_bytes // dtype.itemsize
                    if actual_elements == 0:
                        return {
                            "error": f"Element size ({dtype.itemsize} bytes) exceeds max_bytes limit"
                        }

                # Read flat preview
                flat_data = dataset.flat[:actual_elements]
                preview_values = flat_data.tolist()

            return {
                "path": dataset_path,
                "shape": shape,
                "dtype": str(dtype),
                "preview_elements": len(preview_values)
                if isinstance(preview_values, list)
                else 1,
                "total_elements": math.prod(shape),
                "preview_values": preview_values,
                "axis": axis,
            }

    except Exception as e:
        return {"error": f"Failed to preview dataset: {str(e)}"}


def dataset_missing_values(
    file_path: str | Path,
    dataset_path: str,
    max_bytes: int,
) -> dict[str, Any]:
    """Count NaN and infinity values in a numeric dataset.

    Args:
        file_path: Path to HDF5 file
        dataset_path: Path to dataset within file
        max_bytes: Maximum bytes to load at once

    Returns:
        Dictionary with missing value counts
    """
    try:
        with h5py.File(file_path, "r") as f:
            if dataset_path not in f:
                return {"error": f"Dataset not found: {dataset_path}"}

            dataset = f[dataset_path]

            if not isinstance(dataset, h5py.Dataset):
                return {"error": f"Path is not a dataset: {dataset_path}"}

            # Check if float type
            if not np.issubdtype(dataset.dtype, np.floating):
                return {
                    "path": dataset_path,
                    "dtype": str(dataset.dtype),
                    "message": "Dataset is not floating-point type, no NaN/inf values possible",
                }

            total_elements = dataset.size
            num_nan = 0
            num_posinf = 0
            num_neginf = 0

            # Chunk through dataset
            chunk_size = compute_chunk_size(
                dataset.shape[0],
                dataset.dtype.itemsize * math.prod(dataset.shape[1:])
                if len(dataset.shape) > 1
                else dataset.dtype.itemsize,
                max_bytes,
            )

            for i in range(0, dataset.shape[0], chunk_size):
                end = min(i + chunk_size, dataset.shape[0])
                chunk = dataset[i:end]
                chunk_flat = chunk.flatten()

                num_nan += np.count_nonzero(np.isnan(chunk_flat))
                num_posinf += np.count_nonzero(np.isposinf(chunk_flat))
                num_neginf += np.count_nonzero(np.isneginf(chunk_flat))

            return {
                "path": dataset_path,
                "shape": dataset.shape,
                "dtype": str(dataset.dtype),
                "total_elements": total_elements,
                "num_nan": int(num_nan),
                "num_positive_inf": int(num_posinf),
                "num_negative_inf": int(num_neginf),
                "num_finite": int(total_elements - num_nan - num_posinf - num_neginf),
                "fraction_nan": float(num_nan / total_elements),
                "fraction_positive_inf": float(num_posinf / total_elements),
                "fraction_negative_inf": float(num_neginf / total_elements),
                "fraction_finite": float(
                    (total_elements - num_nan - num_posinf - num_neginf)
                    / total_elements
                ),
            }

    except Exception as e:
        return {"error": f"Failed to count missing values: {str(e)}"}


def dataset_value_counts(
    file_path: str | Path,
    dataset_path: str,
    max_bytes: int,
    max_unique: int = 50,
) -> dict[str, Any]:
    """Compute value counts for categorical/discrete datasets.

    Args:
        file_path: Path to HDF5 file
        dataset_path: Path to dataset within file
        max_bytes: Maximum bytes to load at once
        max_unique: Maximum unique values to track

    Returns:
        Dictionary with value counts
    """
    try:
        with h5py.File(file_path, "r") as f:
            if dataset_path not in f:
                return {"error": f"Dataset not found: {dataset_path}"}

            dataset = f[dataset_path]

            if not isinstance(dataset, h5py.Dataset):
                return {"error": f"Path is not a dataset: {dataset_path}"}

            # Check if suitable dtype
            if not np.issubdtype(dataset.dtype, np.integer) and not np.issubdtype(
                dataset.dtype, np.bool_
            ):
                return {
                    "path": dataset_path,
                    "dtype": str(dataset.dtype),
                    "message": "Value counts work best with integer or boolean dtypes. For float data, consider binning first.",
                }

            value_counts: dict[Any, int] = {}
            truncated = False
            total_elements = 0

            # Chunk through dataset
            chunk_size = compute_chunk_size(
                dataset.shape[0],
                dataset.dtype.itemsize * math.prod(dataset.shape[1:])
                if len(dataset.shape) > 1
                else dataset.dtype.itemsize,
                max_bytes,
            )

            for i in range(0, dataset.shape[0], chunk_size):
                end = min(i + chunk_size, dataset.shape[0])
                chunk = dataset[i:end]
                chunk_flat = chunk.flatten()
                total_elements += chunk_flat.size

                for value in chunk_flat:
                    # Convert numpy types to Python types for JSON
                    value_key = (
                        int(value) if np.issubdtype(type(value), np.integer) else value
                    )

                    if value_key in value_counts:
                        value_counts[value_key] += 1
                    elif len(value_counts) < max_unique:
                        value_counts[value_key] = 1
                    else:
                        truncated = True

            # Sort by count descending
            sorted_counts = sorted(
                value_counts.items(), key=lambda x: x[1], reverse=True
            )

            return {
                "path": dataset_path,
                "shape": dataset.shape,
                "dtype": str(dataset.dtype),
                "total_elements": total_elements,
                "num_unique_values": len(value_counts),
                "truncated": truncated,
                "max_unique": max_unique,
                "value_counts": [{"value": v, "count": c} for v, c in sorted_counts],
            }

    except Exception as e:
        return {"error": f"Failed to compute value counts: {str(e)}"}


def dataset_storage_info(
    file_path: str | Path,
    dataset_path: str,
) -> dict[str, Any]:
    """Get storage and I/O characteristics of a dataset.

    Args:
        file_path: Path to HDF5 file
        dataset_path: Path to dataset within file

    Returns:
        Dictionary with storage information
    """
    try:
        with h5py.File(file_path, "r") as f:
            if dataset_path not in f:
                return {"error": f"Dataset not found: {dataset_path}"}

            dataset = f[dataset_path]

            if not isinstance(dataset, h5py.Dataset):
                return {"error": f"Path is not a dataset: {dataset_path}"}

            logical_size_bytes = dataset.size * dataset.dtype.itemsize

            result: dict[str, Any] = {
                "path": dataset_path,
                "shape": dataset.shape,
                "dtype": str(dataset.dtype),
                "num_elements": dataset.size,
                "logical_size_bytes": logical_size_bytes,
                "logical_size_mb": logical_size_bytes / (1024 * 1024),
                "logical_size_gb": logical_size_bytes / (1024 * 1024 * 1024),
                "chunks": dataset.chunks,
                "compression": dataset.compression,
                "compression_opts": dataset.compression_opts,
                "shuffle": dataset.shuffle if hasattr(dataset, "shuffle") else None,
                "fletcher32": dataset.fletcher32
                if hasattr(dataset, "fletcher32")
                else None,
                "scaleoffset": dataset.scaleoffset
                if hasattr(dataset, "scaleoffset")
                else None,
            }

            # Try to get storage size (on-disk size)
            try:
                # get_storage_size() returns size in bytes
                on_disk_size = dataset.id.get_storage_size()
                if on_disk_size > 0:
                    result["on_disk_size_bytes"] = on_disk_size
                    result["on_disk_size_mb"] = on_disk_size / (1024 * 1024)
                    result["on_disk_size_gb"] = on_disk_size / (1024 * 1024 * 1024)
                    result["compression_ratio"] = logical_size_bytes / on_disk_size
                else:
                    result["on_disk_size_note"] = "Dataset not yet allocated or virtual"
            except Exception:
                result[
                    "on_disk_size_note"
                ] = "Unable to determine on-disk size (may be virtual dataset)"

            return result

    except Exception as e:
        return {"error": f"Failed to get storage info: {str(e)}"}


def dataset_slice(
    file_path: str | Path,
    dataset_path: str,
    slice_spec: str,
    max_bytes: int,
    max_elements: int = 1000,
) -> dict[str, Any]:
    """Safely return a bounded slice of a dataset.

    Args:
        file_path: Path to HDF5 file
        dataset_path: Path to dataset within file
        slice_spec: Slice specification (e.g., ":100", "0:10,0:10", "::2")
        max_bytes: Maximum bytes to load
        max_elements: Maximum elements to return

    Returns:
        Dictionary with sliced data
    """
    try:
        with h5py.File(file_path, "r") as f:
            if dataset_path not in f:
                return {"error": f"Dataset not found: {dataset_path}"}

            dataset = f[dataset_path]

            if not isinstance(dataset, h5py.Dataset):
                return {"error": f"Path is not a dataset: {dataset_path}"}

            # Parse slice specification
            slice_parts = slice_spec.split(",")
            if len(slice_parts) > len(dataset.shape):
                return {
                    "error": f"Slice has {len(slice_parts)} dimensions but dataset has {len(dataset.shape)}"
                }

            slices = []
            for part in slice_parts:
                part = part.strip()
                if ":" not in part:
                    # Single index
                    slices.append(int(part))
                else:
                    # Slice notation
                    components = part.split(":")
                    start = int(components[0]) if components[0] else None
                    stop = (
                        int(components[1])
                        if len(components) > 1 and components[1]
                        else None
                    )
                    step = (
                        int(components[2])
                        if len(components) > 2 and components[2]
                        else None
                    )
                    slices.append(slice(start, stop, step))

            # Convert to tuple
            slice_tuple = tuple(slices)

            # Compute result shape and size
            try:
                # Get a view to check shape
                test_slice = dataset[slice_tuple]
                result_shape = test_slice.shape
                result_elements = math.prod(result_shape)
                result_bytes = result_elements * dataset.dtype.itemsize

                # Check limits
                if result_elements > max_elements:
                    return {
                        "error": f"Slice would return {result_elements} elements, exceeds max_elements={max_elements}"
                    }

                if result_bytes > max_bytes:
                    return {
                        "error": f"Slice would be {result_bytes / (1024*1024):.2f} MB, exceeds limit of {max_bytes / (1024*1024):.2f} MB"
                    }

                # Actually read the data
                data = dataset[slice_tuple]
                values = data.tolist()

                return {
                    "path": dataset_path,
                    "slice_spec": slice_spec,
                    "original_shape": dataset.shape,
                    "result_shape": result_shape,
                    "dtype": str(dataset.dtype),
                    "num_elements": result_elements,
                    "values": values,
                }

            except Exception as e:
                return {"error": f"Invalid slice: {str(e)}"}

    except Exception as e:
        return {"error": f"Failed to slice dataset: {str(e)}"}


def dataset_correlation(
    file_path: str | Path,
    x_dataset_path: str,
    y_dataset_path: str,
    max_bytes: int,
    method: str = "pearson",
) -> dict[str, Any]:
    """Compute correlation between two numeric 1D datasets.

    Args:
        file_path: Path to HDF5 file
        x_dataset_path: Path to first dataset
        y_dataset_path: Path to second dataset
        max_bytes: Maximum bytes to load at once
        method: Correlation method ("pearson" or "spearman")

    Returns:
        Dictionary with correlation results
    """
    try:
        with h5py.File(file_path, "r") as f:
            if x_dataset_path not in f:
                return {"error": f"Dataset not found: {x_dataset_path}"}
            if y_dataset_path not in f:
                return {"error": f"Dataset not found: {y_dataset_path}"}

            x_dataset = f[x_dataset_path]
            y_dataset = f[y_dataset_path]

            if not isinstance(x_dataset, h5py.Dataset):
                return {"error": f"Path is not a dataset: {x_dataset_path}"}
            if not isinstance(y_dataset, h5py.Dataset):
                return {"error": f"Path is not a dataset: {y_dataset_path}"}

            # Check numeric
            if not np.issubdtype(x_dataset.dtype, np.number):
                return {"error": f"{x_dataset_path} is not numeric"}
            if not np.issubdtype(y_dataset.dtype, np.number):
                return {"error": f"{y_dataset_path} is not numeric"}

            # Check shapes
            if x_dataset.shape != y_dataset.shape:
                return {
                    "error": f"Shape mismatch: {x_dataset.shape} vs {y_dataset.shape}"
                }

            if len(x_dataset.shape) != 1:
                return {"error": f"Datasets must be 1D, got shape {x_dataset.shape}"}

            n_samples = x_dataset.shape[0]

            if method == "spearman":
                return {
                    "path_x": x_dataset_path,
                    "path_y": y_dataset_path,
                    "method": method,
                    "message": "Spearman correlation requires ranking and is not yet implemented for chunked processing. Use Pearson correlation or run_python tool for custom analysis.",
                }

            # Pearson correlation via streaming computation
            # cor(X,Y) = cov(X,Y) / (std(X) * std(Y))
            # cov(X,Y) = E[XY] - E[X]E[Y]

            sum_x = 0.0
            sum_y = 0.0
            sum_x2 = 0.0
            sum_y2 = 0.0
            sum_xy = 0.0
            count = 0

            # Determine chunk size (both datasets same shape)
            element_size = x_dataset.dtype.itemsize + y_dataset.dtype.itemsize
            chunk_size = compute_chunk_size(n_samples, element_size, max_bytes)

            for i in range(0, n_samples, chunk_size):
                end = min(i + chunk_size, n_samples)
                x_chunk = x_dataset[i:end].astype(float)
                y_chunk = y_dataset[i:end].astype(float)

                sum_x += np.sum(x_chunk)
                sum_y += np.sum(y_chunk)
                sum_x2 += np.sum(x_chunk**2)
                sum_y2 += np.sum(y_chunk**2)
                sum_xy += np.sum(x_chunk * y_chunk)
                count += len(x_chunk)

            mean_x = sum_x / count
            mean_y = sum_y / count

            var_x = (sum_x2 / count) - (mean_x**2)
            var_y = (sum_y2 / count) - (mean_y**2)
            cov_xy = (sum_xy / count) - (mean_x * mean_y)

            std_x = math.sqrt(max(0, var_x))
            std_y = math.sqrt(max(0, var_y))

            if std_x == 0 or std_y == 0:
                return {
                    "error": "One or both datasets have zero variance (constant values)"
                }

            correlation = cov_xy / (std_x * std_y)

            return {
                "path_x": x_dataset_path,
                "path_y": y_dataset_path,
                "method": method,
                "n_samples": n_samples,
                "correlation_coefficient": float(correlation),
                "mean_x": float(mean_x),
                "mean_y": float(mean_y),
                "std_x": float(std_x),
                "std_y": float(std_y),
            }

    except Exception as e:
        return {"error": f"Failed to compute correlation: {str(e)}"}
