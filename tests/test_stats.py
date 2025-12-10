"""Tests for statistical analysis tools."""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from h5nry.tools.stats import compute_chunk_size, dataset_histogram, dataset_stats


@pytest.fixture
def sample_numeric_file():
    """Create a sample HDF5 file with numeric datasets."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    # Create HDF5 file
    with h5py.File(tmp_path, "w") as f:
        # Small dataset that fits in memory
        small_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        f.create_dataset("small", data=small_data)

        # Larger dataset for chunking tests
        large_data = np.random.rand(1000, 100)
        f.create_dataset("large", data=large_data)

        # Non-numeric dataset
        string_data = np.array([b"hello", b"world"])
        f.create_dataset("strings", data=string_data)

    yield tmp_path

    # Cleanup
    tmp_path.unlink()


def test_compute_chunk_size():
    """Test chunk size computation."""
    # Small dataset
    chunk_size = compute_chunk_size(
        total_elements=100,
        element_size_bytes=8,
        max_bytes=1000,
    )
    assert chunk_size == 100  # Fits entirely

    # Large dataset
    chunk_size = compute_chunk_size(
        total_elements=10000,
        element_size_bytes=8,
        max_bytes=1000,
    )
    assert chunk_size == 125  # 1000 / 8

    # Ensure at least 1 element
    chunk_size = compute_chunk_size(
        total_elements=1,
        element_size_bytes=10000,
        max_bytes=100,
    )
    assert chunk_size == 1


def test_dataset_stats_small(sample_numeric_file):
    """Test statistics on a small dataset."""
    max_bytes = 1000  # Small limit

    stats = dataset_stats(sample_numeric_file, "/small", max_bytes)

    assert "error" not in stats
    assert stats["min"] == 1.0
    assert stats["max"] == 5.0
    assert stats["mean"] == 3.0
    assert stats["std"] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0, 5.0]), rel=1e-5)
    assert stats["median"] == 3.0
    assert stats["chunked"] is False


def test_dataset_stats_with_chunking(sample_numeric_file):
    """Test statistics with forced chunking."""
    # Very small max_bytes to force chunking
    max_bytes = 1000  # Will force chunking on large dataset

    stats = dataset_stats(sample_numeric_file, "/large", max_bytes)

    assert "error" not in stats
    assert "min" in stats
    assert "max" in stats
    assert "mean" in stats
    assert "std" in stats
    assert stats["chunked"] is True
    assert stats["num_chunks"] > 1

    # Values should be reasonable for random data
    assert 0.0 <= stats["min"] <= 1.0
    assert 0.0 <= stats["max"] <= 1.0
    assert 0.0 <= stats["mean"] <= 1.0


def test_dataset_stats_nonexistent(sample_numeric_file):
    """Test statistics on non-existent dataset."""
    stats = dataset_stats(sample_numeric_file, "/nonexistent", 1000000)
    assert "error" in stats


def test_dataset_stats_non_numeric(sample_numeric_file):
    """Test statistics on non-numeric dataset."""
    stats = dataset_stats(sample_numeric_file, "/strings", 1000000)
    assert "error" in stats
    assert "not numeric" in stats["error"].lower()


def test_dataset_histogram_small(sample_numeric_file):
    """Test histogram on small dataset."""
    max_bytes = 1000

    hist = dataset_histogram(
        sample_numeric_file,
        "/small",
        max_bytes,
        bins=5,
        range_min=0.0,
        range_max=6.0,
    )

    assert "error" not in hist
    assert hist["bins"] == 5
    assert len(hist["bin_edges"]) == 6
    assert len(hist["counts"]) == 5
    assert hist["total_count"] == 5
    assert hist["log_scale"] is False


def test_dataset_histogram_with_chunking(sample_numeric_file):
    """Test histogram with forced chunking."""
    max_bytes = 1000  # Force chunking

    hist = dataset_histogram(
        sample_numeric_file,
        "/large",
        max_bytes,
        bins=10,
    )

    assert "error" not in hist
    assert hist["bins"] == 10
    assert len(hist["counts"]) == 10
    assert hist["total_count"] == 1000 * 100  # Total elements


def test_dataset_histogram_log_scale(sample_numeric_file):
    """Test histogram with logarithmic scale."""
    # Create positive data for log scale
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    with h5py.File(tmp_path, "w") as f:
        data = np.logspace(0, 3, 100)  # 1 to 1000
        f.create_dataset("log_data", data=data)

    try:
        hist = dataset_histogram(
            tmp_path,
            "/log_data",
            max_bytes=10000,
            bins=10,
            range_min=1.0,
            range_max=1000.0,
            log_scale=True,
        )

        assert "error" not in hist
        assert hist["log_scale"] is True

    finally:
        tmp_path.unlink()


def test_dataset_histogram_auto_range(sample_numeric_file):
    """Test histogram with automatic range detection."""
    max_bytes = 1000

    hist = dataset_histogram(
        sample_numeric_file,
        "/small",
        max_bytes,
        bins=5,
    )

    assert "error" not in hist
    assert hist["range"][0] == 1.0
    assert hist["range"][1] == 5.0
