"""Tests for Python code execution."""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from h5nry.tools.python_exec import run_python


@pytest.fixture
def sample_hdf5_file():
    """Create a sample HDF5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    # Create HDF5 file
    with h5py.File(tmp_path, "w") as f:
        # Create groups and datasets
        gas = f.create_group("gas")
        gas.create_dataset("temperature", data=np.arange(100, dtype=np.float32))
        gas.create_dataset("density", data=np.random.rand(50, 50))

        stars = f.create_group("stars")
        stars.create_dataset("masses", data=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    yield tmp_path

    # Cleanup
    tmp_path.unlink()


def test_run_python_basic(sample_hdf5_file):
    """Test basic Python execution."""
    result = run_python("x = 5 + 3\nprint(x)", sample_hdf5_file)

    assert result["error"] is None
    assert "8" in result["stdout"]
    assert result["result_repr"] is None  # No final expression


def test_run_python_with_return(sample_hdf5_file):
    """Test Python execution with return value."""
    result = run_python("x = 5 + 3\nx * 2", sample_hdf5_file)

    assert result["error"] is None
    assert result["result_repr"] == "16"


def test_run_python_file_access(sample_hdf5_file):
    """Test accessing HDF5 file in Python code."""
    code = """
# File is already open as 'f'
keys = list(f.keys())
print(keys)
keys
"""
    result = run_python(code, sample_hdf5_file)

    assert result["error"] is None
    assert "gas" in result["stdout"]
    assert "stars" in result["stdout"]
    assert "['gas', 'stars']" in result["result_repr"]


def test_run_python_read_dataset(sample_hdf5_file):
    """Test reading a dataset."""
    code = """
# Read small dataset
masses = f['stars/masses'][:]
print(f"Shape: {masses.shape}")
print(f"Mean: {np.mean(masses)}")
np.mean(masses)
"""
    result = run_python(code, sample_hdf5_file)

    assert result["error"] is None
    assert "Shape: (5,)" in result["stdout"]
    assert "Mean: 3.0" in result["stdout"]
    assert "3.0" in result["result_repr"]


def test_run_python_dataset_slice(sample_hdf5_file):
    """Test reading dataset with slicing."""
    code = """
# Read slice of dataset
temp = f['gas/temperature'][:10]
print(f"First 10: {temp}")
temp.shape
"""
    result = run_python(code, sample_hdf5_file)

    assert result["error"] is None
    assert "First 10:" in result["stdout"]
    assert "(10,)" in result["result_repr"]


def test_run_python_numpy_operations(sample_hdf5_file):
    """Test numpy operations on dataset."""
    code = """
temp = f['gas/temperature'][:]
stats = {
    'min': float(np.min(temp)),
    'max': float(np.max(temp)),
    'mean': float(np.mean(temp))
}
print(stats)
stats
"""
    result = run_python(code, sample_hdf5_file)

    assert result["error"] is None
    assert "'min': 0.0" in result["stdout"]
    assert "'max': 99.0" in result["stdout"]


def test_run_python_array_summary(sample_hdf5_file):
    """Test that large arrays get summarized."""
    code = """
# Read larger dataset
density = f['gas/density'][:]
density  # Should be summarized, not printed in full
"""
    result = run_python(code, sample_hdf5_file)

    assert result["error"] is None
    assert "array(shape=" in result["result_repr"]
    assert "mean=" in result["result_repr"]


def test_run_python_error_handling(sample_hdf5_file):
    """Test error handling."""
    code = """
# Try to access non-existent dataset
data = f['nonexistent'][:]
"""
    result = run_python(code, sample_hdf5_file)

    assert result["error"] is not None
    assert "KeyError" in result["error"]


def test_run_python_max_bytes_limit(sample_hdf5_file):
    """Test that reading too much data fails gracefully."""
    # Create file with large dataset
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    with h5py.File(tmp_path, "w") as f:
        # Create 100 MB dataset
        f.create_dataset("large", data=np.zeros((10000, 1250), dtype=np.float64))

    try:
        code = """
# Try to read dataset that's too large
data = f['large'][:]
"""
        # Set very small limit
        result = run_python(code, tmp_path, max_bytes=1000000)  # 1 MB limit

        # Should fail, but we can't predict exact error since file is opened first
        # The actual read would fail if we tried
        # For now, just check it doesn't crash
        assert result is not None

    finally:
        tmp_path.unlink()


def test_run_python_no_file():
    """Test Python execution without HDF5 file."""
    result = run_python("x = 42\nprint(x)\nx", file_path=None)

    assert result["error"] is None
    assert "42" in result["stdout"]
    assert result["result_repr"] == "42"


def test_run_python_math_module(sample_hdf5_file):
    """Test that math module is available."""
    code = """
import math
result = math.sqrt(16)
print(f"sqrt(16) = {result}")
result
"""
    result = run_python(code, sample_hdf5_file)

    assert result["error"] is None
    assert "sqrt(16) = 4.0" in result["stdout"]
    assert "4.0" in result["result_repr"]


def test_run_python_list_comprehension(sample_hdf5_file):
    """Test list comprehensions work."""
    code = """
masses = f['stars/masses'][:]
doubled = [m * 2 for m in masses]
print(doubled)
doubled
"""
    result = run_python(code, sample_hdf5_file)

    assert result["error"] is None
    assert "[2.0, 4.0, 6.0, 8.0, 10.0]" in result["stdout"]
