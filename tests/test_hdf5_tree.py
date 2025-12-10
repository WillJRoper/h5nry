"""Tests for HDF5 tree building and inspection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from h5nry.tools.hdf5_tree import (
    DatasetNode,
    GroupNode,
    build_tree,
    get_node_info,
    list_children,
    summarize_tree,
)


@pytest.fixture
def sample_hdf5_file():
    """Create a sample HDF5 file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    # Create HDF5 file with structure
    with h5py.File(tmp_path, "w") as f:
        # Root attributes
        f.attrs["description"] = "Test HDF5 file"
        f.attrs["version"] = "1.0"

        # Create groups
        gas = f.create_group("gas")
        gas.attrs["description"] = "Gas particle data"

        stars = f.create_group("stars")
        stars.attrs["description"] = "Stellar particle data"

        # Create datasets
        gas.create_dataset("temperature", data=np.random.rand(100, 50))
        gas["temperature"].attrs["description"] = "Gas temperature in Kelvin"
        gas["temperature"].attrs["units"] = "K"

        gas.create_dataset("density", data=np.random.rand(100, 50))

        stars.create_dataset("masses", data=np.random.rand(200))
        stars["masses"].attrs["description"] = "Stellar masses"

    yield tmp_path

    # Cleanup
    tmp_path.unlink()


def test_build_tree(sample_hdf5_file):
    """Test building tree from HDF5 file."""
    tree = build_tree(sample_hdf5_file)

    assert isinstance(tree, GroupNode)
    assert tree.name == "/"
    assert tree.path == "/"
    assert len(tree.children) == 2

    # Check that description was loaded
    assert "description" in tree.attributes
    assert tree.attributes["description"] == "Test HDF5 file"

    # Check that other attributes have metadata only
    assert "version" in tree.attributes
    assert isinstance(tree.attributes["version"], dict)
    assert tree.attributes["version"]["_metadata"] is True


def test_tree_structure(sample_hdf5_file):
    """Test that tree structure matches file structure."""
    tree = build_tree(sample_hdf5_file)

    # Find gas group
    gas_node = next(child for child in tree.children if child.name == "gas")
    assert isinstance(gas_node, GroupNode)
    assert gas_node.attributes["description"] == "Gas particle data"
    assert len(gas_node.children) == 2

    # Find temperature dataset
    temp_node = next(
        child for child in gas_node.children if child.name == "temperature"
    )
    assert isinstance(temp_node, DatasetNode)
    assert temp_node.shape == (100, 50)
    assert temp_node.attributes["description"] == "Gas temperature in Kelvin"

    # Units attribute should be metadata only
    assert "units" in temp_node.attributes
    assert isinstance(temp_node.attributes["units"], dict)


def test_summarize_tree(sample_hdf5_file):
    """Test tree summarization."""
    tree = build_tree(sample_hdf5_file)
    summary = summarize_tree(tree, max_depth=3, max_children=20)

    assert "/" in summary
    assert "gas/" in summary
    assert "stars/" in summary
    assert "temperature" in summary
    assert "masses" in summary
    assert "shape=" in summary


def test_get_node_info(sample_hdf5_file):
    """Test getting node information."""
    tree = build_tree(sample_hdf5_file)

    # Get root info
    root_info = get_node_info(tree, "/")
    assert root_info["type"] == "group"
    assert root_info["num_children"] == 2

    # Get dataset info
    temp_info = get_node_info(tree, "/gas/temperature")
    assert temp_info["type"] == "dataset"
    assert temp_info["shape"] == (100, 50)
    assert "dtype" in temp_info
    assert "size_mb" in temp_info

    # Non-existent path
    missing_info = get_node_info(tree, "/nonexistent")
    assert "error" in missing_info


def test_list_children(sample_hdf5_file):
    """Test listing children of a group."""
    tree = build_tree(sample_hdf5_file)

    # List root children
    root_children = list_children(tree, "/")
    assert root_children["num_children"] == 2
    assert len(root_children["children"]) == 2

    child_names = [child["name"] for child in root_children["children"]]
    assert "gas" in child_names
    assert "stars" in child_names

    # List gas children
    gas_children = list_children(tree, "/gas")
    assert gas_children["num_children"] == 2

    # Try listing children of a dataset (should error)
    dataset_children = list_children(tree, "/gas/temperature")
    assert "error" in dataset_children
