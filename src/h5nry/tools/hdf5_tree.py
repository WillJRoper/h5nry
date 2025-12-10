"""HDF5 tree inspection tools."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np


@dataclass
class H5Node:
    """Base class for HDF5 nodes."""

    name: str
    path: str
    type: Literal["group", "dataset"]
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupNode(H5Node):
    """HDF5 group node."""

    type: Literal["group", "dataset"] = field(default="group")
    children: list[H5Node] = field(default_factory=list)


@dataclass
class DatasetNode(H5Node):
    """HDF5 dataset node."""

    type: Literal["group", "dataset"] = field(default="dataset")
    shape: tuple[int, ...] = field(default_factory=tuple)
    dtype: str = ""
    size_bytes: int = 0
    chunks: tuple[int, ...] | None = None
    compression: str | None = None


def _should_load_attribute_value(attr_name: str) -> bool:
    """Check if an attribute value should be fully loaded.

    Only "description" attributes (case-insensitive) are loaded by default.

    Args:
        attr_name: Attribute name

    Returns:
        Whether to load the full value
    """
    return attr_name.lower() == "description"


def _parse_attributes(h5obj: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    """Parse attributes from an HDF5 object.

    Args:
        h5obj: HDF5 group or dataset

    Returns:
        Dictionary of attributes with metadata
    """
    attrs = {}
    for key in h5obj.attrs:
        attr_value = h5obj.attrs[key]
        if _should_load_attribute_value(key):
            # Load full value for description attributes
            attrs[key] = attr_value
        else:
            # Store only metadata for other attributes
            attrs[key] = {
                "_metadata": True,
                "type": str(type(attr_value).__name__),
                "present": True,
            }
    return attrs


def _build_tree_recursive(h5obj: h5py.Group | h5py.Dataset, path: str) -> H5Node:
    """Recursively build tree from HDF5 object.

    Args:
        h5obj: HDF5 group or dataset
        path: Full path to this object

    Returns:
        Tree node representing this object and its children
    """
    name = path.split("/")[-1] if "/" in path else path
    attrs = _parse_attributes(h5obj)

    if isinstance(h5obj, h5py.Dataset):
        return DatasetNode(
            name=name,
            path=path,
            type="dataset",
            attributes=attrs,
            shape=h5obj.shape,
            dtype=str(h5obj.dtype),
            size_bytes=h5obj.size * h5obj.dtype.itemsize,
            chunks=h5obj.chunks,
            compression=h5obj.compression,
        )
    else:  # Group
        children = []
        for key in h5obj:
            child_path = f"{path}/{key}" if path != "/" else f"/{key}"
            children.append(_build_tree_recursive(h5obj[key], child_path))

        return GroupNode(
            name=name or "/",
            path=path or "/",
            type="group",
            attributes=attrs,
            children=children,
        )


def build_tree(file_path: str | Path) -> H5Node:
    """Build a tree representation of an HDF5 file.

    Args:
        file_path: Path to HDF5 file

    Returns:
        Root node of the tree
    """
    with h5py.File(file_path, "r") as f:
        return _build_tree_recursive(f, "/")


def _summarize_node(
    node: H5Node,
    current_depth: int,
    max_depth: int,
    max_children: int,
) -> list[str]:
    """Recursively summarize a node.

    Args:
        node: Node to summarize
        current_depth: Current recursion depth
        max_depth: Maximum depth to traverse
        max_children: Maximum children to show per group

    Returns:
        List of summary lines
    """
    lines = []
    indent = "  " * current_depth

    if isinstance(node, DatasetNode):
        size_mb = node.size_bytes / (1024 * 1024)
        lines.append(
            f"{indent}{node.name} [dataset: shape={node.shape}, "
            f"dtype={node.dtype}, size={size_mb:.2f}MB]"
        )
        # Add description if present
        if "description" in node.attributes:
            desc = node.attributes["description"]
            if isinstance(desc, str | bytes):
                desc_str = desc.decode() if isinstance(desc, bytes) else desc
                lines.append(f"{indent}  Description: {desc_str[:100]}")
    else:  # Group
        num_children = len(node.children)
        lines.append(f"{indent}{node.name}/ [group: {num_children} items]")

        # Add description if present
        if "description" in node.attributes:
            desc = node.attributes["description"]
            if isinstance(desc, str | bytes):
                desc_str = desc.decode() if isinstance(desc, bytes) else desc
                lines.append(f"{indent}  Description: {desc_str[:100]}")

        if current_depth < max_depth:
            children_to_show = node.children[:max_children]
            for child in children_to_show:
                lines.extend(
                    _summarize_node(child, current_depth + 1, max_depth, max_children)
                )

            if num_children > max_children:
                lines.append(
                    f"{indent}  ... and {num_children - max_children} more items"
                )

    return lines


def summarize_tree(
    root: H5Node,
    max_depth: int = 3,
    max_children: int = 20,
) -> str:
    """Create a text summary of the HDF5 tree.

    Args:
        root: Root node of the tree
        max_depth: Maximum depth to traverse
        max_children: Maximum children to show per group

    Returns:
        Text summary of the tree
    """
    lines = _summarize_node(root, 0, max_depth, max_children)
    return "\n".join(lines)


def _find_node(root: H5Node, path: str) -> H5Node | None:
    """Find a node by path.

    Args:
        root: Root node
        path: Path to find

    Returns:
        Node at path or None if not found
    """
    if path == "/" or path == root.path:
        return root

    # Normalize path
    path = path.strip("/")
    parts = path.split("/")

    current = root
    for part in parts:
        if not isinstance(current, GroupNode):
            return None

        # Find child with matching name
        found = False
        for child in current.children:
            if child.name == part:
                current = child
                found = True
                break

        if not found:
            return None

    return current


def get_node_info(root: H5Node, path: str) -> dict[str, Any]:
    """Get information about a node at a specific path.

    Args:
        root: Root node of the tree
        path: Path to the node

    Returns:
        Dictionary with node information
    """
    node = _find_node(root, path)
    if node is None:
        return {"error": f"Node not found at path: {path}"}

    info: dict[str, Any] = {
        "name": node.name,
        "path": node.path,
        "type": node.type,
        "attributes": node.attributes,
    }

    if isinstance(node, DatasetNode):
        info.update(
            {
                "shape": node.shape,
                "dtype": node.dtype,
                "size_bytes": node.size_bytes,
                "size_mb": node.size_bytes / (1024 * 1024),
                "chunks": node.chunks,
                "compression": node.compression,
            }
        )
    elif isinstance(node, GroupNode):
        info["num_children"] = len(node.children)
        info["children"] = [
            {"name": child.name, "type": child.type} for child in node.children
        ]

    return info


def list_children(root: H5Node, path: str = "/") -> dict[str, Any]:
    """List children of a group.

    Args:
        root: Root node of the tree
        path: Path to the group

    Returns:
        Dictionary with children information
    """
    node = _find_node(root, path)
    if node is None:
        return {"error": f"Node not found at path: {path}"}

    if not isinstance(node, GroupNode):
        return {"error": f"Node at {path} is not a group"}

    children = []
    for child in node.children:
        child_info: dict[str, Any] = {
            "name": child.name,
            "type": child.type,
        }
        if isinstance(child, DatasetNode):
            child_info.update(
                {
                    "shape": child.shape,
                    "dtype": child.dtype,
                    "size_mb": child.size_bytes / (1024 * 1024),
                }
            )
        elif isinstance(child, GroupNode):
            child_info["num_children"] = len(child.children)

        children.append(child_info)

    return {
        "path": path,
        "num_children": len(children),
        "children": children,
    }


def list_attributes(file_path: str | Path, path: str) -> dict[str, Any]:
    """List all attributes on a node without loading large values.

    Args:
        file_path: Path to HDF5 file
        path: Path to the node

    Returns:
        Dictionary with attribute metadata
    """
    try:
        with h5py.File(file_path, "r") as f:
            if path not in f:
                return {"error": f"Node not found: {path}"}

            obj = f[path]
            attrs_info = []

            for attr_name in obj.attrs:
                attr_value = obj.attrs[attr_name]
                attr_info = {
                    "name": attr_name,
                    "dtype": str(type(attr_value).__name__),
                }

                # Add shape information
                if hasattr(attr_value, "shape"):
                    attr_info["shape"] = attr_value.shape
                elif hasattr(attr_value, "__len__") and not isinstance(
                    attr_value, str | bytes
                ):
                    attr_info["shape"] = (len(attr_value),)
                else:
                    attr_info["shape"] = ()

                attrs_info.append(attr_info)

            return {
                "path": path,
                "num_attributes": len(attrs_info),
                "attributes": attrs_info,
            }

    except Exception as e:
        return {"error": f"Failed to list attributes: {str(e)}"}


def get_attribute(
    file_path: str | Path,
    path: str,
    attr_name: str,
    preview_len: int = 100,
) -> dict[str, Any]:
    """Get a single attribute value with truncation for large values.

    Args:
        file_path: Path to HDF5 file
        path: Path to the node
        attr_name: Name of the attribute
        preview_len: Maximum elements/characters to return

    Returns:
        Dictionary with attribute value and metadata
    """
    try:
        with h5py.File(file_path, "r") as f:
            if path not in f:
                return {"error": f"Node not found: {path}"}

            obj = f[path]

            if attr_name not in obj.attrs:
                return {"error": f"Attribute not found: {attr_name}"}

            attr_value = obj.attrs[attr_name]

            result: dict[str, Any] = {
                "name": attr_name,
                "dtype": str(type(attr_value).__name__),
            }

            # Handle different attribute types
            if isinstance(attr_value, str | bytes):
                value_str = (
                    attr_value.decode() if isinstance(attr_value, bytes) else attr_value
                )
                if len(value_str) <= preview_len:
                    result["value"] = value_str
                    result["truncated"] = False
                else:
                    result["value"] = value_str[:preview_len]
                    result["truncated"] = True
                    result["total_length"] = len(value_str)

            elif isinstance(attr_value, np.ndarray):
                if attr_value.size <= preview_len:
                    result["value"] = attr_value.tolist()
                    result["truncated"] = False
                else:
                    result["value"] = attr_value.flat[:preview_len].tolist()
                    result["truncated"] = True
                    result["total_length"] = attr_value.size
                result["shape"] = attr_value.shape

            elif hasattr(attr_value, "__len__") and not isinstance(
                attr_value, str | bytes
            ):
                # List or array-like
                if len(attr_value) <= preview_len:
                    result["value"] = list(attr_value)
                    result["truncated"] = False
                else:
                    result["value"] = list(attr_value[:preview_len])
                    result["truncated"] = True
                    result["total_length"] = len(attr_value)

            else:
                # Scalar value
                result["value"] = attr_value
                result["truncated"] = False

            return result

    except Exception as e:
        return {"error": f"Failed to get attribute: {str(e)}"}


def file_overview(file_path: str | Path, tree: H5Node) -> dict[str, Any]:
    """Provide a compact file-level summary.

    Args:
        file_path: Path to HDF5 file
        tree: Pre-parsed tree

    Returns:
        Dictionary with file overview
    """
    try:
        # Count groups and datasets
        def count_nodes(node: H5Node) -> tuple[int, int, int]:
            """Return (num_groups, num_datasets, total_bytes)."""
            if isinstance(node, DatasetNode):
                return (0, 1, node.size_bytes)
            else:
                groups = 1
                datasets = 0
                total_bytes = 0
                for child in node.children:
                    g, d, b = count_nodes(child)
                    groups += g
                    datasets += d
                    total_bytes += b
                return (groups, datasets, total_bytes)

        num_groups, num_datasets, total_logical_bytes = count_nodes(tree)

        # Try to get file size on disk
        try:
            file_size_bytes = Path(file_path).stat().st_size
        except Exception:
            file_size_bytes = None

        result: dict[str, Any] = {
            "file_path": str(file_path),
            "num_groups": num_groups,
            "num_datasets": num_datasets,
            "total_logical_size_bytes": total_logical_bytes,
            "total_logical_size_mb": total_logical_bytes / (1024 * 1024),
            "total_logical_size_gb": total_logical_bytes / (1024 * 1024 * 1024),
        }

        if file_size_bytes is not None:
            result["file_size_on_disk_bytes"] = file_size_bytes
            result["file_size_on_disk_mb"] = file_size_bytes / (1024 * 1024)
            result["file_size_on_disk_gb"] = file_size_bytes / (1024 * 1024 * 1024)
            result["compression_ratio"] = (
                total_logical_bytes / file_size_bytes if file_size_bytes > 0 else 0
            )

        # Add file-level attributes if any
        with h5py.File(file_path, "r") as f:
            if f.attrs:
                file_attrs = {}
                for key in f.attrs:
                    value = f.attrs[key]
                    # Keep it compact - only simple types
                    if isinstance(value, str | bytes | int | float | bool):
                        file_attrs[key] = value
                if file_attrs:
                    result["file_attributes"] = file_attrs

        return result

    except Exception as e:
        return {"error": f"Failed to generate file overview: {str(e)}"}


def search_paths(
    tree: H5Node,
    pattern: str,
    node_type: str = "any",
    max_results: int = 50,
) -> dict[str, Any]:
    """Search for paths by name/pattern.

    Args:
        tree: Pre-parsed tree
        pattern: Substring or glob pattern to match
        node_type: Filter by "group", "dataset", or "any"
        max_results: Maximum number of results to return

    Returns:
        Dictionary with matching paths
    """
    try:
        results = []

        def search_recursive(node: H5Node) -> None:
            if len(results) >= max_results:
                return

            # Check if node type matches filter
            if node_type != "any" and node.type != node_type:
                # Still need to search children for groups
                if isinstance(node, GroupNode):
                    for child in node.children:
                        search_recursive(child)
                return

            # Check if pattern matches
            name_match = pattern.lower() in node.name.lower()
            path_match = pattern.lower() in node.path.lower()
            glob_match = fnmatch.fnmatch(node.name.lower(), pattern.lower())

            if name_match or path_match or glob_match:
                result_entry: dict[str, Any] = {
                    "path": node.path,
                    "name": node.name,
                    "type": node.type,
                }

                if isinstance(node, DatasetNode):
                    result_entry["shape"] = node.shape
                    result_entry["dtype"] = node.dtype
                    result_entry["size_mb"] = node.size_bytes / (1024 * 1024)

                results.append(result_entry)

            # Search children
            if isinstance(node, GroupNode):
                for child in node.children:
                    search_recursive(child)

        search_recursive(tree)

        return {
            "pattern": pattern,
            "node_type": node_type,
            "num_results": len(results),
            "max_results": max_results,
            "truncated": len(results) >= max_results,
            "results": results,
        }

    except Exception as e:
        return {"error": f"Failed to search paths: {str(e)}"}


def summarize_group(
    tree: H5Node,
    path: str,
    max_children: int = 20,
) -> dict[str, Any]:
    """Create a human-friendly summary of a group's contents.

    Args:
        tree: Pre-parsed tree
        path: Path to the group
        max_children: Maximum children to show

    Returns:
        Dictionary with group summary
    """
    node = _find_node(tree, path)
    if node is None:
        return {"error": f"Node not found at path: {path}"}

    if not isinstance(node, GroupNode):
        return {"error": f"Node at {path} is not a group"}

    # Count descendants
    def count_descendants(n: H5Node) -> tuple[int, int]:
        """Return (num_groups, num_datasets)."""
        if isinstance(n, DatasetNode):
            return (0, 1)
        else:
            groups = 0
            datasets = 0
            for child in n.children:
                g, d = count_descendants(child)
                groups += g + 1 if isinstance(child, GroupNode) else 0
                datasets += d
            return (groups, datasets)

    total_groups, total_datasets = count_descendants(node)

    result: dict[str, Any] = {
        "path": path,
        "num_direct_children": len(node.children),
        "num_child_groups": sum(1 for c in node.children if isinstance(c, GroupNode)),
        "num_child_datasets": sum(
            1 for c in node.children if isinstance(c, DatasetNode)
        ),
        "total_descendant_groups": total_groups,
        "total_descendant_datasets": total_datasets,
    }

    # Add description if present
    if "description" in node.attributes:
        desc = node.attributes["description"]
        if isinstance(desc, str | bytes):
            result["description"] = desc.decode() if isinstance(desc, bytes) else desc

    # Add example children
    example_children = []
    for child in node.children[:max_children]:
        child_info: dict[str, Any] = {
            "name": child.name,
            "type": child.type,
        }

        if isinstance(child, DatasetNode):
            child_info["shape"] = child.shape
            child_info["dtype"] = child.dtype
            child_info["size_mb"] = child.size_bytes / (1024 * 1024)
        elif isinstance(child, GroupNode):
            child_info["num_children"] = len(child.children)

        # Add description if present
        if "description" in child.attributes:
            desc = child.attributes["description"]
            if isinstance(desc, str | bytes):
                child_info["description"] = (
                    desc.decode() if isinstance(desc, bytes) else desc
                )[:100]

        example_children.append(child_info)

    result["example_children"] = example_children

    if len(node.children) > max_children:
        result["children_truncated"] = True
        result["children_shown"] = max_children

    return result


def search_by_attribute(
    file_path: str | Path,
    tree: H5Node,
    attr_name: str,
    value_contains: str | None = None,
    max_results: int = 50,
) -> dict[str, Any]:
    """Find nodes based on attributes.

    Args:
        file_path: Path to HDF5 file
        tree: Pre-parsed tree
        attr_name: Attribute name to search for
        value_contains: Optional substring that attribute value must contain
        max_results: Maximum results to return

    Returns:
        Dictionary with matching nodes
    """
    try:
        results = []

        def search_recursive(node: H5Node) -> None:
            if len(results) >= max_results:
                return

            # Check if node has the attribute
            if attr_name in node.attributes:
                attr_value = node.attributes[attr_name]

                # If value_contains is specified, need to check value
                if value_contains is not None:
                    # For metadata-only attributes, need to load actual value
                    if isinstance(attr_value, dict) and attr_value.get("_metadata"):
                        # Load the actual value from file
                        try:
                            with h5py.File(file_path, "r") as f:
                                if node.path in f:
                                    actual_value = f[node.path].attrs[attr_name]
                                    value_str = str(actual_value)
                                else:
                                    return
                        except Exception:
                            return
                    else:
                        value_str = str(attr_value)

                    # Check if substring matches
                    if value_contains.lower() not in value_str.lower():
                        # Continue searching children
                        if isinstance(node, GroupNode):
                            for child in node.children:
                                search_recursive(child)
                        return

                # This node matches
                result_entry: dict[str, Any] = {
                    "path": node.path,
                    "type": node.type,
                }

                # Add attribute summary
                if isinstance(attr_value, dict) and attr_value.get("_metadata"):
                    result_entry["attribute_type"] = attr_value.get("type", "unknown")
                    result_entry["attribute_value"] = "<not loaded>"
                else:
                    attr_str = str(attr_value)
                    if len(attr_str) > 200:
                        result_entry["attribute_value"] = attr_str[:200] + "..."
                    else:
                        result_entry["attribute_value"] = attr_str

                results.append(result_entry)

            # Search children
            if isinstance(node, GroupNode):
                for child in node.children:
                    search_recursive(child)

        search_recursive(tree)

        return {
            "attr_name": attr_name,
            "value_contains": value_contains,
            "num_results": len(results),
            "max_results": max_results,
            "truncated": len(results) >= max_results,
            "results": results,
        }

    except Exception as e:
        return {"error": f"Failed to search by attribute: {str(e)}"}


def check_compatibility(
    file_path: str | Path,
    a_path: str,
    b_path: str,
) -> dict[str, Any]:
    """Check compatibility of two datasets for operations.

    Args:
        file_path: Path to HDF5 file
        a_path: Path to first dataset
        b_path: Path to second dataset

    Returns:
        Dictionary with compatibility analysis
    """
    try:
        with h5py.File(file_path, "r") as f:
            if a_path not in f:
                return {"error": f"Dataset not found: {a_path}"}
            if b_path not in f:
                return {"error": f"Dataset not found: {b_path}"}

            a_dataset = f[a_path]
            b_dataset = f[b_path]

            if not isinstance(a_dataset, h5py.Dataset):
                return {"error": f"Path is not a dataset: {a_path}"}
            if not isinstance(b_dataset, h5py.Dataset):
                return {"error": f"Path is not a dataset: {b_path}"}

            a_shape = a_dataset.shape
            b_shape = b_dataset.shape
            a_dtype = str(a_dataset.dtype)
            b_dtype = str(b_dataset.dtype)

            # Check same length for 1D
            same_length_1d = False
            if len(a_shape) == 1 and len(b_shape) == 1:
                same_length_1d = a_shape[0] == b_shape[0]

            # Check broadcastable
            broadcastable = False
            try:
                np.broadcast_shapes(a_shape, b_shape)
                broadcastable = True
            except ValueError:
                pass

            # Generate notes
            notes = []
            if same_length_1d:
                notes.append(
                    f"Both datasets are 1D with length {a_shape[0]} - suitable for scatter plots, line plots, and correlation"
                )
            elif broadcastable:
                notes.append(
                    f"Shapes {a_shape} and {b_shape} are broadcastable - element-wise operations possible"
                )
            else:
                notes.append(
                    f"Shapes {a_shape} and {b_shape} are not directly compatible for element-wise operations"
                )

            # Check numeric types
            a_numeric = np.issubdtype(a_dataset.dtype, np.number)
            b_numeric = np.issubdtype(b_dataset.dtype, np.number)

            if a_numeric and b_numeric:
                notes.append(
                    "Both datasets are numeric - suitable for mathematical operations"
                )
            else:
                if not a_numeric:
                    notes.append(f"{a_path} is not numeric (dtype: {a_dtype})")
                if not b_numeric:
                    notes.append(f"{b_path} is not numeric (dtype: {b_dtype})")

            return {
                "a_path": a_path,
                "b_path": b_path,
                "a_shape": a_shape,
                "b_shape": b_shape,
                "a_dtype": a_dtype,
                "b_dtype": b_dtype,
                "same_length_1d": same_length_1d,
                "broadcastable": broadcastable,
                "both_numeric": a_numeric and b_numeric,
                "notes": " | ".join(notes),
            }

    except Exception as e:
        return {"error": f"Failed to check compatibility: {str(e)}"}


# Tool definitions for LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_node_info",
            "description": "Get detailed information about a specific node (group or dataset) in the HDF5 file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the node in the HDF5 file (e.g., '/gas/temperature')",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_children",
            "description": "List all children of a group in the HDF5 file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the group (default: '/')",
                        "default": "/",
                    },
                },
                "required": [],
            },
        },
    },
]
