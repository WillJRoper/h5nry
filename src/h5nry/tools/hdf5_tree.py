"""HDF5 tree inspection tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import h5py


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
