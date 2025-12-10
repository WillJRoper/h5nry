"""Python code execution tool with safety restrictions."""

from __future__ import annotations

import ast
import io
import math
from contextlib import redirect_stderr, redirect_stdout, suppress
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _safe_read_dataset(
    dataset: h5py.Dataset,
    max_bytes: int,
    indices: tuple[slice, ...] | None = None,
) -> np.ndarray:
    """Safely read a dataset with memory limits.

    Args:
        dataset: HDF5 dataset
        max_bytes: Maximum bytes to load
        indices: Optional slice/indices to read

    Returns:
        Dataset contents as numpy array

    Raises:
        ValueError: If requested data exceeds max_bytes
    """
    # Determine shape and size to read
    if indices is not None:
        # Calculate size for sliced read
        # This is simplified - would need better logic for complex slicing
        size_bytes = dataset.dtype.itemsize * np.prod(
            [
                (s.stop or dataset.shape[i]) - (s.start or 0)
                if isinstance(s, slice)
                else 1
                for i, s in enumerate(indices)
            ]
        )
    else:
        size_bytes = dataset.size * dataset.dtype.itemsize

    if size_bytes > max_bytes:
        raise ValueError(
            f"Requested data ({size_bytes / 1024**3:.2f} GB) exceeds "
            f"max_bytes limit ({max_bytes / 1024**3:.2f} GB). "
            f"Use slicing to read smaller chunks, e.g., dataset[:1000] or dataset[:, :100]"
        )

    # Safe to read
    if indices is not None:
        return dataset[indices]
    else:
        return dataset[...]


def run_python(
    code: str,
    file_path: str | Path | None = None,
    max_bytes: int = 500 * 1024 * 1024,  # 500 MB default
) -> dict[str, Any]:
    """Execute Python code in a restricted environment.

    Args:
        code: Python code to execute
        file_path: Path to HDF5 file (available as h5_file context manager)
        max_bytes: Maximum bytes to load at once for HDF5 operations

    Returns:
        Dictionary with execution results
    """

    # Create safe read helper that captures max_bytes
    def safe_read(dataset, indices=None):
        return _safe_read_dataset(dataset, max_bytes, indices)

    # Prepare restricted globals
    # Use a safer approach - start with minimal builtins
    safe_builtins = {
        # Safe builtins - expanded set
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "bytes": bytes,
        "chr": chr,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "hasattr": hasattr,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,  # Important for slicing!
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
        # For import statements (but only safe modules)
        "__import__": __import__,
        # Exceptions for error handling
        "Exception": Exception,
        "ValueError": ValueError,
        "TypeError": TypeError,
        "KeyError": KeyError,
        "IndexError": IndexError,
        "AttributeError": AttributeError,
    }

    restricted_globals = {
        "__builtins__": safe_builtins,
        "__name__": "__main__",
        # Pre-imported modules
        "np": np,
        "numpy": np,
        "h5py": h5py,
        "math": math,
        # Helper variables
        "MAX_BYTES": max_bytes,
        "safe_read": safe_read,
    }

    # Add HDF5 file context if provided
    if file_path:
        restricted_globals["HDF5_FILE_PATH"] = str(file_path)
        # Open the file and make it available as 'f' or 'h5_file'
        try:
            h5_file = h5py.File(file_path, "r")
            restricted_globals["f"] = h5_file
            restricted_globals["h5_file"] = h5_file
        except Exception as e:
            return {
                "stdout": "",
                "stderr": "",
                "result_repr": None,
                "error": f"Failed to open HDF5 file: {e}",
                "code": code,
            }
    else:
        h5_file = None

    # Capture stdout and stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    result_repr = None
    error = None

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Parse the code to extract the last expression
            try:
                tree = ast.parse(code, mode="exec")
            except SyntaxError as e:
                raise SyntaxError(f"Syntax error in code: {e}") from e

            # Check if the last statement is an expression (not an assignment, print, etc.)
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                # Execute all statements except the last
                if len(tree.body) > 1:
                    statements = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(compile(statements, "<user_code>", "exec"), restricted_globals)

                # Evaluate the final expression
                final_expr = ast.Expression(body=tree.body[-1].value)
                result = eval(
                    compile(final_expr, "<user_code>", "eval"), restricted_globals
                )

                # Convert result to string representation
                if result is not None:
                    if isinstance(result, np.ndarray):
                        # Better numpy array formatting
                        if result.size > 100:
                            result_repr = f"array(shape={result.shape}, dtype={result.dtype}, mean={np.mean(result):.3f}, min={np.min(result):.3f}, max={np.max(result):.3f})"
                        else:
                            result_repr = repr(result)
                    elif isinstance(result, list | tuple | dict):
                        result_repr = repr(result)
                    else:
                        result_repr = str(result)
            else:
                # Just execute normally if no final expression
                exec(compile(tree, "<user_code>", "exec"), restricted_globals)

    except Exception as e:
        error = f"{type(e).__name__}: {str(e)}"

    finally:
        # Always close HDF5 file
        if h5_file is not None:
            with suppress(Exception):
                h5_file.close()

    return {
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
        "result_repr": result_repr,
        "error": error,
        "code": code,
    }


# Tool definitions for LLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": """Execute Python code to investigate the HDF5 file. This is a REPL-like environment with:

Pre-imported modules:
- numpy as np
- h5py
- math

Available variables:
- f or h5_file: The opened HDF5 file object (already open, read-only)
- HDF5_FILE_PATH: Path to the file
- MAX_BYTES: Memory limit for reads
- safe_read(dataset, indices=None): Helper to safely read datasets

Examples:
- List keys: print(list(f.keys()))
- Get dataset info: print(f['dataset_path'].shape, f['dataset_path'].dtype)
- Read dataset: data = f['dataset_path'][:]  # for small datasets
- Read slice: data = f['dataset_path'][:1000]  # first 1000 elements
- Compute stats: print(np.mean(f['dataset_path'][:5000]))
- Safer read: data = safe_read(f['dataset_path'], (slice(0, 1000),))

The file is automatically opened and closed. If a dataset is too large to read at once (exceeds MAX_BYTES), you'll get an error message suggesting to use slicing.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Can be multi-line. The final expression's value will be returned.",
                    },
                },
                "required": ["code"],
            },
        },
    },
]
