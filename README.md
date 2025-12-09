# H5nry

An AI assistant for investigating HDF5 files in your terminal.

H5nry (pronounced "Henry") is a Textual TUI + CLI tool that lets you chat with a Large Language Model about the contents of HDF5 files. The LLM can explore file structure, inspect metadata, compute statistics, generate plots, and optionally run Python snippets—all while respecting configurable memory limits.

## Features

- 🤖 **AI-Powered Exploration**: Chat naturally with your HDF5 files using OpenAI, Anthropic, or Google Gemini
- 📊 **Smart Data Handling**: Automatically chunks large datasets to respect memory limits
- 🎨 **Interactive TUI**: Full-screen terminal interface built with Textual
- 🔒 **Safety Levels**: Choose between tools-only or tools-plus-python execution modes
- 📈 **Built-in Analytics**: Compute statistics, generate histograms, and create plots
- ⚡ **Async Design**: Responsive UI that doesn't block during LLM calls

## Installation

```bash
# Clone the repository
git clone https://github.com/WillJRoper/h5nry.git
cd h5nry

# Install in development mode
pip install -e .

# Install pre-commit hooks (for contributors)
pre-commit install
```

## Quick Start

### 1. Login to an AI Provider

Before using H5nry, you need to configure an API key for your chosen LLM provider:

```bash
# For OpenAI
h5nry login openai

# For Anthropic
h5nry login anthropic

# For Google Gemini
h5nry login gemini
```

Alternatively, set environment variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`

### 2. Interactive Mode (TUI)

Launch the full-screen chat interface:

```bash
h5nry path/to/your/file.h5
```

Inside the TUI:
- Type your questions and press Enter
- Use `/history` to view recent code snippets
- Use `/show N` to display a specific code snippet
- Press Ctrl+C to exit

### 3. One-Shot Mode

Get a quick answer without entering the TUI:

```bash
h5nry ask path/to/file.h5 "What is the mean of /gas/temperature?"
h5nry ask data.h5 "Give me a high-level summary of this file"
```

## Configuration

H5nry stores configuration in `~/.h5nry/config.yaml`. You can customize:

### Provider Settings

```yaml
# Which LLM provider to use
provider: openai  # options: openai, anthropic, gemini

# Model name
model: gpt-4-turbo-preview

# Temperature (0.0 - 1.0)
temperature: 0.1

# Max tokens (optional)
max_tokens: null

# Enable streaming responses
stream: true
```

### Safety Settings

```yaml
# Safety level determines what tools are available
safety_level: tools_only  # options: tools_only, tools_plus_python
```

- **`tools_only`**: The LLM can only use curated HDF5 inspection, statistics, and plotting tools. No arbitrary code execution.
- **`tools_plus_python`**: In addition to curated tools, the LLM can execute Python snippets in a restricted environment.

### Memory Limits

```yaml
# Maximum data to load into memory at once (in GB)
max_data_gb: 0.5
```

All dataset-reading operations automatically chunk data to respect this limit. This prevents out-of-memory errors when working with large HDF5 files.

### Code History

```yaml
# Maximum number of executed code snippets to keep in memory
recent_code_limit: 20
```

## How It Works

### HDF5 Pre-Parsing

When you open a file, H5nry:
1. Walks the entire HDF5 tree structure
2. Builds a lightweight in-memory representation
3. Reads all attribute names and types
4. Automatically loads "description" attributes (case-insensitive)
5. Provides this context to the LLM in the system prompt

### Memory-Safe Operations

All data-reading operations (statistics, histograms, plotting) respect the `max_data_gb` limit by:
- Computing dataset sizes before reading
- Automatically chunking large datasets
- Aggregating results in streaming fashion
- Raising clear errors if an operation can't be performed within the limit

### Available Tools

The LLM has access to these tool families:

- **HDF5 Tree Tools**: List groups, inspect datasets, read attributes
- **Statistics Tools**: Compute min/max/mean/std, create histograms
- **Plotting Tools**: Generate histogram plots saved to disk
- **Python Execution** (if `safety_level: tools_plus_python`): Run small Python snippets with `numpy` and `h5py` pre-imported

## Development

### Running Tests

```bash
pytest
```

### Code Quality

This project uses:
- **ruff** for linting and formatting
- **pre-commit** for automated checks
- **pytest** for testing

```bash
# Run ruff manually
ruff check .
ruff format .

# Run pre-commit on all files
pre-commit run --all-files
```

### Project Structure

```
h5nry/
├── src/h5nry/
│   ├── __init__.py
│   ├── app.py              # Main orchestrator
│   ├── cli.py              # CLI entrypoint
│   ├── config.py           # Configuration management
│   ├── session.py          # LLM + tools orchestration
│   ├── tui.py              # Textual TUI
│   ├── data/
│   │   └── default_config.yaml
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py         # Abstract LLM client
│   │   ├── openai_client.py
│   │   ├── anthropic_client.py
│   │   └── gemini_client.py
│   └── tools/
│       ├── __init__.py
│       ├── hdf5_tree.py    # HDF5 inspection tools
│       ├── stats.py        # Statistics tools
│       ├── plotting.py     # Plotting tools
│       └── python_exec.py  # Python execution tool
├── tests/
│   ├── test_config.py
│   ├── test_hdf5_tree.py
│   └── test_stats.py
├── pyproject.toml
├── README.md
└── LICENSE
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `pre-commit run --all-files`
5. Submit a pull request

## Acknowledgments

H5nry is designed for HPC developers and scientists working with large HDF5 datasets. Inspired by modern AI coding assistants and built with [Textual](https://textual.textualize.io/).
