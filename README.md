# Jranformers
Jax re-implementation of various transformer modules.

<img src="https://github.com/user-attachments/assets/bde3710e-1f54-4a18-83a7-3207b8cb4f2d" alt="jransformers" width="200"/>

## Installation

This project uses UV for package management. To get started:

1. Install UV:
```bash
pip install uv
```

2. Create a virtual environment:
```bash
uv venv .venv
```

3. Activate the environment:
```bash
source .venv/bin/activate  # On Unix/macOS
.venv\Scripts\activate     # On Windows
```

4. Install dependencies:
```bash
uv pip install .
```

For CUDA support:
```bash
uv pip install .[cuda]
```

## Usage

Run training:
```bash
uv run train
```
