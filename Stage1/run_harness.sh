#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Automatically locate cuDNN library directory if installed in venv
CUDNN_LIB_DIR="$("$VENV_PYTHON" -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0] + '/lib')" 2>/dev/null || true)"
if [ -n "$CUDNN_LIB_DIR" ] && [ -d "$CUDNN_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

# Multi-GPU routing is managed dynamically by harness.py via --gpus.
# Only set CUDA_VISIBLE_DEVICES if the user explicitly provided one in their environment.

exec "$VENV_PYTHON" "$SCRIPT_DIR/harness.py" "$@"

