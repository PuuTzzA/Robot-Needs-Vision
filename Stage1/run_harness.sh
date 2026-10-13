#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# Automatically locate cuDNN library directory if installed in venv
CUDNN_LIB_DIR="$("$VENV_PYTHON" -c "import os, nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__) + '/lib')" 2>/dev/null || true)"
if [ -n "$CUDNN_LIB_DIR" ] && [ -d "$CUDNN_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

# Route to first GPU by default unless user specified otherwise
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec "$VENV_PYTHON" "$SCRIPT_DIR/harness.py" "$@"

