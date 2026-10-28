#!/usr/bin/env bash
# Stage 2: Size Estimation Server Bringup on Remote SSH GPU Server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [ -f "$VENV_PYTHON" ]; then
    PYTHON_EXEC="$VENV_PYTHON"
else
    PYTHON_EXEC="python3"
fi

# Locate cuDNN libraries if installed in venv
CUDNN_LIB_DIR="$("$PYTHON_EXEC" -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0] + '/lib')" 2>/dev/null || true)"
if [ -n "$CUDNN_LIB_DIR" ] && [ -d "$CUDNN_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

echo "====================================================================="
echo " TurtleBot 4 Stage 2 SSH Size Estimation Server Bringup"
echo " Python: $PYTHON_EXEC"
echo "====================================================================="

exec "$PYTHON_EXEC" "${SCRIPT_DIR}/size_server.py" "$@"

