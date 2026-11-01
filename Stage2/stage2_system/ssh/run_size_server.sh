#!/usr/bin/env bash
# Stage 2: Size Estimation Server Bringup on Remote SSH GPU Server
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [ -x "$VENV_PYTHON" ]; then
    PYTHON_EXEC="$VENV_PYTHON"
else
    PYTHON_EXEC="python3"
fi

# Locate pip-installed CUDA / cuDNN runtime libraries in venv before Python starts
NVIDIA_LIB_DIRS="$("$PYTHON_EXEC" -c "
import os
dirs = []
for mod_name in ('cudnn', 'cublas', 'cuda_runtime'):
    try:
        mod = __import__(f'nvidia.{mod_name}', fromlist=[mod_name])
        d = os.path.join(mod.__path__[0], 'lib')
        if os.path.isdir(d):
            dirs.append(d)
    except Exception:
        pass
print(':'.join(dirs))
" 2>/dev/null || true)"
if [ -n "$NVIDIA_LIB_DIRS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS:${LD_LIBRARY_PATH:-}"
fi

if ! "$PYTHON_EXEC" -c "import ultralytics" >/dev/null 2>&1; then
    echo "[WARN] 'ultralytics' is not installed in $PYTHON_EXEC (YOLO segmentation will be unavailable)."
    echo "       Install Stage 2 GPU deps with: \"$PYTHON_EXEC\" -m pip install -r \"$PROJECT_ROOT/Stage2/stage2_system/requirements-ssh.txt\""
fi

echo "====================================================================="
echo " TurtleBot 4 Stage 2 SSH Size Estimation Server Bringup"
echo " Python: $PYTHON_EXEC"
echo "====================================================================="

exec "$PYTHON_EXEC" "${SCRIPT_DIR}/size_server.py" "$@"

