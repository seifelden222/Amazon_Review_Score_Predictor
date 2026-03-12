#!/usr/bin/env bash
set -euo pipefail

# run_main.sh - activates .venv if present and runs main.py with any arguments
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PYTHON="$VENV_DIR/bin/python"
else
  PYTHON="python"
fi

if [[ "$PYTHON" == */* ]]; then
  if [ ! -x "$PYTHON" ]; then
    echo "Python executable not found: $PYTHON" >&2
    exit 1
  fi
elif ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python executable not found in PATH: $PYTHON" >&2
  exit 1
fi

# Run main.py with forwarded args
exec "$PYTHON" "$(dirname "$0")/main.py" "$@"
