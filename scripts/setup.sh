#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
REUSE_VENV=false

if [[ -z "$PYTHON_BIN" && -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
  REUSE_VENV=true
fi

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      candidate_version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      if [[ "$candidate_version" == "3.11" || "$candidate_version" == "3.12" || "$candidate_version" == "3.13" ]]; then
        PYTHON_BIN="$candidate"
        break
      fi
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11, 3.12 or 3.13 is required. Set PYTHON_BIN if it is not on PATH." >&2
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.11" && "$PYTHON_VERSION" != "3.12" && "$PYTHON_VERSION" != "3.13" ]]; then
  echo "Unsupported Python $PYTHON_VERSION. Use Python 3.11, 3.12 or 3.13." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Node.js and npm are required." >&2
  exit 1
fi
NODE_MAJOR="$(node -p 'Number(process.versions.node.split(".")[0])')"
if (( NODE_MAJOR < 18 )); then
  echo "Node.js 18 or newer is required (found $(node --version))." >&2
  exit 1
fi

if [[ "$REUSE_VENV" == true ]]; then
  echo "[1/3] Reusing Python $PYTHON_VERSION virtual environment"
else
  echo "[1/3] Creating Python $PYTHON_VERSION virtual environment"
  "$PYTHON_BIN" -m venv "$PROJECT_DIR/.venv"
fi
echo "[2/3] Installing backend dependencies"
"$PROJECT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$PROJECT_DIR/.venv/bin/pip" install -r "$PROJECT_DIR/backend/requirements.txt"
echo "[3/3] Installing frontend dependencies"
npm --prefix "$PROJECT_DIR/frontend" install

echo "CashGap Lab is ready. Run: python3 start.py"
