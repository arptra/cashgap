#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11 is required. Install it or set PYTHON_BIN to a compatible interpreter." >&2
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

echo "[1/3] Creating Python virtual environment"
"$PYTHON_BIN" -m venv "$PROJECT_DIR/.venv"
echo "[2/3] Installing backend dependencies"
"$PROJECT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$PROJECT_DIR/.venv/bin/pip" install -r "$PROJECT_DIR/backend/requirements.txt"
echo "[3/3] Installing frontend dependencies"
npm --prefix "$PROJECT_DIR/frontend" install

echo "CashGap Lab is ready. Run: make dev"
