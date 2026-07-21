#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$PROJECT_DIR/.venv/bin/pytest" ]]; then
  echo "Dependencies are missing. Run make setup first." >&2
  exit 1
fi

echo "Running backend tests"
(
  cd "$PROJECT_DIR/backend"
  "$PROJECT_DIR/.venv/bin/pytest"
)
echo "Building frontend"
npm --prefix "$PROJECT_DIR/frontend" run build

