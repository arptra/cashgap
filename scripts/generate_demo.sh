#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  echo "Dependencies are missing. Run make setup first." >&2
  exit 1
fi
cd "$PROJECT_DIR/backend"
exec "$PROJECT_DIR/.venv/bin/python" -m app.cli generate --clients 500 --months 18 --seed 42 --target-rate 0.10

