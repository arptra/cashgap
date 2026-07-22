#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${CASHGAP_API_PORT:-8000}"
UI_PORT="${CASHGAP_UI_PORT:-5173}"
if [[ ! -x "$PROJECT_DIR/.venv/bin/uvicorn" || ! -d "$PROJECT_DIR/frontend/node_modules" ]]; then
  echo "Dependencies are missing. Run python3 start.py from the repository root." >&2
  exit 1
fi

port_in_use() {
  "$PROJECT_DIR/.venv/bin/python" - "$1" <<'PY'
import socket
import sys

with socket.socket() as probe:
    try:
        probe.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(0)
raise SystemExit(1)
PY
}

select_free_port() {
  local requested="$1"
  local selected="$requested"
  while port_in_use "$selected"; do
    selected=$((selected + 1))
  done
  if [[ "$selected" != "$requested" ]]; then
    echo "Port $requested is busy; using $selected instead." >&2
  fi
  SELECTED_PORT="$selected"
}

select_free_port "$API_PORT"
API_PORT="$SELECTED_PORT"
select_free_port "$UI_PORT"
UI_PORT="$SELECTED_PORT"

cleanup() {
  trap - INT TERM EXIT
  kill "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
  wait "${BACKEND_PID:-}" "${FRONTEND_PID:-}" 2>/dev/null || true
}
handle_signal() {
  cleanup
  exit 0
}
trap handle_signal INT TERM
trap cleanup EXIT

(
  cd "$PROJECT_DIR/backend"
  exec "$PROJECT_DIR/.venv/bin/uvicorn" app.main:app --host 127.0.0.1 --port "$API_PORT"
) &
BACKEND_PID=$!

(
  cd "$PROJECT_DIR/frontend"
  exec env CASHGAP_API_PORT="$API_PORT" npm run dev -- --host 127.0.0.1 --port "$UI_PORT" --strictPort
) &
FRONTEND_PID=$!

echo "API: http://127.0.0.1:$API_PORT/docs"
echo "UI:  http://127.0.0.1:$UI_PORT"
while kill -0 "$BACKEND_PID" 2>/dev/null && kill -0 "$FRONTEND_PID" 2>/dev/null; do
  sleep 1
done

set +e
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
  wait "$BACKEND_PID"
  RESULT=$?
else
  wait "$FRONTEND_PID"
  RESULT=$?
fi
set -e
cleanup
exit "$RESULT"
