#!/usr/bin/env bash
# One command to write a poem. Sets up the venv on first run.
#   ./run.sh          chat with Mr. Meter in the browser
#   ./run.sh --cli    the original four-question terminal flow
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Setting up (first run only)..."
  python3 -m venv .venv
fi

# Reinstall whenever requirements.txt changes. v1 only installed when .venv was
# missing, so new dependencies never landed on an existing checkout.
STAMP=.venv/.deps-installed
if [ ! -f "$STAMP" ] || [ requirements.txt -nt "$STAMP" ]; then
  echo "Installing dependencies..."
  .venv/bin/pip install -q -r requirements.txt
  touch "$STAMP"
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env — paste your API key into it, then run this again."
  exit 1
fi

if [ "${1:-}" = "--cli" ]; then
  exec .venv/bin/python poem.py
fi

PORT="${PORT:-8000}"
echo "Mr. Meter is at http://127.0.0.1:$PORT  (Ctrl-C to stop)"
( sleep 1; open "http://127.0.0.1:$PORT" >/dev/null 2>&1 ) &
# exec so Ctrl-C reaches uvicorn directly. Bound to localhost: this process
# holds your API key and has no auth. No --reload — it would wipe live sessions.
exec .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port "$PORT"
