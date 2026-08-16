#!/usr/bin/env bash
# One command to write a poem. Sets up the venv on first run.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Setting up (first run only)..."
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env — paste your API key into it, then run this again."
  exit 1
fi

exec .venv/bin/python poem.py
