#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ -x .venv/bin/langgraph ]; then
  PATH="$(pwd)/.venv/bin:$PATH"
  export PATH
fi
# langgraph.json points at .env. Create an empty file when it is missing so
# the server can start. Do not copy .env.example here: that file contains
# blank keys, and loading them would clobber variables injected by Railway.
if [ ! -f .env ]; then
  touch .env
fi
PORT="${PORT:-2024}"
exec langgraph dev --host 0.0.0.0 --port "$PORT" --no-reload --no-browser --allow-blocking
