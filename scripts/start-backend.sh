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
# LangGraph's own default is CORS_ALLOW_ORIGINS=*. Do not use that. The
# browser calls this process from the UI origin. Locally that is the Next
# app. On Railway set the UI's public origin (comma-separated if several),
# for example https://trip-planner-ui.up.railway.app.
if [ -z "${CORS_ALLOW_ORIGINS:-}" ]; then
  CORS_ALLOW_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
fi
export CORS_ALLOW_ORIGINS
exec langgraph dev --host 0.0.0.0 --port "$PORT" --no-reload --no-browser --allow-blocking
