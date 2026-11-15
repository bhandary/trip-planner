FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=2024

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml ./
COPY agent.py http_app.py langgraph.json ./
COPY scripts/start-backend.sh scripts/start-backend.sh

RUN pip install --no-cache-dir . \
    && chmod +x scripts/start-backend.sh

EXPOSE 2024

# All configured MCP servers are remote streamable HTTP endpoints, so this
# image does not install Node or launch npx/stdio processes. A server that
# cannot be reached is reported as unavailable and the planner keeps going.
CMD ["scripts/start-backend.sh"]
