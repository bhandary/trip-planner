.PHONY: test backend ui dev probe

test:
	.venv/bin/python -m pytest

backend:
	sh scripts/start-backend.sh

ui:
	cd ui && pnpm dev

dev:
	$(MAKE) -j2 backend ui

probe:
	.venv/bin/python scripts/probe_mcp.py
