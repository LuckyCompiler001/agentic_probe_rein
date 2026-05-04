.PHONY: dev backend frontend install help

PYTHON ?= venv/bin/python
UVICORN ?= venv/bin/uvicorn
API_PORT ?= 8765
WEB_PORT ?= 3000

help:
	@echo "Targets:"
	@echo "  make install       — install Python + Node deps"
	@echo "  make dev           — run backend + frontend together (foreground)"
	@echo "  make backend       — run only the FastAPI server"
	@echo "  make frontend      — run only the Next.js dev server"

install:
	@test -d venv || python3 -m venv venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install fastapi 'uvicorn[standard]' python-multipart
	cd web && pnpm install

backend:
	$(UVICORN) server.app:app --port $(API_PORT) --reload

frontend:
	cd web && pnpm dev

# Run both. SIGINT (Ctrl-C) stops both via the wait/trap.
dev:
	@trap 'kill 0' INT TERM EXIT; \
	$(UVICORN) server.app:app --port $(API_PORT) --log-level info & \
	(cd web && pnpm dev) & \
	wait
