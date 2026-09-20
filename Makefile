.DEFAULT_GOAL := help
TEST_DB := postgresql://postgres:pg@localhost:55499/vcb
PY := uv run
GPU_HOST ?= host.docker.internal

help: ## List targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install: ## Install backend and frontend dependencies
	uv sync
	cd frontend && npm ci --no-audit --no-fund

up: ## Local stack: Postgres, LiveKit, API, agent, frontend, Prometheus, Grafana
	@test -f .env || cp .env.example .env
	docker compose up -d --build
	@echo "app http://localhost:3000 · api http://localhost:8080/docs · grafana http://localhost:3001"

gpu-targets: ## Point Prometheus at the GPU node: make gpu-targets GPU_HOST=<ip or hostname>
	@printf '%s' '$(GPU_HOST)' | grep -Eq '^[A-Za-z0-9.-]+$$' \
	  || { echo "GPU_HOST must be an IP or hostname, got '$(GPU_HOST)'" >&2; exit 1; }
	@{ echo '['; \
	  echo '  { "targets": ["$(GPU_HOST):8000"], "labels": { "service": "vllm" } },'; \
	  echo '  { "targets": ["$(GPU_HOST):8001"], "labels": { "service": "whisper" } },'; \
	  echo '  { "targets": ["$(GPU_HOST):8002"], "labels": { "service": "kokoro" } },'; \
	  echo '  { "targets": ["$(GPU_HOST):9400"], "labels": { "service": "dcgm" } }'; \
	  echo ']'; } > config/prometheus/gpu_targets.json
	@echo "Prometheus scrapes $(GPU_HOST) within 10 s (config/prometheus/gpu_targets.json)"

down: ## Stop the local stack
	docker compose down

dev-api: ## API with reload, outside Docker
	$(PY) uvicorn backend.interfaces.http.app:app --factory --reload --port 8080

dev-agent: ## LiveKit agent worker, outside Docker
	$(PY) python -m backend.interfaces.agent.livekit_agent dev

dev-frontend: ## Next.js dev server
	cd frontend && npm run dev

migrate: ## Apply database migrations
	$(PY) alembic upgrade head

lint: ## Ruff + mypy --strict on domain and application + tsc
	$(PY) ruff check .
	$(PY) ruff format --check .
	$(PY) mypy
	cd frontend && npx tsc --noEmit

test-db: ## Throwaway Postgres for the repository tests
	@docker start vcb-test-pg >/dev/null 2>&1 || docker run -d --name vcb-test-pg \
	  -e POSTGRES_PASSWORD=pg -e POSTGRES_DB=vcb -p 55499:5432 postgres:16-alpine >/dev/null
	@until docker exec vcb-test-pg pg_isready -U postgres >/dev/null 2>&1; do sleep 1; done
	@DATABASE_URL=$(TEST_DB) $(PY) alembic upgrade head >/dev/null

test: test-db fixtures ## Everything except paid tests; coverage gate 90% on domain+application
	TEST_DATABASE_URL=$(TEST_DB) $(PY) pytest --timeout 120 --cov --cov-report=term

test-paid: ## One real call per provider (costs cents; needs API keys)
	$(PY) pytest -m paid tests/smoke -v

e2e: fixtures ## Playwright: scripted browser call against the running stack
	cd frontend && npx playwright test

fixtures: ## Render fixture and browser-test audio (macOS `say`, or espeak-ng on Linux)
	$(PY) python scripts/make_fixtures.py e2e

wer-audio: ## Render the WER corpus (~20 MB, not committed)
	$(PY) python scripts/make_fixtures.py wer

wer: ## STT word error rate, both pipelines, German and English
	$(PY) python -m backend.interfaces.cli.loadtest wer --pipeline api --language de
	$(PY) python -m backend.interfaces.cli.loadtest wer --pipeline selfhosted --language de
	$(PY) python -m backend.interfaces.cli.loadtest wer --pipeline api --language en
	$(PY) python -m backend.interfaces.cli.loadtest wer --pipeline selfhosted --language en

baseline: ## API pipeline, one call at a time: the comparison baseline
	$(PY) python -m backend.interfaces.cli.loadtest run --pipeline api -c 1 --duration 180 \
	  --out results/api_baseline.json

benchmark: ## Self-hosted sweep 1/5/10/20/40 and on until p95 > 900 ms (run on the GPU node)
	$(PY) python -m backend.interfaces.cli.loadtest sweep --pipeline selfhosted --duration 120 \
	  --baseline results/api_baseline.json --forward

benchmark-sim: ## Same sweep against a simulated GPU: offline, for exercising the harness
	$(PY) python -m backend.interfaces.cli.loadtest sweep --pipeline simulated --duration 20 \
	  --out results/benchmark_simulated.json

.PHONY: help install up gpu-targets down dev-api dev-agent dev-frontend migrate lint test-db test test-paid \
	e2e fixtures wer-audio wer baseline benchmark benchmark-sim
