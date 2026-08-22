UV := uv
RUN := $(UV) run
PYTEST := $(RUN) pytest

.PHONY: help format format-check lint typecheck contract architecture unit scientific integration e2e smoke test audit-all

help: ## Show this help and the full public command surface
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-28s %s\n", $$1, $$2}'

format: ## Auto-format Python sources with Ruff
	$(RUN) ruff format .

format-check: ## Verify Ruff formatting
	$(RUN) ruff format --check .

lint: ## Run Ruff lint
	$(RUN) ruff check .

typecheck: ## Run strict Pyright type checking
	$(RUN) pyright

contract: ## Verify the scientific-contract snapshot matches configs/fedorbit.yaml
	$(PYTEST) tests/unit/config -k contract -q

architecture: ## Run the repository architecture enforcement suite
	$(PYTEST) tests/architecture -q

unit: ## Run unit tests
	$(PYTEST) tests/unit -q

scientific: ## Run scientific contract tests
	$(PYTEST) tests/scientific -q

integration: ## Run integration tests
	$(PYTEST) tests/integration -q

e2e: ## Run end-to-end tests
	$(PYTEST) tests/e2e -q

smoke: ## Run the nonclaim smoke suite
	$(PYTEST) tests/smoke -q

test: ## Run the complete pytest suite
	$(PYTEST) -q

audit-all: format-check lint typecheck contract architecture unit scientific integration e2e smoke ## Run every repository quality gate
	$(PYTEST) -q
