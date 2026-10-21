UV ?= uv
PYTEST_ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help setup data-dirs check lint fix test test-unit test-scripts

help:
	@printf '%s\n' \
	  'make setup        Install locked dependencies and create local data folders' \
	  'make check        Run lint and the complete automated test suite' \
	  'make lint         Check Python code with Ruff' \
	  'make fix          Apply Ruff autofixes' \
	  'make test         Run all automated tests' \
	  'make test-unit    Run application/domain tests' \
	  'make test-scripts Run operator-tool tests' \
	  'make data-dirs    Create local data folders without changing existing files' \
	  'Append PYTEST_ARGS="-k name -x" to narrow a test run.'

setup: data-dirs
	$(UV) sync --locked

data-dirs:
	mkdir -p data/history data/live/clips data/live/debug data/reference_bg \
	  data/reports/investigations data/reports/reviews data/reports/debug \
	  data/reports/scratch data/reports/archive data/backups/database data/backups/labels data/logs

check: lint test

lint:
	$(UV) run ruff check .

fix:
	$(UV) run ruff check . --fix

test:
	$(UV) run pytest -q $(PYTEST_ARGS)

test-unit:
	$(UV) run pytest -q tests/unit $(PYTEST_ARGS)

test-scripts:
	$(UV) run pytest -q tests/scripts $(PYTEST_ARGS)
