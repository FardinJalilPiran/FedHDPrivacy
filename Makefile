.PHONY: help install dev test lint format smoke run clean

help:
	@echo "install  Install the package"
	@echo "dev      Install with all optional and development dependencies"
	@echo "test     Run the test suite"
	@echo "lint     Check formatting and lint rules"
	@echo "format   Auto-fix formatting and lint rules"
	@echo "smoke    Offline 3-round run on synthetic data"
	@echo "run      Full run with configs/default.yaml (downloads UCI HAR)"
	@echo "clean    Remove caches and build artefacts"

install:
	pip install -e .

dev:
	pip install -e ".[all]"

test:
	pytest

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

format:
	ruff format src tests scripts
	ruff check --fix src tests scripts

smoke:
	fedhdprivacy --config configs/quick.yaml

run:
	fedhdprivacy --config configs/default.yaml

clean:
	rm -rf build dist .pytest_cache .ruff_cache **/__pycache__ *.egg-info src/*.egg-info
