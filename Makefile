.PHONY: install data baselines gbm test lint clean

install:
	uv sync --extra dev

data:
	uv run python scripts/prepare_data.py

baselines:
	uv run python scripts/run_baselines.py

gbm:
	uv run python scripts/run_gbm.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts

clean:
	rm -rf data/processed results
