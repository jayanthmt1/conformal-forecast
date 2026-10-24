.PHONY: install data baselines test lint clean

install:
	uv sync --extra dev

data:
	uv run python scripts/prepare_data.py

baselines:
	uv run python scripts/run_baselines.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts

clean:
	rm -rf data/processed results
