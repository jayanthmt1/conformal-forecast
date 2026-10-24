.PHONY: install data baselines gbm conformal decision ablation shift report all test lint app clean

install:
	uv sync --extra dev --extra app

data:
	uv run python scripts/prepare_data.py

baselines:
	uv run python scripts/run_baselines.py

gbm:
	uv run python scripts/run_gbm.py

conformal:
	uv run python scripts/run_conformal.py

decision:
	uv run python scripts/run_decision.py

ablation:
	uv run python scripts/run_ablation.py

shift:
	uv run python scripts/run_shift.py

report:
	uv run python scripts/make_report.py

# the whole experiment, from raw data to the README results block
all: clean data baselines gbm conformal decision ablation shift report

test:
	uv run pytest -q

lint:
	uv run ruff check src tests scripts app

app:
	uv run --extra app streamlit run app/streamlit_app.py

clean:
	rm -rf data/processed results
