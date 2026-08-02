.PHONY: install lint test demo check

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -q

demo:
	python scripts/demo_pipeline.py --config configs/experiments/exp001_template.yaml

check: lint test

