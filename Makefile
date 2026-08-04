.PHONY: install lint test demo check architecture architecture-validate

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -q

demo:
	python scripts/demo_pipeline.py --config configs/experiments/exp001_template.yaml

check: lint test

architecture:
	python scripts/generate_architecture_assets.py --spec docs/architecture/architecture-spec.yaml --output-root docs/architecture

architecture-validate: architecture
	python scripts/validate_architecture_assets.py --spec docs/architecture/architecture-spec.yaml --drawio docs/architecture/TRIAGE_EG_Complete_System.drawio
	pytest -q tests/architecture
