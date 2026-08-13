"""Team-neutral AIC2026 evaluation contracts and infrastructure."""

from .contracts import TASKS, contract_document, validate_query
from .scoring import evaluate
from .validation import validate_predictions

__all__ = [
    "TASKS",
    "contract_document",
    "evaluate",
    "validate_predictions",
    "validate_query",
]
