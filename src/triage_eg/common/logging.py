"""Consistent library logging setup."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure concise process-level logging if no handlers exist."""

    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
