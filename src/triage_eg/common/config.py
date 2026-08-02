"""YAML configuration loading and environment-variable expansion."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def resolve_environment_variables(config: Any) -> Any:
    """Recursively replace ``${NAME}`` placeholders with environment values."""

    if isinstance(config, str):

        def replace(match: re.Match[str]) -> str:
            variable = match.group(1)
            if variable not in os.environ:
                raise ValueError(f"Environment variable {variable} is not set")
            return os.environ[variable]

        return _ENV_PATTERN.sub(replace, config)
    if isinstance(config, Mapping):
        return {key: resolve_environment_variables(value) for key, value in config.items()}
    if isinstance(config, Sequence) and not isinstance(config, str | bytes):
        return [resolve_environment_variables(value) for value in config]
    return config


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and resolve environment-variable placeholders."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {config_path}")
    return resolve_environment_variables(loaded)


def validate_required_keys(config: Mapping[str, Any], required_keys: Sequence[str]) -> None:
    """Validate dotted required keys such as ``features.dimension``."""

    missing: list[str] = []
    for dotted_key in required_keys:
        current: Any = config
        for part in dotted_key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                missing.append(dotted_key)
                break
            current = current[part]
    if missing:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")
