"""Lazy sequential heavy-model lifecycle and optional-plugin fail-open boundary."""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class PluginStatus:
    name: str
    enabled: bool
    status: str
    detail: str = ""


class SequentialModelRegistry:
    def __init__(self) -> None:
        self.active_name: str | None = None
        self.active_model: Any = None
        self.events: list[dict[str, Any]] = []

    def load(self, name: str, loader: Callable[[], Any]) -> Any:
        if self.active_model is not None:
            raise RuntimeError("FS1_SIMULTANEOUS_HEAVY_MODEL_LOAD_FORBIDDEN")
        self.active_name, self.active_model = name, loader()
        self.events.append({"event": "LOAD", "model": name})
        return self.active_model

    def unload(self) -> None:
        if self.active_model is not None:
            self.events.append({"event": "UNLOAD", "model": self.active_name})
        self.active_name, self.active_model = None, None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def repair_optional(self, name: str, attempt: Callable[[], Any]) -> PluginStatus:
        try:
            model = self.load(name, attempt)
            return PluginStatus(
                name,
                model is not None,
                "PASS" if model is not None else "DISABLED",
                "BOUNDED_REPAIR_ATTEMPT_1",
            )
        except Exception as error:  # fail-open is protocol-mandated for optional plugins
            return PluginStatus(
                name, False, f"{name.upper()}_DISABLED", f"{type(error).__name__}: {error}"
            )
        finally:
            self.unload()
