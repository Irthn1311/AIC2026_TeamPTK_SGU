"""Official FP32 Grounding-DINO adapter and post-processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class GroundingDinoAdapter:
    def __init__(self, asset_root: Path, device: str = "cuda") -> None:
        self.root = Path(asset_root).resolve(strict=True)
        self.device = device
        self.processor: Any = None
        self.model: Any = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.root, local_files_only=True)
        self.model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(
                self.root, local_files_only=True, dtype=torch.float32
            )
            .to(self.device)
            .eval()
        )

    def detect(self, image: Any, prompt: str) -> list[dict[str, Any]]:
        import torch

        if self.model is None or self.processor is None:
            raise RuntimeError("DINO_ADAPTER_NOT_LOADED")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([[image.height, image.width]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=0.25,
            text_threshold=0.25,
            target_sizes=target_sizes,
        )[0]
        records = []
        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"], strict=True
        ):
            values = [float(item) for item in box.tolist()]
            if not all(map(lambda item: item == item and abs(item) != float("inf"), values)):
                raise RuntimeError("DINO_NONFINITE_BOX")
            x1, y1, x2, y2 = values
            if not (0 <= x1 <= x2 <= image.width and 0 <= y1 <= y2 <= image.height):
                raise RuntimeError("DINO_BOX_OUT_OF_BOUNDS")
            records.append({"box": values, "score": float(score), "label": str(label)})
        return records

    def unload(self) -> None:
        self.processor = self.model = None
