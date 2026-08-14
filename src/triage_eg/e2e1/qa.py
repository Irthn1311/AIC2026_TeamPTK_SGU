"""Bounded non-VLM QA answerer for E2E-1."""

from __future__ import annotations

import re
import shutil
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np

QA_INTENTS = (
    "OCR_TEXT",
    "OCR_NUMERIC",
    "COLOR",
    "TIME_OF_DAY",
    "VIEWPOINT",
    "LOCATION",
    "GARMENT",
    "VEHICLE",
    "CONTAINER",
    "OBJECT",
    "ANIMAL",
    "ACTIVITY",
    "MATERIAL",
    "GENERIC_VISUAL",
)


@dataclass(frozen=True)
class AnswerCandidate:
    canonical_id: str
    english_clip_text: str
    vi_output: str
    en_output: str


def _candidate(value: str, vi: str | None = None, en: str | None = None) -> AnswerCandidate:
    label = value.replace("_", " ")
    return AnswerCandidate(value, label, vi or label, en or label)


VOCABULARIES: dict[str, tuple[AnswerCandidate, ...]] = {
    "COLOR": tuple(
        _candidate(en, vi)
        for en, vi in (
            ("black", "đen"),
            ("white", "trắng"),
            ("red", "đỏ"),
            ("orange", "cam"),
            ("yellow", "vàng"),
            ("green", "xanh lá"),
            ("blue", "xanh dương"),
            ("purple", "tím"),
            ("pink", "hồng"),
            ("brown", "nâu"),
            ("gray", "xám"),
            ("beige", "be"),
        )
    ),
    "TIME_OF_DAY": (_candidate("daytime", "ban ngày"), _candidate("nighttime", "ban đêm")),
    "VIEWPOINT": tuple(
        _candidate(en, vi)
        for en, vi in (
            ("aerial top-down", "từ trên cao"),
            ("eye-level", "ngang tầm mắt"),
            ("low-angle", "góc thấp"),
            ("close-up", "cận cảnh"),
        )
    ),
    "LOCATION": tuple(
        _candidate(en, vi)
        for en, vi in (
            ("indoor", "trong nhà"),
            ("outdoor", "ngoài trời"),
            ("kitchen", "nhà bếp"),
            ("laboratory", "phòng thí nghiệm"),
            ("hospital clinic", "bệnh viện hoặc phòng khám"),
            ("swimming pool", "hồ bơi"),
            ("garden farm", "vườn hoặc nông trại"),
            ("road street", "đường phố"),
            ("stage", "sân khấu"),
            ("studio", "trường quay"),
            ("river lake sea", "sông hồ hoặc biển"),
        )
    ),
    "GARMENT": tuple(
        _candidate(en, vi)
        for en, vi in (
            ("shirt", "áo"),
            ("jacket", "áo khoác"),
            ("dress", "váy"),
            ("hat", "mũ"),
            ("uniform", "đồng phục"),
            ("mask", "khẩu trang"),
        )
    ),
    "VEHICLE": tuple(
        _candidate(en, vi)
        for en, vi in (
            ("car", "ô tô"),
            ("motorcycle", "xe máy"),
            ("bicycle", "xe đạp"),
            ("bus", "xe buýt"),
            ("truck", "xe tải"),
            ("boat", "thuyền"),
            ("train", "tàu hỏa"),
            ("airplane", "máy bay"),
        )
    ),
    "CONTAINER": tuple(
        _candidate(en, vi)
        for en, vi in (
            ("bottle", "chai"),
            ("bowl", "bát"),
            ("cup", "cốc"),
            ("box", "hộp"),
            ("bag", "túi"),
            ("plate", "đĩa"),
            ("jar", "lọ"),
            ("packet", "gói"),
        )
    ),
    "ANIMAL": tuple(
        _candidate(en, vi)
        for en, vi in (
            ("dog", "chó"),
            ("cat", "mèo"),
            ("bird", "chim"),
            ("horse", "ngựa"),
            ("cow", "bò"),
            ("fish", "cá"),
        )
    ),
    "ACTIVITY": tuple(
        _candidate(en, vi)
        for en, vi in (
            ("walking", "đi bộ"),
            ("running", "chạy"),
            ("cooking", "nấu ăn"),
            ("talking", "nói chuyện"),
            ("playing", "chơi"),
            ("working", "làm việc"),
            ("swimming", "bơi"),
        )
    ),
    "MATERIAL": tuple(
        _candidate(en, vi)
        for en, vi in (
            ("wood", "gỗ"),
            ("metal", "kim loại"),
            ("plastic", "nhựa"),
            ("glass", "thủy tinh"),
            ("paper", "giấy"),
            ("fabric", "vải"),
        )
    ),
    "OBJECT": tuple(
        _candidate(en, vi)
        for en, vi in (
            ("person", "người"),
            ("phone", "điện thoại"),
            ("book", "sách"),
            ("chair", "ghế"),
            ("table", "bàn"),
            ("food", "thức ăn"),
            ("tool", "dụng cụ"),
        )
    ),
}

PROMPTS = {
    "COLOR": "The dominant color is {answer}.",
    "TIME_OF_DAY": "The scene is during {answer}.",
    "VIEWPOINT": "The camera viewpoint is {answer}.",
    "LOCATION": "The scene location is {answer}.",
    "GARMENT": "The visible garment is {answer}.",
    "VEHICLE": "The visible vehicle is {answer}.",
    "CONTAINER": "The visible container is {answer}.",
    "ANIMAL": "The visible animal is {answer}.",
    "ACTIVITY": "The visible activity is {answer}.",
    "MATERIAL": "The material is {answer}.",
    "OBJECT": "The visible object is {answer}.",
    "GENERIC_VISUAL": "The answer to the visual question is {answer}.",
}


def _plain(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in value if not unicodedata.combining(char)).replace("đ", "d")


def route_intent(question: str) -> str:
    value = _plain(question)
    rules = (
        ("OCR_NUMERIC", ("bao nhieu", "con so", "number", "how many", "what digit")),
        ("COLOR", ("mau gi", "what color", "which color", "colour")),
        ("TIME_OF_DAY", ("ban ngay", "ban dem", "day or night", "time of day")),
        ("VIEWPOINT", ("goc nhin", "goc may", "viewpoint", "camera angle")),
        ("LOCATION", ("o dau", "noi nao", "where", "location")),
        (
            "LOCATION",
            ("khu vuc nao", "boi canh nao", "tu dau", "loai khu vuc", "where", "location"),
        ),
        ("GARMENT", ("mac gi", "trang phuc", "loai ao", "garment", "wearing")),
        ("VEHICLE", ("phuong tien", "loai xe", "vehicle")),
        ("CONTAINER", ("vat chua", "dung trong", "container")),
        ("ANIMAL", ("con vat", "dong vat", "con gi", "animal")),
        (
            "ACTIVITY",
            ("dang lam gi", "hoat dong", "nghe thu cong", "mon the thao", "doing", "activity"),
        ),
        ("MATERIAL", ("chat lieu", "lam bang", "material", "made of")),
        (
            "OCR_TEXT",
            (
                "chu gi",
                "tu gi",
                "tu lon",
                "viet gi",
                "ten ",
                "thuong hieu",
                "nhan hieu",
                "what word",
                "what text",
                "read",
            ),
        ),
        (
            "COLOR",
            ("mau chu dao", "co mau", "hai mau", "what color", "which color", "colour"),
        ),
        (
            "OBJECT",
            (
                "vat gi",
                "do vat",
                "dung cu gi",
                "thiet bi gi",
                "cong trinh",
                "what object",
                "what item",
                "what is",
            ),
        ),
    )
    return next(
        (intent for intent, patterns in rules if any(x in value for x in patterns)),
        "GENERIC_VISUAL",
    )


def dynamic_object_candidates(names: list[str]) -> tuple[AnswerCandidate, ...]:
    cleaned = []
    for raw in names:
        value = str(raw).strip().split("/")[-1].replace("_", " ")
        if value and not value.isdigit() and value not in cleaned:
            cleaned.append(value)
    return tuple(_candidate(value.replace(" ", "_"), en=value) for value in cleaned)


def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"(?<!\w)[+-]?(?:\d+[.,]?\d*|[.,]\d+)(?!\w)", text)


class OptionalTesseract:
    def __init__(self) -> None:
        self.module: Any = None
        self.status = "UNAVAILABLE"
        self.last_error: str | None = None
        try:
            import pytesseract

            if shutil.which("tesseract"):
                self.module = pytesseract
                self.status = "AVAILABLE"
        except (ImportError, OSError):
            pass

    def read(self, image: np.ndarray) -> list[dict[str, Any]]:
        if self.module is None:
            return []
        try:
            output = self.module.image_to_data(
                image, output_type=self.module.Output.DICT, config="--psm 11"
            )
        except Exception as error:  # optional external binary adapter must fail open
            self.last_error = f"{type(error).__name__}: {error}"
            return []
        rows = []
        for index, text in enumerate(output.get("text", [])):
            token = str(text).strip()
            try:
                confidence = float(output["conf"][index])
                area = int(output["width"][index]) * int(output["height"][index])
            except (KeyError, TypeError, ValueError):
                continue
            if token and confidence >= 0:
                rows.append(
                    {
                        "text": token,
                        "confidence": confidence,
                        "area": area,
                        "salience": confidence * area,
                    }
                )
        return sorted(rows, key=lambda row: (-row["salience"], row["text"]))


def score_answers(
    image_embedding: np.ndarray,
    text_embeddings: np.ndarray,
    candidates: tuple[AnswerCandidate, ...],
    language: str,
) -> tuple[str, float, float]:
    image = np.asarray(image_embedding, dtype=np.float32).reshape(-1)
    texts = np.asarray(text_embeddings, dtype=np.float32)
    if image.shape != (512,) or texts.shape != (len(candidates), 512) or not candidates:
        raise ValueError("QA answer scoring requires aligned 512-dimensional embeddings")
    scores = texts @ image
    order = np.argsort(-scores, kind="stable")
    winner = candidates[int(order[0])]
    second = float(scores[int(order[1])]) if len(order) > 1 else float(scores[int(order[0])])
    answer = winner.vi_output if language == "vi" else winner.en_output
    return answer, float(scores[int(order[0])]), float(scores[int(order[0])] - second)


__all__ = [
    "AnswerCandidate",
    "OptionalTesseract",
    "PROMPTS",
    "QA_INTENTS",
    "VOCABULARIES",
    "dynamic_object_candidates",
    "numeric_tokens",
    "route_intent",
    "score_answers",
]
