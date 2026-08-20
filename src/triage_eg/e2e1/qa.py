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


@dataclass(frozen=True)
class GroundingCandidate:
    video_id: str
    frame_id: int
    grounding_rank: int
    sources: tuple[str, ...]


@dataclass(frozen=True)
class AnswerHypothesis:
    answer: str
    answer_type: str
    evidence_sources: tuple[str, ...]
    evidence_sufficient: bool
    confidence: float | None = None


SHORT_SEMANTIC_TYPES = frozenset({"COUNT", "COLOR", "OBJECT", "PERSON", "YES_NO", "ACTION"})
TEXT_PRESERVING_TYPES = frozenset({"LOCATION_NAME", "TITLE", "QUOTE_OR_VISIBLE_TEXT", "SPEECH"})
_MID = re.compile(r"^(?:/m/)?[0-9a-z_]*\d[0-9a-z_]*$", re.I)
_PUNCTUATION_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)


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
        original = str(raw).strip()
        value = original.split("/")[-1].replace("_", " ")
        if (
            value
            and not value.isdigit()
            and not is_metadata_id(original)
            and not is_metadata_id(value.replace(" ", "_"))
            and value not in cleaned
        ):
            cleaned.append(value)
    return tuple(_candidate(value.replace(" ", "_"), en=value) for value in cleaned)


def is_metadata_id(value: str) -> bool:
    compact = str(value).strip().casefold()
    suffix = compact.removeprefix("/m/")
    return bool(
        _MID.fullmatch(suffix)
        and any(char.isdigit() for char in suffix)
        and any(char.isalpha() for char in suffix)
        and " " not in suffix
    )


def compiled_qa_intent(answer_type: str | None, question: str) -> tuple[str, str]:
    kind = str(answer_type or "OTHER").upper()
    mapping = {
        "COUNT": "OCR_NUMERIC",
        "COLOR": "COLOR",
        "OBJECT": "OBJECT",
        "PERSON": "OBJECT",
        "LOCATION_NAME": "OCR_TEXT",
        "TITLE": "OCR_TEXT",
        "QUOTE_OR_VISIBLE_TEXT": "OCR_TEXT",
        "ACTION": "ACTIVITY",
        "SPEECH": "OCR_TEXT",
        "YES_NO": "GENERIC_VISUAL",
    }
    if kind in mapping:
        return mapping[kind], "COMPILED_ANSWER_TYPE"
    return route_intent(question), "LEGACY_FALLBACK_FOR_OTHER"


def answer_policy_for_type(answer_type: str | None) -> str:
    kind = str(answer_type or "OTHER").upper()
    return "TEXT_PRESERVING" if kind in TEXT_PRESERVING_TYPES else "SHORT_SEMANTIC"


def normalize_answer_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).strip().split())


def garbage_reason(value: str, answer_type: str | None) -> str | None:
    answer = normalize_answer_text(value)
    kind = str(answer_type or "OTHER").upper()
    if not answer:
        return "EMPTY"
    if len(answer) > 100:
        return "OVER_100_CHARACTERS"
    if is_metadata_id(answer):
        return "OPENIMAGES_OR_METADATA_ID"
    if _PUNCTUATION_ONLY.fullmatch(answer):
        return "PUNCTUATION_ONLY"
    if len(answer) == 1 and not (kind == "COUNT" and answer.isdigit()):
        return "ONE_CHARACTER_JUNK"
    if kind in TEXT_PRESERVING_TYPES and re.fullmatch(r"[_|>\\/\-]*[A-Za-z]{0,2}", answer):
        return "UNSUPPORTED_ISOLATED_OCR_FRAGMENT"
    return None


def reconstruct_ocr_lines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if float(row.get("confidence", -1)) < 20:
            continue
        token = normalize_answer_text(row.get("text", ""))
        if garbage_reason(token, "QUOTE_OR_VISIBLE_TEXT"):
            continue
        key = (
            int(row.get("block_num", 0)),
            int(row.get("par_num", 0)),
            int(row.get("line_num", row.get("top", 0))),
        )
        groups.setdefault(key, []).append({**row, "text": token})
    lines = []
    for key, tokens in groups.items():
        tokens.sort(key=lambda row: (int(row.get("left", 0)), -float(row["confidence"])))
        text = normalize_answer_text(" ".join(row["text"] for row in tokens))
        confidence = sum(float(row["confidence"]) for row in tokens) / len(tokens)
        if not garbage_reason(text, "QUOTE_OR_VISIBLE_TEXT"):
            lines.append({"text": text, "confidence": confidence, "line_key": key})
    return sorted(lines, key=lambda row: (row["line_key"], -row["confidence"]))


def select_text_preserving_answer(
    rows: list[dict[str, Any]], answer_type: str
) -> tuple[str | None, dict[str, Any]]:
    lines = reconstruct_ocr_lines(rows)
    kind = str(answer_type).upper()
    selected: list[dict[str, Any]] = []
    if kind == "LOCATION_NAME":
        anchored = [
            row
            for row in lines
            if re.search(r"\b(xã|phường|huyện|tỉnh|thành phố)\b", row["text"], re.I)
        ]
        selected = anchored[:1]
    elif kind == "QUOTE_OR_VISIBLE_TEXT":
        selected = lines[:2]
    elif kind == "TITLE":
        selected = sorted(lines, key=lambda row: (-row["confidence"], -len(row["text"])))[:1]
    elif kind == "SPEECH":
        selected = []
    if not selected:
        return None, {"ocr_lines": lines[:20], "selection": "NO_SUPPORTED_CONTEXTUAL_SPAN"}
    answer = normalize_answer_text(" / ".join(row["text"] for row in selected))
    conflict = len(answer) > 100
    if conflict:
        words, bounded = answer.split(), []
        for word in words:
            candidate = " ".join([*bounded, word])
            if len(candidate) > 100:
                break
            bounded.append(word)
        answer = " ".join(bounded)
    reason = garbage_reason(answer, kind)
    return (
        None if reason else answer,
        {
            "ocr_lines": lines[:20],
            "selected_lines": selected,
            "over_100_character_conflict": conflict,
            "garbage_rejection": reason,
        },
    )


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
                        "left": int(output.get("left", [0] * len(output["text"]))[index]),
                        "top": int(output.get("top", [0] * len(output["text"]))[index]),
                        "width": int(output["width"][index]),
                        "height": int(output["height"][index]),
                        "block_num": int(output.get("block_num", [0] * len(output["text"]))[index]),
                        "par_num": int(output.get("par_num", [0] * len(output["text"]))[index]),
                        "line_num": int(output.get("line_num", [0] * len(output["text"]))[index]),
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
    "AnswerHypothesis",
    "GroundingCandidate",
    "OptionalTesseract",
    "PROMPTS",
    "QA_INTENTS",
    "VOCABULARIES",
    "dynamic_object_candidates",
    "answer_policy_for_type",
    "compiled_qa_intent",
    "garbage_reason",
    "is_metadata_id",
    "numeric_tokens",
    "normalize_answer_text",
    "reconstruct_ocr_lines",
    "route_intent",
    "score_answers",
    "select_text_preserving_answer",
]
