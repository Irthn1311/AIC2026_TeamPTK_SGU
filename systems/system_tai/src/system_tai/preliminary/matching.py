from typing import Protocol

DEFAULT_HARMLESS_PUNCTUATION = ".,!?;:"


class AnswerMatcher(Protocol):
    def match(self, prediction: str, accepted: tuple[str, ...]) -> bool: ...


class NormalizedAliasAnswerMatcher:
    def __init__(
        self,
        strip_punctuation: bool = True,
        harmless_punctuation: str = DEFAULT_HARMLESS_PUNCTUATION,
    ):
        self.strip_punctuation = strip_punctuation
        self.harmless_punctuation = harmless_punctuation

    def normalize(self, s: str) -> str:
        s = s.casefold().strip()
        s = " ".join(s.split())
        if self.strip_punctuation:
            s = s.rstrip(self.harmless_punctuation).strip()
        return s

    def match(self, prediction: str, accepted: tuple[str, ...]) -> bool:
        pred_norm = self.normalize(prediction)
        for acc in accepted:
            if pred_norm == self.normalize(acc):
                return True
        return False
