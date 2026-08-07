import string
from typing import Protocol


class AnswerMatcher(Protocol):
    def match(self, prediction: str, accepted: tuple[str, ...]) -> bool: ...


class NormalizedAliasAnswerMatcher:
    def __init__(self, strip_punctuation: bool = True):
        self.strip_punctuation = strip_punctuation

    def match(self, prediction: str, accepted: tuple[str, ...]) -> bool:
        pred_norm = self._normalize(prediction)
        for acc in accepted:
            if pred_norm == self._normalize(acc):
                return True
        return False

    def _normalize(self, s: str) -> str:
        s = s.casefold().strip()
        s = " ".join(s.split())
        if self.strip_punctuation:
            for p in string.punctuation:
                s = s.rstrip(p)
        return s
