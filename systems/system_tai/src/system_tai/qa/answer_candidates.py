from typing import Protocol, runtime_checkable

from .models import AnswerHypothesis
from .question_types import QuestionType


@runtime_checkable
class AnswerCandidateProvider(Protocol):
    def get_candidates(self, question_type: QuestionType) -> tuple[AnswerHypothesis, ...]:
        ...


class BaselineQuestionCandidateProvider:
    def __init__(self) -> None:
        self._color_candidates = (
            AnswerHypothesis(
                "đỏ", ("đỏ", "red"), ("red", "red color", "a red object")
            ),
            AnswerHypothesis(
                "xanh dương",
                ("xanh dương", "xanh", "blue"),
                ("blue", "blue color", "a blue object"),
            ),
            AnswerHypothesis(
                "xanh lá", ("xanh lá", "green"), ("green", "green color", "a green object")
            ),
            AnswerHypothesis(
                "vàng", ("vàng", "yellow"), ("yellow", "yellow color", "a yellow object")
            ),
            AnswerHypothesis(
                "đen", ("đen", "black"), ("black", "black color", "a black object")
            ),
            AnswerHypothesis(
                "trắng", ("trắng", "white"), ("white", "white color", "a white object")
            ),
            AnswerHypothesis(
                "cam", ("cam", "orange"), ("orange", "orange color", "an orange object")
            ),
            AnswerHypothesis(
                "tím", ("tím", "purple"), ("purple", "purple color", "a purple object")
            ),
            AnswerHypothesis(
                "hồng", ("hồng", "pink"), ("pink", "pink color", "a pink object")
            ),
            AnswerHypothesis(
                "nâu", ("nâu", "brown"), ("brown", "brown color", "a brown object")
            ),
            AnswerHypothesis(
                "xám", ("xám", "grey", "gray"), ("grey", "gray color", "a grey object")
            ),
        )

        num_words_vi = [
            "không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười"
        ]
        num_words_en = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"
        ]
        count_list = []
        for i in range(11):
            s = str(i)
            aliases = (s, num_words_vi[i], num_words_en[i])
            prompts = (s, f"{num_words_en[i]} objects", f"{i} items")
            count_list.append(AnswerHypothesis(s, aliases, prompts))
        self._count_candidates = tuple(count_list)

        self._yes_no_candidates = (
            AnswerHypothesis("có", ("có", "yes", "có phải"), ("yes", "present")),
            AnswerHypothesis("không", ("không", "no", "không có"), ("no", "absent")),
        )

        self._direction_candidates = (
            AnswerHypothesis(
                "trái", ("trái", "bên trái", "left"), ("left", "on the left side")
            ),
            AnswerHypothesis(
                "phải", ("phải", "bên phải", "right"), ("right", "on the right side")
            ),
            AnswerHypothesis(
                "thẳng",
                ("thẳng", "phía trước", "straight", "front"),
                ("straight", "in front"),
            ),
            AnswerHypothesis(
                "sau", ("sau", "phía sau", "behind", "back"), ("behind", "in the back")
            ),
        )

    def get_candidates(self, question_type: QuestionType) -> tuple[AnswerHypothesis, ...]:
        if question_type == QuestionType.COLOR:
            return self._color_candidates
        elif question_type == QuestionType.COUNT:
            return self._count_candidates
        elif question_type == QuestionType.YES_NO:
            return self._yes_no_candidates
        elif question_type == QuestionType.DIRECTION:
            return self._direction_candidates
        return ()
