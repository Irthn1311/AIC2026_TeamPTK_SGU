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

        self._dual_color_candidates = (
            AnswerHypothesis(
                "đỏ và trắng",
                ("đỏ và trắng", "trắng và đỏ", "red and white"),
                ("red and white colors", "red and white symbol"),
            ),
            AnswerHypothesis(
                "xanh và trắng",
                ("xanh và trắng", "trắng và xanh", "blue and white"),
                ("blue and white colors",),
            ),
            AnswerHypothesis(
                "vàng và đỏ",
                ("vàng và đỏ", "đỏ và vàng", "yellow and red"),
                ("yellow and red colors",),
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

        self._animal_candidates = (
            AnswerHypothesis(
                "hà mã",
                ("hà mã", "con hà mã", "hippo", "hippopotamus"),
                ("a hippopotamus", "a hippo birthday party"),
            ),
            AnswerHypothesis(
                "trâu",
                ("trâu", "con trâu", "water buffalo", "buffalo"),
                ("a water buffalo in mud", "buffalo race"),
            ),
            AnswerHypothesis(
                "tinh tinh",
                ("tinh tinh", "con tinh tinh", "chimpanzee", "chimp", "tinh tinh (chimpanzee)"),
                ("a chimpanzee", "chimp on screen"),
            ),
            AnswerHypothesis(
                "chó", ("chó", "con chó", "dog"), ("a dog", "dogs walking")
            ),
            AnswerHypothesis(
                "mèo", ("mèo", "con mèo", "cat"), ("a cat", "cats")
            ),
            AnswerHypothesis(
                "ngựa", ("ngựa", "con ngựa", "horse"), ("a horse", "horses racing")
            ),
            AnswerHypothesis(
                "bò", ("bò", "con bò", "cow", "cattle"), ("a cow", "cattle")
            ),
            AnswerHypothesis(
                "voi", ("voi", "con voi", "elephant"), ("an elephant", "elephants")
            ),
            AnswerHypothesis(
                "khỉ", ("khỉ", "con khỉ", "monkey"), ("a monkey", "monkeys")
            ),
            AnswerHypothesis(
                "chim", ("chim", "con chim", "bird"), ("a bird", "birds")
            ),
            AnswerHypothesis(
                "cá", ("cá", "con cá", "fish"), ("a fish", "fishes")
            ),
            AnswerHypothesis(
                "hổ", ("hổ", "con hổ", "tiger"), ("a tiger", "tiger")
            ),
            AnswerHypothesis(
                "sư tử", ("sư tử", "lion"), ("a lion", "lion")
            ),
            AnswerHypothesis(
                "gấu", ("gấu", "con gấu", "bear"), ("a bear", "bear")
            ),
        )

        self._general_object_candidates = (
            AnswerHypothesis(
                "kính lúp",
                ("kính lúp", "magnifying glass"),
                ("a boy using a magnifying glass to observe small object", "a magnifying glass"),
            ),
            AnswerHypothesis(
                "xích đu",
                ("xích đu", "swing"),
                ("a person sitting on a swing high above city", "a playground swing"),
            ),
            AnswerHypothesis(
                "cần cẩu",
                ("cần cẩu", "máy cẩu", "crane"),
                ("a large construction crane reaching into sky", "a crane"),
            ),
            AnswerHypothesis(
                "vàng miếng",
                ("vàng miếng", "thỏi vàng", "gold bar", "gold bullion"),
                ("yellow metal gold bars in plastic box", "gold bars"),
            ),
            AnswerHypothesis(
                "song sắt",
                ("song sắt", "khung cửa sắt", "iron bars", "metal bars"),
                ("a man sitting behind iron bars", "metal prison bars"),
            ),
            AnswerHypothesis(
                "taxi",
                ("taxi", "xe taxi", "cab"),
                ("a convoy of white taxi cabs", "a taxi car"),
            ),
            AnswerHypothesis(
                "điện thoại thông minh",
                ("điện thoại thông minh", "điện thoại", "smartphone"),
                ("people holding smartphone photographing sky", "a smartphone camera"),
            ),
            AnswerHypothesis(
                "xe tải",
                ("xe tải", "truck"),
                ("a large truck on the left of frame", "a cargo truck"),
            ),
            AnswerHypothesis(
                "xe buýt",
                ("xe buýt", "bus"),
                ("a passenger bus", "a bus"),
            ),
            AnswerHypothesis(
                "ô tô",
                ("ô tô", "xe ô tô", "car"),
                ("a passenger car", "cars"),
            ),
            AnswerHypothesis(
                "xe máy",
                ("xe máy", "mô tô", "motorcycle"),
                ("a motorcycle", "motorbikes"),
            ),
            AnswerHypothesis(
                "máy bay",
                ("máy bay", "airplane"),
                ("an airplane in sky", "airplane"),
            ),
            AnswerHypothesis(
                "thuyền",
                ("thuyền", "boat"),
                ("a boat on water", "boat"),
            ),
        )

        self._action_candidates = (
            AnswerHypothesis(
                "đu dây",
                ("đu dây", "zipline", "ziplining"),
                ("people moving along a cable ziplining", "zipline activity"),
            ),
            AnswerHypothesis(
                "dệt",
                ("dệt", "đan thổ cẩm", "weaving"),
                ("women weaving traditional textiles", "weaving craft"),
            ),
            AnswerHypothesis(
                "làm bài thi",
                ("làm bài thi", "viết bài", "taking exam", "writing test"),
                ("students taking exam in classroom", "writing exam"),
            ),
            AnswerHypothesis(
                "chữa cháy",
                ("chữa cháy", "dập lửa", "firefighting"),
                ("firefighters putting out a fire", "firefighting"),
            ),
            AnswerHypothesis(
                "chạy đua",
                ("chạy đua", "đua", "racing"),
                ("mud racing", "running race"),
            ),
            AnswerHypothesis(
                "phỏng vấn",
                ("phỏng vấn", "interview"),
                ("interviewing on street", "an interview"),
            ),
        )

        self._scene_candidates = (
            AnswerHypothesis(
                "phiên tòa",
                ("phiên tòa", "phòng xét xử", "courtroom", "court"),
                ("a courtroom trial setting", "courtroom with judges"),
            ),
            AnswerHypothesis(
                "đất bùn",
                ("đất bùn", "đất sạt lở", "mud", "landslide"),
                ("muddy terrain from landslide", "mud and debris"),
            ),
            AnswerHypothesis(
                "một vụ cháy rừng",
                (
                    "một vụ cháy rừng",
                    "cháy rừng",
                    "đám cháy lớn",
                    "đám cháy lớn ban đêm",
                    "wildfire",
                    "forest fire",
                ),
                ("a wildfire burning at night on screen", "large night forest fire"),
            ),
            AnswerHypothesis(
                "trường quay",
                ("trường quay", "studio"),
                ("news studio setting with presenters", "television studio"),
            ),
            AnswerHypothesis(
                "bệnh viện",
                ("bệnh viện", "hospital"),
                ("doctor examining patient in hospital", "medical clinic"),
            ),
            AnswerHypothesis(
                "công viên",
                ("công viên", "park"),
                ("outdoor park collage", "public park"),
            ),
        )

    def supports(self, question_type: QuestionType) -> bool:
        return question_type in (
            QuestionType.COLOR,
            QuestionType.COUNT,
            QuestionType.YES_NO,
            QuestionType.DIRECTION,
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

    def get_candidates_for_query(
        self,
        question_type: QuestionType,
        question_text: str,
    ) -> tuple[AnswerHypothesis, ...]:
        if question_type == QuestionType.YES_NO:
            return self._yes_no_candidates
        if question_type == QuestionType.DIRECTION:
            return self._direction_candidates
        if question_type == QuestionType.COLOR:
            q_lower = question_text.lower()
            if any(w in q_lower for w in ["hai màu", "hai mau", "two colors", "two main colors"]):
                return self._dual_color_candidates + self._color_candidates
            return self._color_candidates
        if question_type == QuestionType.COUNT:
            return self._count_candidates
        return self.get_candidates(question_type)

    def get_extended_candidates_for_query(
        self,
        question_type: QuestionType,
        question_text: str,
    ) -> tuple[AnswerHypothesis, ...]:
        q_lower = question_text.lower()
        if any(w in q_lower for w in ["con vật", "con vat", "động vật", "dong vat", "animal"]):
            return self._animal_candidates
        if any(w in q_lower for w in ["loại xe", "loai xe", "type of vehicle"]):
            return self._general_object_candidates
        if any(
            w in q_lower
            for w in [
                "vật kim loại",
                "vat kim loai",
                "dụng cụ",
                "dung cu",
                "thiết bị",
                "thiet bi",
                "vật gì",
                "vat gi",
                "ngồi",
                "ngoi",
                "object",
                "tool",
                "device",
                "vehicle",
                "sitting",
            ]
        ):
            return self._general_object_candidates
        if any(
            w in q_lower
            for w in [
                "làm gì",
                "lam gi",
                "hoạt động",
                "hoat dong",
                "nghề",
                "nghe",
                "activity",
                "craft",
                "doing",
            ]
        ):
            return self._action_candidates
        if any(
            w in q_lower
            for w in [
                "bối cảnh",
                "boi canh",
                "địa hình",
                "dia hinh",
                "cảnh gì",
                "canh gi",
                "setting",
                "terrain",
                "scene is displayed",
            ]
        ):
            return self._scene_candidates
        if question_type in (QuestionType.COUNT, QuestionType.OBJECT_COUNT) or any(
            w in q_lower for w in ["bao nhiêu", "bao nhieu", "mấy", "may", "how many"]
        ):
            return self._count_candidates
        return self.get_candidates_for_query(question_type, question_text)
