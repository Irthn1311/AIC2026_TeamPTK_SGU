import pytest
from system_tai.retrieval.query_decomposition import QueryVariants, decompose_query


def test_decompose_query_literal_only():
    q = decompose_query("Tìm video cảnh đường phố", "Find a video of street scene")
    assert q.literal == "find a video of street scene"
    assert len(q.as_list()) >= 1
    assert q.as_list()[0] == ("literal", "find a video of street scene")


def test_decompose_query_action_and_entity():
    q = decompose_query(
        "Người đàn ông làm gì sau khi bước xuống xe?",
        "What is the man doing after getting out of the car?",
    )
    assert q.literal == "what is the man doing after getting out of the car"
    variants_dict = dict(q.as_list())
    assert "literal" in variants_dict
    assert "compact_keywords" in variants_dict
    assert "man" in variants_dict["compact_keywords"]
    assert "car" in variants_dict["compact_keywords"]


def test_decompose_query_empty_fail_closed():
    q = decompose_query("", "")
    assert q.literal == ""
    assert q.entity_focused is None
    assert q.action_focused is None
    assert q.as_list() == [("literal", "")]


def test_decompose_query_count_query():
    q = decompose_query(
        "Có bao nhiêu người dẫn chương trình trong trường quay?",
        "How many presenters are visible in the television studio?",
    )
    variants_dict = dict(q.as_list())
    assert "literal" in variants_dict
    if "scene_context" in variants_dict:
        assert "studio" in variants_dict["scene_context"]
