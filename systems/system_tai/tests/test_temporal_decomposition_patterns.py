"""Unit tests for Vietnamese narrative temporal decomposition patterns."""

from __future__ import annotations

import pytest

from system_tai.retrieval.semantic_query import (
    SemanticUnitRole,
    decompose_vietnamese_semantic_units,
)


def test_single_scene_with_supporting_attribute_decomposition() -> None:
    # Pattern: 1 primary action scene + 1 static attribute/count description
    query_vi = (
        "Cảnh quay một nhóm hơn 5 người xếp thành hàng tập thể dục, cùng thực hiện động tác hai tay chạm mũi chân. "
        "Trong nhóm chỉ có một người đeo kính và ba người đội nón có màu đỏ."
    )
    units = decompose_vietnamese_semantic_units(
        query_id="test-single-scene-attrs",
        query_vi=query_vi,
    )

    # Unit 0 is FULL_QUERY
    assert units[0].role is SemanticUnitRole.FULL_QUERY
    assert units[0].temporal_index is None

    # Unit 1 is the primary scene action
    assert units[1].role is SemanticUnitRole.PRIMARY_SCENE
    assert units[1].temporal_index is None

    # Unit 2 is supporting attributes
    assert units[2].role is SemanticUnitRole.SUPPORTING_ATTRIBUTE
    assert units[2].temporal_index is None


def test_two_scene_temporal_narrative_with_sau_do_boundary() -> None:
    # Pattern: Scene 1, sau đó là Scene 2
    query_vi = (
        "Đoạn phim bắt đầu bằng một bản đồ, trên đó một loại công trình thủy lợi lần lượt xuất hiện bốn lần. "
        "Sau đó chuyển sang cảnh một công trình thủy lợi lớn đang mở cửa xả nước dưới trời mưa."
    )
    units = decompose_vietnamese_semantic_units(
        query_id="test-two-scene-map-dam",
        query_vi=query_vi,
    )

    assert units[0].role is SemanticUnitRole.FULL_QUERY

    # Scene 1: temporal_index = 1
    assert units[1].role is SemanticUnitRole.TEMPORAL_SCENE
    assert units[1].temporal_index == 1

    # Scene 2: temporal_index = 2
    assert units[2].role is SemanticUnitRole.TEMPORAL_SCENE
    assert units[2].temporal_index == 2


def test_two_scene_temporal_narrative_within_single_sentence() -> None:
    # Pattern: "Cảnh người làm việc A, sau đó là cảnh B"
    query_vi = (
        "Cảnh một người cho đậu hà lan vào chảo xào cùng mực, "
        "sau đó là cảnh quay chậm cảnh người này xóc chảo để đảo đều thức ăn."
    )
    units = decompose_vietnamese_semantic_units(
        query_id="test-two-scene-cooking",
        query_vi=query_vi,
    )

    assert units[0].role is SemanticUnitRole.FULL_QUERY
    assert units[1].role is SemanticUnitRole.TEMPORAL_SCENE
    assert units[1].temporal_index == 1
    assert units[2].role is SemanticUnitRole.TEMPORAL_SCENE
    assert units[2].temporal_index == 2


def test_two_scene_temporal_narrative_gemstone_quarry() -> None:
    query_vi = (
        "Cảnh người đàn ông đang xem xét một khối đá quý, "
        "sau đó là cảnh một mỏ khai thác đá quý lộ thiên dạng bậc thang nhìn từ trên cao."
    )
    units = decompose_vietnamese_semantic_units(
        query_id="test-two-scene-gemstone",
        query_vi=query_vi,
    )

    assert units[0].role is SemanticUnitRole.FULL_QUERY
    assert units[1].role is SemanticUnitRole.TEMPORAL_SCENE
    assert units[1].temporal_index == 1
    assert units[2].role is SemanticUnitRole.TEMPORAL_SCENE
    assert units[2].temporal_index == 2
