"""
Unit test for visual content filter in semantic query compiler.
"""

import pytest
from system_tai.retrieval.semantic_query import (
    SemanticQueryConfig,
    has_visual_content,
)


def test_has_visual_content():
    # Pure non-visual connectives should return False
    assert not has_visual_content("Next,")
    assert not has_visual_content("Next")
    assert not has_visual_content("Then,")
    assert not has_visual_content("Sau đó,")
    assert not has_visual_content("Tiếp theo,")
    assert not has_visual_content("sau vài giây")
    assert not has_visual_content("")

    # Short phrases with substantive visual content should return True
    assert has_visual_content("four red hats")
    assert has_visual_content("London Zoo sign")
    assert has_visual_content("rough gemstone")
    assert has_visual_content("a man sitting in a chair")
    assert has_visual_content("dam under heavy rain")
