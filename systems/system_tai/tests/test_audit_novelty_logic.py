import pytest
from system_tai.retrieval.multi_query import QueryLanguage, QueryVariant, QueryVariantType


def test_query_variant_instantiation():
    qv = QueryVariant(
        variant_id="QA-01:champ",
        text="test query text",
        language=QueryLanguage.ENGLISH,
        variant_type=QueryVariantType.ENGLISH_TRANSLATION,
        weight=1.0,
    )
    assert qv.variant_id == "QA-01:champ"
    assert qv.language == QueryLanguage.ENGLISH
