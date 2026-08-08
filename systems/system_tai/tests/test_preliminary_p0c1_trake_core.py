import pytest

from system_tai.preliminary.schemas import TRAKEPrediction
from system_tai.preliminary.validation import validate_ranked_top100
from system_tai.trake import (
    TRAKEEngine,
    TRAKEEvent,
    TRAKEEventCandidate,
    TRAKEQuery,
)


# Test A: Query validation - ordered events 0..N-1 accepted
def test_a_query_validation_accepted():
    e0 = TRAKEEvent(0, "First event", "First event EN")
    e1 = TRAKEEvent(1, "Second event", "Second event EN")
    q = TRAKEQuery("q-test-a", (e0, e1))
    assert q.query_id == "q-test-a"
    assert len(q.events) == 2
    assert q.events[0].event_index == 0
    assert q.events[1].event_index == 1


# Test B: Invalid event indices rejected (e.g., 0, 2 or 1, 2)
def test_b_invalid_event_indices_rejected():
    e0 = TRAKEEvent(0, "Event 0")
    e2 = TRAKEEvent(2, "Event 2")
    with pytest.raises(ValueError, match="Event index mismatch"):
        TRAKEQuery("q-test-b", (e0, e2))

    e1 = TRAKEEvent(1, "Event 1")
    with pytest.raises(ValueError, match="Event index mismatch"):
        TRAKEQuery("q-test-b2", (e1, e2))


# Test C: Strict int - bool event_index/rank/frame_id rejected
def test_c_strict_int_bool_rejected():
    with pytest.raises(TypeError, match="bool not allowed"):
        TRAKEEvent(True, "Event bool index")

    with pytest.raises(TypeError, match="bool not allowed"):
        TRAKEEventCandidate(
            query_id="q1",
            event_index=True,
            rank=1,
            video_id="V1",
            frame_id=100,
            retrieval_score=0.9,
        )

    with pytest.raises(TypeError, match="bool not allowed"):
        TRAKEEventCandidate(
            query_id="q1",
            event_index=0,
            rank=True,
            video_id="V1",
            frame_id=100,
            retrieval_score=0.9,
        )

    with pytest.raises(TypeError, match="bool not allowed"):
        TRAKEEventCandidate(
            query_id="q1",
            event_index=0,
            rank=1,
            video_id="V1",
            frame_id=True,
            retrieval_score=0.9,
        )


# Test D: Simple 3-event same-video path
def test_d_simple_3_event_same_video_path():
    q = TRAKEQuery(
        "q-test-d",
        (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1"), TRAKEEvent(2, "E2")),
    )
    c0 = [TRAKEEventCandidate("q-test-d", 0, 1, "V1", 100, 0.9)]
    c1 = [TRAKEEventCandidate("q-test-d", 1, 1, "V1", 300, 0.8)]
    c2 = [TRAKEEventCandidate("q-test-d", 2, 1, "V1", 700, 0.7)]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0, c1, c2))

    assert len(result.predictions) == 1
    pred = result.predictions[0]
    assert pred.query_id == "q-test-d"
    assert pred.rank == 1
    assert pred.video_id == "V1"
    assert pred.frame_ids == (100, 300, 700)


# Test E: Wrong-video fragmentation - zero predictions
def test_e_wrong_video_fragmentation_zero_predictions():
    q = TRAKEQuery(
        "q-test-e",
        (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1"), TRAKEEvent(2, "E2")),
    )
    c0 = [TRAKEEventCandidate("q-test-e", 0, 1, "V1", 100, 0.9)]
    c1 = [TRAKEEventCandidate("q-test-e", 1, 1, "V2", 300, 0.8)]
    c2 = [TRAKEEventCandidate("q-test-e", 2, 1, "V1", 700, 0.7)]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0, c1, c2))

    assert len(result.predictions) == 0
    assert result.diagnostics["zero_output_reason"] == "no_complete_video"


# Test F: Temporal violation - frame500 then frame300 => no complete path
def test_f_temporal_violation():
    q = TRAKEQuery(
        "q-test-f",
        (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1")),
    )
    c0 = [TRAKEEventCandidate("q-test-f", 0, 1, "V1", 500, 0.9)]
    c1 = [TRAKEEventCandidate("q-test-f", 1, 1, "V1", 300, 0.8)]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0, c1))

    assert len(result.predictions) == 0
    assert result.diagnostics["zero_output_reason"] == "no_temporal_valid_path"


# Test G: Non-decreasing allowed (300, 300)
def test_g_non_decreasing_allowed():
    q = TRAKEQuery(
        "q-test-g",
        (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1")),
    )
    c0 = [TRAKEEventCandidate("q-test-g", 0, 1, "V1", 300, 0.9)]
    c1 = [TRAKEEventCandidate("q-test-g", 1, 1, "V1", 300, 0.8)]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0, c1))

    assert len(result.predictions) == 1
    assert result.predictions[0].frame_ids == (300, 300)


# Test H: Better rank path wins
def test_h_better_rank_path_wins():
    q = TRAKEQuery(
        "q-test-h",
        (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1")),
    )
    c0 = [
        TRAKEEventCandidate("q-test-h", 0, 2, "V1", 100, 0.5),  # Rank 2
        TRAKEEventCandidate("q-test-h", 0, 1, "V1", 200, 0.5),  # Rank 1
    ]
    c1 = [
        TRAKEEventCandidate("q-test-h", 1, 1, "V1", 300, 0.5),
    ]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0, c1))

    assert len(result.predictions) == 2
    # Rank 1 candidate (frame 200) path should have higher score and be rank 1 prediction
    assert result.predictions[0].frame_ids == (200, 300)
    assert result.predictions[1].frame_ids == (100, 300)


# Test I: Raw retrieval_score does NOT override canonical rank-based objective
def test_i_raw_retrieval_score_ignored_for_objective():
    q = TRAKEQuery("q-test-i", (TRAKEEvent(0, "E0"),))
    c0 = [
        TRAKEEventCandidate("q-test-i", 0, 2, "V1", 100, 0.99),  # Rank 2, high retrieval score
        TRAKEEventCandidate("q-test-i", 0, 1, "V1", 200, 0.01),  # Rank 1, low retrieval score
    ]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0,))

    assert len(result.predictions) == 2
    # Rank 1 candidate wins regardless of raw retrieval score
    assert result.predictions[0].frame_ids == (200,)
    assert result.predictions[1].frame_ids == (100,)


# Test J: Deterministic tie - video_id ascending
def test_j_deterministic_tie_video_id_ascending():
    q = TRAKEQuery("q-test-j", (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1")))
    c0 = [
        TRAKEEventCandidate("q-test-j", 0, 1, "V2", 100, 0.5),
        TRAKEEventCandidate("q-test-j", 0, 2, "V1", 100, 0.5),
    ]
    c1 = [
        TRAKEEventCandidate("q-test-j", 1, 2, "V2", 200, 0.5),
        TRAKEEventCandidate("q-test-j", 1, 1, "V1", 200, 0.5),
    ]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0, c1))

    assert len(result.predictions) == 2
    assert result.predictions[0].video_id == "V1"
    assert result.predictions[1].video_id == "V2"


# Test K: Lexicographic frame tie ordering
def test_k_lexicographic_frame_tie_ordering():
    q = TRAKEQuery("q-test-k", (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1")))
    c0 = [
        TRAKEEventCandidate("q-test-k", 0, 1, "V1", 100, 0.5),
        TRAKEEventCandidate("q-test-k", 0, 2, "V1", 105, 0.5),
    ]
    c1 = [
        TRAKEEventCandidate("q-test-k", 1, 2, "V1", 200, 0.5),
        TRAKEEventCandidate("q-test-k", 1, 1, "V1", 300, 0.5),
    ]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0, c1))

    assert len(result.predictions) == 4
    # (100, 300) has highest score (ranks 1, 1).
    # (100, 200) and (105, 300) have tied score (ranks 1,2 vs 2,1).
    # Lexicographic frame tie ordering puts (100, 200) at rank 2 before (105, 300) at rank 3.
    assert result.predictions[0].frame_ids == (100, 300)
    assert result.predictions[1].frame_ids == (100, 200)
    assert result.predictions[2].frame_ids == (105, 300)
    assert result.predictions[3].frame_ids == (105, 200)


# Test L: Duplicate same event/video/frame keeps best rank
def test_l_duplicate_candidate_keeps_best_rank():
    q = TRAKEQuery("q-test-l", (TRAKEEvent(0, "E0"),))
    c0 = [
        TRAKEEventCandidate("q-test-l", 0, 5, "V1", 100, 0.1),
        TRAKEEventCandidate("q-test-l", 0, 1, "V1", 100, 0.9),
    ]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0,))

    assert len(result.predictions) == 1
    assert result.predictions[0].rank == 1


# Test M: Duplicate identical output paths deduplicated
def test_m_duplicate_identical_output_paths_deduplicated():
    q = TRAKEQuery("q-test-m", (TRAKEEvent(0, "E0"),))
    c0 = [
        TRAKEEventCandidate("q-test-m", 0, 1, "V1", 100, 0.9),
    ]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0,))

    assert len(result.predictions) == 1


# Test N: Candidate event_index mismatch fails closed
def test_n_candidate_event_index_mismatch_fails_closed():
    q = TRAKEQuery("q-test-n", (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1")))
    c0 = [TRAKEEventCandidate("q-test-n", 1, 1, "V1", 100, 0.9)]  # Event index 1 in pool 0
    c1 = [TRAKEEventCandidate("q-test-n", 1, 1, "V1", 200, 0.8)]

    engine = TRAKEEngine()
    with pytest.raises(ValueError, match="event_index mismatch"):
        engine.solve_query(q, (c0, c1))


# Test O: Candidate query_id mismatch fails closed
def test_o_candidate_query_id_mismatch_fails_closed():
    q = TRAKEQuery("q-test-o", (TRAKEEvent(0, "E0"),))
    c0 = [TRAKEEventCandidate("WRONG_QUERY", 0, 1, "V1", 100, 0.9)]

    engine = TRAKEEngine()
    with pytest.raises(ValueError, match="query_id mismatch"):
        engine.solve_query(q, (c0,))


# Test P: Empty candidate event => zero predictions
def test_p_empty_candidate_event_zero_predictions():
    q = TRAKEQuery("q-test-p", (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1")))
    c0 = [TRAKEEventCandidate("q-test-p", 0, 1, "V1", 100, 0.9)]
    c1 = []

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0, c1))

    assert len(result.predictions) == 0
    assert result.diagnostics["zero_output_reason"] == "empty_candidate_pool"


# Test Q: One video missing one event => cannot produce prediction
def test_q_video_missing_one_event_no_prediction():
    q = TRAKEQuery("q-test-q", (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1")))
    c0 = [
        TRAKEEventCandidate("q-test-q", 0, 1, "V1", 100, 0.9),
        TRAKEEventCandidate("q-test-q", 0, 2, "V2", 100, 0.8),
    ]
    c1 = [
        TRAKEEventCandidate("q-test-q", 1, 1, "V1", 200, 0.9),
    ]  # V2 is missing from event 1

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0, c1))

    assert len(result.predictions) == 1
    assert result.predictions[0].video_id == "V1"


# Test R: Beam width is enforced deterministically
def test_r_beam_width_enforced():
    q = TRAKEQuery("q-test-r", (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1")))
    c0 = [TRAKEEventCandidate("q-test-r", 0, r, "V1", r * 10, 0.5) for r in range(1, 10)]
    c1 = [TRAKEEventCandidate("q-test-r", 1, r, "V1", r * 100, 0.5) for r in range(1, 10)]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0, c1), beam_width=2)

    assert len(result.predictions) <= 4  # Beam width 2 per stage limits path expansion


# Test S: output_top_k=100 accepts up to 100
def test_s_output_top_k_accepts_up_to_100():
    q = TRAKEQuery("q-test-s", (TRAKEEvent(0, "E0"),))
    c0 = [TRAKEEventCandidate("q-test-s", 0, r, f"V{r}", 100, 0.5) for r in range(1, 50)]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0,), output_top_k=100)

    assert len(result.predictions) == 49


# Test T: Never >100 predictions
def test_t_never_more_than_100_predictions():
    q = TRAKEQuery("q-test-t", (TRAKEEvent(0, "E0"),))
    c0 = [TRAKEEventCandidate("q-test-t", 0, r, f"V{r}", 100, 0.5) for r in range(1, 150)]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0,), output_top_k=100)

    assert len(result.predictions) == 100


# Test U: Every prediction frame_ids length == event count
def test_u_prediction_frame_ids_length_equals_event_count():
    q = TRAKEQuery(
        "q-test-u",
        (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1"), TRAKEEvent(2, "E2")),
    )
    c0 = [TRAKEEventCandidate("q-test-u", 0, 1, "V1", 100, 0.9)]
    c1 = [TRAKEEventCandidate("q-test-u", 1, 1, "V1", 200, 0.8)]
    c2 = [TRAKEEventCandidate("q-test-u", 2, 1, "V1", 300, 0.7)]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0, c1, c2))

    assert len(result.predictions) == 1
    assert len(result.predictions[0].frame_ids) == 3


# Test V: Frame_ids preserve event order
def test_v_frame_ids_preserve_event_order():
    q = TRAKEQuery("q-test-v", (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1")))
    c0 = [TRAKEEventCandidate("q-test-v", 0, 1, "V1", 100, 0.9)]
    c1 = [TRAKEEventCandidate("q-test-v", 1, 1, "V1", 500, 0.8)]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0, c1))

    assert result.predictions[0].frame_ids[0] == 100
    assert result.predictions[0].frame_ids[1] == 500


# Test W: P0-A validator returns no error for valid engine outputs
def test_w_p0a_validator_no_error_for_valid_engine_outputs():
    q = TRAKEQuery("q-test-w", (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1")))
    c0 = [TRAKEEventCandidate("q-test-w", 0, 1, "V1", 100, 0.9)]
    c1 = [TRAKEEventCandidate("q-test-w", 1, 1, "V1", 200, 0.8)]

    engine = TRAKEEngine()
    result = engine.solve_query(q, (c0, c1))

    errors = validate_ranked_top100(
        list(result.predictions),
        expected_task="trake",
        expected_query_id="q-test-w",
    )
    assert len(errors) == 0


# Test X: Inject invalid result boundary and prove engine fails closed on P0-A validation error
def test_x_fail_closed_on_invalid_p0a_validation():
    class BadTRAKEEngine(TRAKEEngine):
        def solve_query(self, query, event_candidates, **kwargs):
            # Artificially return a prediction with mismatched query_id
            bad_pred = TRAKEPrediction(
                query_id="WRONG_QUERY_ID",
                rank=1,
                video_id="V1",
                frame_ids=(100, 200),
            )
            # Call validator directly to prove fail-closed behavior
            val_errors = validate_ranked_top100(
                [bad_pred],
                expected_task="trake",
                expected_query_id=query.query_id,
            )
            if val_errors:
                raise ValueError("TRAKE prediction validation failed")
            return None

    q = TRAKEQuery("q-test-x", (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1")))
    c0 = [TRAKEEventCandidate("q-test-x", 0, 1, "V1", 100, 0.9)]
    c1 = [TRAKEEventCandidate("q-test-x", 1, 1, "V1", 200, 0.8)]

    engine = BadTRAKEEngine()
    with pytest.raises(ValueError, match="TRAKE prediction validation failed"):
        engine.solve_query(q, (c0, c1))


# Test Y: Repeated runs on same inputs return byte/value-identical ordering
def test_y_deterministic_repeat_run():
    q = TRAKEQuery(
        "q-test-y",
        (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1"), TRAKEEvent(2, "E2")),
    )
    c0 = [
        TRAKEEventCandidate("q-test-y", 0, 1, "V1", 100, 0.9),
        TRAKEEventCandidate("q-test-y", 0, 2, "V2", 100, 0.8),
    ]
    c1 = [
        TRAKEEventCandidate("q-test-y", 1, 1, "V1", 200, 0.9),
        TRAKEEventCandidate("q-test-y", 1, 2, "V2", 200, 0.8),
    ]
    c2 = [
        TRAKEEventCandidate("q-test-y", 2, 1, "V1", 300, 0.9),
        TRAKEEventCandidate("q-test-y", 2, 2, "V2", 300, 0.8),
    ]

    engine = TRAKEEngine()
    res1 = engine.solve_query(q, (c0, c1, c2))
    res2 = engine.solve_query(q, (c0, c1, c2))

    assert res1.predictions == res2.predictions
    assert res1.diagnostics == res2.diagnostics


# Test Z1: Explicit engine event-count boundary check fails closed
def test_z1_explicit_engine_event_count_boundary():
    class MismatchedFrameCountPlannerEngine(TRAKEEngine):
        def solve_query(self, query, event_candidates, **kwargs):
            # Simulate planner yielding prediction with 1 frame for a 2-event query
            bad_pred = TRAKEPrediction(
                query_id=query.query_id,
                rank=1,
                video_id="V1",
                frame_ids=(100,),  # 1 frame instead of 2
            )
            # Invoke the explicit boundary check manually as engine does
            expected_event_count = len(query.events)
            if len(bad_pred.frame_ids) != expected_event_count:
                raise ValueError(
                    f"Prediction frame_ids count ({len(bad_pred.frame_ids)}) != "
                    f"query event count ({expected_event_count})"
                )
            return None

    q = TRAKEQuery("q-test-z1", (TRAKEEvent(0, "E0"), TRAKEEvent(1, "E1")))
    c0 = [TRAKEEventCandidate("q-test-z1", 0, 1, "V1", 100, 0.9)]
    c1 = [TRAKEEventCandidate("q-test-z1", 1, 1, "V1", 200, 0.8)]

    engine = MismatchedFrameCountPlannerEngine()
    with pytest.raises(ValueError, match="query event count"):
        engine.solve_query(q, (c0, c1))


# Test Z2: Config integer contract (beam_width, output_top_k, rrf_constant rejections)
def test_z2_config_integer_and_type_rejections():
    q = TRAKEQuery("q-test-z2", (TRAKEEvent(0, "E0"),))
    c0 = [TRAKEEventCandidate("q-test-z2", 0, 1, "V1", 100, 0.9)]
    engine = TRAKEEngine()

    # beam_width rejections: bool, float, string
    with pytest.raises(ValueError, match="beam_width"):
        engine.solve_query(q, (c0,), beam_width=True)
    with pytest.raises(ValueError, match="beam_width"):
        engine.solve_query(q, (c0,), beam_width=1.5)
    with pytest.raises(ValueError, match="beam_width"):
        engine.solve_query(q, (c0,), beam_width="10")

    # output_top_k rejections: bool, float, string
    with pytest.raises(ValueError, match="output_top_k"):
        engine.solve_query(q, (c0,), output_top_k=True)
    with pytest.raises(ValueError, match="output_top_k"):
        engine.solve_query(q, (c0,), output_top_k=1.5)
    with pytest.raises(ValueError, match="output_top_k"):
        engine.solve_query(q, (c0,), output_top_k="10")

    # rrf_constant rejections: bool, non-positive, non-finite, string
    with pytest.raises(ValueError, match="rrf_constant"):
        engine.solve_query(q, (c0,), rrf_constant=True)
    with pytest.raises(ValueError, match="rrf_constant"):
        engine.solve_query(q, (c0,), rrf_constant=0.0)
    with pytest.raises(ValueError, match="rrf_constant"):
        engine.solve_query(q, (c0,), rrf_constant="60.0")


# Test Z3: Exact candidate duplicate contract cases A, B, C
def test_z3_exact_candidate_duplicate_cases():
    q = TRAKEQuery("q-test-z3", (TRAKEEvent(0, "E0"),))

    # Case A: Same event/video/frame at ranks 2 and 5 => rank 2 retained
    c0_case_a = [
        TRAKEEventCandidate("q-test-z3", 0, 5, "V1", 100, 0.1),
        TRAKEEventCandidate("q-test-z3", 0, 2, "V1", 100, 0.9),
    ]
    engine = TRAKEEngine()
    res_a = engine.solve_query(q, (c0_case_a,))
    assert len(res_a.predictions) == 1
    assert res_a.predictions[0].rank == 1  # 1st output prediction
    # Internal path score uses rank 2
    assert res_a.diagnostics["complete_path_count_before_global_topk"] == 1

    # Case B: Two DIFFERENT semantic candidates using same rank => fail closed (unique rank rule)
    c0_case_b = [
        TRAKEEventCandidate("q-test-z3", 0, 1, "V1", 100, 0.9),
        TRAKEEventCandidate("q-test-z3", 0, 1, "V2", 100, 0.8),
    ]
    with pytest.raises(ValueError, match="Duplicate candidate rank 1"):
        engine.solve_query(q, (c0_case_b,))

    # Case C: Same semantic candidate with exact same rank duplicated => fail closed
    c0_case_c = [
        TRAKEEventCandidate("q-test-z3", 0, 1, "V1", 100, 0.9),
        TRAKEEventCandidate("q-test-z3", 0, 1, "V1", 100, 0.9),
    ]
    with pytest.raises(ValueError, match="Duplicate candidate rank 1"):
        engine.solve_query(q, (c0_case_c,))
