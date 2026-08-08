from collections.abc import Sequence

from system_tai.preliminary.validation import validate_ranked_top100

from .models import TRAKEEventCandidate, TRAKEQuery, TRAKEResult
from .planner import plan_trake_paths


class TRAKEEngine:
    """Engine for TRAKE (Temporal Retrieval-Augmented Keyframe Event) planning."""

    def solve_query(
        self,
        query: TRAKEQuery,
        event_candidates: Sequence[Sequence[TRAKEEventCandidate]],
        beam_width: int = 100,
        output_top_k: int = 100,
        rrf_constant: float = 60.0,
    ) -> TRAKEResult:
        predictions, diagnostics = plan_trake_paths(
            query=query,
            event_candidates=event_candidates,
            beam_width=beam_width,
            output_top_k=output_top_k,
            rrf_constant=rrf_constant,
        )

        expected_event_count = len(query.events)
        for pred in predictions:
            if len(pred.frame_ids) != expected_event_count:
                raise ValueError(
                    f"Prediction frame_ids count ({len(pred.frame_ids)}) != "
                    f"query event count ({expected_event_count})"
                )

        val_errors = validate_ranked_top100(
            list(predictions),
            expected_task="trake",
            expected_query_id=query.query_id,
        )
        if val_errors:
            err_msgs = "; ".join(e.message for e in val_errors)
            raise ValueError(
                f"TRAKE prediction validation failed for query {query.query_id}: {err_msgs}"
            )

        return TRAKEResult(
            query_id=query.query_id,
            event_count=len(query.events),
            predictions=predictions,
            diagnostics=diagnostics,
        )
