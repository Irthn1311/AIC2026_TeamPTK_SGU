from __future__ import annotations

import json
from pathlib import Path

import pytest

from system_tai.kis.session_engine import OperationalKISRuntime
from system_tai.kis.session_schema import SessionConfig, TRAKEQueryRequest
from system_tai.preliminary.schemas import TRAKEPrediction
from system_tai.refinement.models import SharedRawRegionRefinementConfig
from system_tai.trake.models import TRAKEEventCandidate, TRAKEResult
from system_tai.trake.runtime import TRAKERuntimeTimings


class _Pipeline:
    def __init__(self, telemetry: dict[str, object]) -> None:
        self.telemetry = telemetry

    def process_trake_query(self, request, **kwargs):
        del kwargs
        predictions = (
            TRAKEPrediction(request.query_id, 1, "V1", (10, 20)),
        )
        result = TRAKEResult(
            query_id=request.query_id,
            event_count=2,
            predictions=predictions,
            diagnostics={"refinement_requested": True, "warnings": []},
        )
        pools = (
            (TRAKEEventCandidate(request.query_id, 0, 1, "V1", 10, 0.9),),
            (TRAKEEventCandidate(request.query_id, 1, 1, "V1", 20, 0.8),),
        )
        diagnostics = {
            "event_candidate_pools": pools,
            "c1_diagnostics": {"path_count": 1},
            "c1_paths": [{"rank": 1, "video_id": "V1", "frame_ids": [10, 20]}],
            "refinement_node_records": [],
            "path_diagnostics": [],
            "flattened_variants": [],
            "shared_raw_region_refinement": self.telemetry,
        }
        return result, TRAKERuntimeTimings(), diagnostics


def _runtime(
    tmp_path: Path,
    *,
    enabled: bool,
    telemetry: dict[str, object],
) -> OperationalKISRuntime:
    runtime = object.__new__(OperationalKISRuntime)
    runtime.config = SessionConfig(
        output_root=tmp_path,
        trake_shared_raw_region_config=SharedRawRegionRefinementConfig(enabled=enabled),
    )
    runtime.output_root = tmp_path
    (tmp_path / "requests").mkdir(parents=True)
    runtime.trake_pipeline = _Pipeline(telemetry)
    runtime._seen_request_ids = set()
    runtime._request_count = 0
    runtime._successful_query_count = 0
    return runtime


def _request(request_id: str) -> TRAKEQueryRequest:
    return TRAKEQueryRequest(
        request_id=request_id,
        query_id="TR-TELEMETRY",
        events=({"description": "first"}, {"description": "second"}),
        refine_top_n=1,
    )


def _artifact(runtime: OperationalKISRuntime, response, key: str) -> dict[str, object]:
    path = runtime.output_root / response["artifacts"][key]
    return json.loads(path.read_text(encoding="utf-8"))


def test_shared_refinement_telemetry_is_persisted_without_recomputation(
    tmp_path: Path,
) -> None:
    telemetry = {
        "shared_raw_region_refinement_enabled": True,
        "refinement_candidate_node_count": 7,
        "unique_video_count": 2,
        "coarse_requested_frame_count": 101,
        "coarse_unique_requested_frame_count": 41,
        "fine_requested_frame_count": 83,
        "fine_unique_requested_frame_count": 37,
        "raw_decode_request_count_before_estimate": 19,
        "raw_decode_request_count_actual": 5,
        "decoded_frame_count_actual": 333,
        "frame_cache_hit_count": 29,
        "frame_embedding_cache_hit_count": 31,
        "coalesced_region_count": 4,
    }
    runtime = _runtime(tmp_path, enabled=True, telemetry=telemetry)
    response = runtime.handle_trake_query(_request("enabled"))

    refinement = _artifact(runtime, response, "trake_refinement_json")
    manifest = _artifact(runtime, response, "trake_request_manifest")
    predictions_path = runtime.output_root / response["artifacts"][
        "trake_predictions_jsonl"
    ]

    assert refinement["shared_raw_region_refinement"] == telemetry
    assert manifest["trake_shared_raw_region_refinement_config"] == {
        "enabled": True,
    }
    assert predictions_path.read_text(encoding="utf-8") == (
        '{"query_id": "TR-TELEMETRY", "rank": 1, "video_id": "V1", '
        '"frame_ids": [10, 20]}\n'
    )


@pytest.mark.parametrize("telemetry", [{}, {"shared_raw_region_refinement_enabled": False}])
def test_disabled_legacy_trake_remains_valid_and_manifest_omits_config(
    tmp_path: Path,
    telemetry: dict[str, object],
) -> None:
    runtime = _runtime(tmp_path, enabled=False, telemetry=telemetry)
    response = runtime.handle_trake_query(_request(f"disabled-{len(telemetry)}"))
    refinement = _artifact(runtime, response, "trake_refinement_json")
    manifest = _artifact(runtime, response, "trake_request_manifest")

    assert response["status"] == "SUCCESS"
    assert response["predictions"] == [
        {
            "query_id": "TR-TELEMETRY",
            "rank": 1,
            "video_id": "V1",
            "frame_ids": [10, 20],
        }
    ]
    assert refinement["shared_raw_region_refinement"] == telemetry
    assert "trake_shared_raw_region_refinement_config" not in manifest
