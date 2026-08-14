"""Unit tests for system_tai FastAPI server endpoints conforming to Sheet 09."""

from fastapi.testclient import TestClient

from system_tai.server.app import create_app


def test_server_health_live() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    res = response.json()
    assert "meta" in res
    assert "data" in res
    assert res["meta"]["api_contract_version"] == "1.0"
    assert res["data"]["status"] == "live"


def test_server_health_ready() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    res = response.json()
    assert res["data"]["status"] in ("ready", "uninitialized")


def test_server_config() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    res = response.json()
    assert "KIS" in res["data"]["supported_tasks"]
    assert "Q&A" in res["data"]["supported_tasks"]
    assert "TRAKE" in res["data"]["supported_tasks"]


def test_server_kis_search_empty_when_no_engine() -> None:
    app = create_app(engine=None)
    client = TestClient(app)
    search_res = client.post(
        "/api/v1/kis/search",
        json={"query": "A person riding a water buffalo", "top_k": 5},
    )
    assert search_res.status_code == 200
    res_data = search_res.json()["data"]
    # Without engine/dataset, should return 0 candidates (no dummy data)
    assert res_data["total_candidates"] == 0
    assert len(res_data["candidates"]) == 0


def test_server_kis_refine() -> None:
    app = create_app()
    client = TestClient(app)
    refine_res = client.post(
        "/api/v1/kis/refine",
        json={
            "video_id": "L21_V001",
            "center_actual_frame_id": 4592,
        },
    )
    assert refine_res.status_code == 200
    ref_data = refine_res.json()["data"]
    assert ref_data["moment_found"] is True
    assert ref_data["recommended_frame"] == 4592


def test_server_qa_search_empty_when_no_engine() -> None:
    app = create_app(engine=None)
    client = TestClient(app)
    search_res = client.post(
        "/api/v1/qa/search",
        json={
            "event_description": "Cuộc đua trong bùn",
            "question": "Cuộc đua trong bùn sử dụng con vật nào?",
            "top_k": 5,
        },
    )
    assert search_res.status_code == 200
    res_data = search_res.json()["data"]
    assert res_data["total_candidates"] == 0
    assert len(res_data["answers"]) == 0


def test_server_qa_verify() -> None:
    app = create_app()
    client = TestClient(app)
    ver_res = client.post(
        "/api/v1/qa/verify",
        json={
            "video_id": "L21_V001",
            "actual_frame_id": 4592,
            "question": "Cuộc đua trong bùn sử dụng con vật nào?",
            "canonical_answer": "Trâu",
        },
    )
    assert ver_res.status_code == 200
    assert ver_res.json()["data"]["supported"] is True


def test_server_trake_search_empty_when_no_engine() -> None:
    app = create_app(engine=None)
    client = TestClient(app)
    search_res = client.post(
        "/api/v1/trake/search",
        json={
            "events": [
                "A person approaches a bus stop",
                "boards the bus",
                "the bus leaves",
            ],
            "top_k": 5,
        },
    )
    assert search_res.status_code == 200
    res_data = search_res.json()["data"]
    assert len(res_data["chains"]) == 0


def test_server_trake_verify() -> None:
    app = create_app()
    client = TestClient(app)
    ver_res = client.post(
        "/api/v1/trake/verify",
        json={
            "video_id": "L21_V001",
            "events": ["E1", "E2", "E3"],
            "actual_frame_ids": [300, 450, 600],
        },
    )
    assert ver_res.status_code == 200
    assert ver_res.json()["data"]["valid"] is True


def test_server_submission_validate_and_export() -> None:
    app = create_app()
    client = TestClient(app)
    # Validate
    val_res = client.post(
        "/api/v1/submissions/validate",
        json={
            "task_type": "KIS",
            "records": [
                {"video_id": "L21_V001", "frame_id": 120},
                {"video_id": "L21_V002", "frame_id": 350},
            ],
        },
    )
    assert val_res.status_code == 200
    assert val_res.json()["data"]["valid"] is True

    # Export
    exp_res = client.post(
        "/api/v1/submissions/export",
        json={
            "task_type": "KIS",
            "records": [
                {"video_id": "L21_V001", "frame_id": 120},
                {"video_id": "L21_V002", "frame_id": 350},
            ],
        },
    )
    assert exp_res.status_code == 200
    assert "L21_V001,120" in exp_res.text
    assert "L21_V002,350" in exp_res.text
