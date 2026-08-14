"""Unit tests for system_tai FastAPI server endpoints."""

from fastapi.testclient import TestClient

from system_tai.server.app import create_app


def test_server_health_check() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ready", "ready_mock")
    assert "KIS" in data["active_tasks"]
    assert "Q&A" in data["active_tasks"]
    assert "TRAKE" in data["active_tasks"]


def test_server_kis_search() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/api/v1/kis/search",
        json={"query": "A person riding a water buffalo", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "A person riding a water buffalo"
    assert "candidates" in data
    assert len(data["candidates"]) > 0
    candidate = data["candidates"][0]
    assert "videoId" in candidate
    assert "frameId" in candidate
    assert "timestamp" in candidate
    assert "score" in candidate


def test_server_qa_ask() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/api/v1/qa/ask",
        json={
            "event_description": "Cuộc đua trong bùn",
            "question": "Cuộc đua trong bùn sử dụng con vật nào?",
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "answers" in data
    assert len(data["answers"]) > 0
    answer = data["answers"][0]
    assert "videoId" in answer
    assert "answer" in answer
    assert "confidence" in answer


def test_server_trake_query() -> None:
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/api/v1/trake/query",
        json={
            "events": [
                "A person approaches a bus stop",
                "boards the bus",
                "the bus leaves",
            ],
            "top_k": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "chains" in data
    assert len(data["chains"]) > 0
    chain = data["chains"][0]
    assert "videoId" in chain
    assert "frames" in chain
    assert len(chain["frames"]) == 3
