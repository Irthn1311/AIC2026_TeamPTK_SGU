"""Live System Smoke Test: Verifies start_system launcher, FastAPI endpoints, and UI static mounting."""

from __future__ import annotations

from pathlib import Path

from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from system_tai.server.app import create_app


def test_live_system_and_ui_smoke() -> None:
    app = create_app()

    # Mount UI dist if present
    ui_dist = Path(__file__).resolve().parent.parent / "ui" / "dist"
    if ui_dist.exists() and (ui_dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(ui_dist), html=True), name="ui")

    client = TestClient(app)

    # 1. Health Endpoints
    live_res = client.get("/api/v1/health/live")
    assert live_res.status_code == 200
    assert live_res.json()["data"]["status"] == "live"

    ready_res = client.get("/api/v1/health/ready")
    assert ready_res.status_code == 200
    assert ready_res.json()["data"]["status"] in ("ready", "uninitialized")

    # 2. Config Endpoint
    cfg_res = client.get("/api/v1/config")
    assert cfg_res.status_code == 200
    assert "KIS" in cfg_res.json()["data"]["supported_tasks"]
    assert "Q&A" in cfg_res.json()["data"]["supported_tasks"]
    assert "TRAKE" in cfg_res.json()["data"]["supported_tasks"]

    # 3. KIS Search & Refine
    kis_res = client.post("/api/v1/kis/search", json={"query": "A blue truck", "top_k": 10})
    assert kis_res.status_code == 200
    assert "candidates" in kis_res.json()["data"]

    refine_res = client.post("/api/v1/kis/refine", json={"video_id": "L21_V001", "center_actual_frame_id": 500})
    assert refine_res.status_code == 200
    assert refine_res.json()["data"]["moment_found"] is True

    # 4. Q&A Search & Verify
    qa_res = client.post("/api/v1/qa/search", json={"event_description": "Event", "question": "What is it?", "top_k": 10})
    assert qa_res.status_code == 200
    assert "answers" in qa_res.json()["data"]

    # 5. TRAKE Search & Verify
    trake_res = client.post("/api/v1/trake/search", json={"events": ["Event 1", "Event 2"], "top_k_chains": 10})
    assert trake_res.status_code == 200
    assert "top_chains" in trake_res.json()["data"]

    # 6. Submission Validation & Export
    val_res = client.post("/api/v1/submissions/validate", json={"task_type": "KIS", "records": [{"video_id": "L21_V001", "frame_id": 500}]})
    assert val_res.status_code == 200
    assert val_res.json()["data"]["valid"] is True

    exp_res = client.post("/api/v1/submissions/export", json={"task_type": "KIS", "records": [{"video_id": "L21_V001", "frame_id": 500}]})
    assert exp_res.status_code == 200
    assert "text/csv" in exp_res.headers["content-type"]

    # 7. UI Static Asset Serving
    if ui_dist.exists() and (ui_dist / "index.html").exists():
        ui_res = client.get("/")
        assert ui_res.status_code == 200
        assert "html" in ui_res.headers["content-type"].lower()
        assert "root" in ui_res.text
