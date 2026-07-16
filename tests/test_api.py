"""Tests for the authenticated single-run HTTP API.

The eval runs against Inspect's mockllm provider and passage prefetch is
stubbed, so these tests make no network calls. They assert bearer-token auth,
request->response parity with the CLI export package, config validation
surfacing as 422, and that an API run persists nothing to results/.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripture_fidelity import runner
from scripture_fidelity.api import TOKEN_ENV_VAR, app
from scripture_fidelity.bible.base import Passage, Verse

TOKEN = "test-secret-token"
TRUTH = "For God so loved the world..."


def make_request(**overrides) -> dict:
    body = {
        "reference": {"ref": "John 3:16", "type": "well_known_single"},
        "method": "unassisted",
        "translation": {
            "id": "BSB", "name": "Berean Standard Bible", "language": "eng",
            "api": "ao_lab", "api_bible_id": "BSB", "rights": "open",
        },
        "language": "eng",
        "language_pairing_mode": "matched",
        "language_pair": ["eng", "BSB"],
        "model": {"provider": "mockllm", "model": "model"},
        "temperature": 0.0,
        "reference_set_size": [1],
    }
    body.update(overrides)
    return body


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Set the bearer token and stub passage prefetch (no network)."""
    monkeypatch.setenv(TOKEN_ENV_VAR, TOKEN)

    async def fake_prefetch(config, service):
        passage = Passage(
            reference="John 3:16", translation_id="BSB",
            verses=[Verse(chapter=3, number=16, text=TRUTH)],
            retrieved_at="2026-07-16T00:00:00+00:00",
        )
        return {
            t.source_key: {r.ref: passage for r in config.references}
            for _, t in config.variant_pairs()
        }

    monkeypatch.setattr(runner, "prefetch_passages", fake_prefetch)


@pytest.fixture
def client():
    return TestClient(app)


def auth(token=TOKEN) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_healthz_needs_no_auth(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_missing_token_is_401(client):
    resp = client.post("/v1/runs", json=make_request())
    assert resp.status_code == 401


def test_wrong_token_is_401(client):
    resp = client.post("/v1/runs", json=make_request(), headers=auth("nope"))
    assert resp.status_code == 401


def test_invalid_config_is_422(client):
    # buffer_transform_selection requires a per-reference description.
    resp = client.post(
        "/v1/runs",
        json=make_request(method="buffer_transform_selection"),
        headers=auth(),
    )
    assert resp.status_code == 422
    assert "description" in resp.json()["detail"]


def test_run_returns_full_package(client):
    resp = client.post("/v1/runs", json=make_request(), headers=auth())
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] in ("completed", "completed_with_errors")
    assert body["run_id"]
    assert isinstance(body["duration_seconds"], (int, float))
    # Full parity with the CLI export package + recomputed report.
    for key in (
        "manifest", "trials", "source_fixtures",
        "method_configs", "scoring_config", "report",
    ):
        assert key in body

    manifest = body["manifest"]
    assert manifest["schema_version"] == "1"
    assert manifest["requested_models"] == ["mockllm/model"]
    assert manifest["counts"]["trial_rows"] == len(body["trials"])
    assert body["trials"][0]["reference"] == "John 3:16"
    assert body["trials"][0]["fixture_id"]
    assert body["source_fixtures"][0]["text"] == TRUTH
    assert body["method_configs"]["unassisted"] == {
        "tool": None, "transform": None
    }
    assert "exact" in body["scoring_config"]["metrics"]
    assert body["report"]["metrics"]


def test_reference_accepts_array(client):
    resp = client.post(
        "/v1/runs",
        json=make_request(
            reference=[
                {"ref": "John 3:16"},
                {"ref": "Psalm 117"},
            ],
            reference_set_size=[1, 2],
        ),
        headers=auth(),
    )
    assert resp.status_code == 200


def test_run_persists_nothing_to_results(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resp = client.post("/v1/runs", json=make_request(), headers=auth())
    assert resp.status_code == 200
    assert not (tmp_path / "results").exists()
    assert not list(tmp_path.glob("sf-run-*"))
