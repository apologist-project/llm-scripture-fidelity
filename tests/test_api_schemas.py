import json
from pathlib import Path

from scripture_fidelity.api import RunRequest, api_schema_documents


def test_committed_api_schemas_are_current():
    root = Path(__file__).resolve().parents[1]
    for name, expected in api_schema_documents().items():
        actual = json.loads((root / "schemas" / name).read_text())
        assert actual == expected


def test_research_request_example_validates():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "examples" / "research-run-request.json").read_text()
    )
    request = RunRequest.model_validate(payload)
    assert request.scenario_id == "explicit-single-john-3-16-bsb"
