"""Authenticated single-run HTTP API for the scripture-fidelity study.

Exposes one endpoint, ``POST /v1/runs``, that executes a single study run for
one parameter permutation and returns the full result package (the same data
the CLI writes to ``export/`` plus the recomputed report). Nothing is written
to a persistent results directory: unlike the CLI, an API run is ephemeral and
the caller owns the returned data.

Authentication is a single bearer token read from the environment
(``ENDPOINT_API_TOKEN``); the server refuses to start without it.

Run locally with ``scripture-fidelity-serve`` (see :func:`serve`).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hmac
import os
from typing import Union
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from scripture_fidelity.config import ConfigError, VALID_APIS, VALID_PROVIDERS

TOKEN_ENV_VAR = "ENDPOINT_API_TOKEN"
TOKEN_ENV_ALIAS = "ENDPOINT_API_KEY"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReferenceModel(StrictModel):
    ref: str
    type: str | None = None
    description: str = ""


class TranslationModel(StrictModel):
    id: str
    language: str
    api: str
    api_bible_id: str
    name: str = ""
    rights: str = "unknown"
    verification: str = ""
    edition: str = ""
    license_basis: str = ""
    public_release: bool = False


class ModelModel(StrictModel):
    provider: str
    model: str
    supports_temperature: bool = True


CONDITION_METHODS = {
    "native_parametric_quote": "unassisted",
    "source_supplied_quote": "rag",
    "tool_call_quote": "tool_call",
    "deterministic_render_given_reference": "buffer_transform",
    "reference_token_then_replace": "buffer_transform_selection",
    "web_search_quote": "web_search",
}


class RunRequest(StrictModel):
    """A single study permutation. ``reference`` accepts one object or a list
    (a list makes multi-reference set sizes such as [1, 3] meaningful)."""

    reference: Union[ReferenceModel, list[ReferenceModel]]
    method: str | None = None
    condition: str | None = None
    translation: TranslationModel
    language: str = "eng"
    language_pairing_mode: str = "matched"
    language_pair: list[str]
    model: ModelModel
    temperature: float | None = None
    reference_set_size: list[int] = Field(default_factory=lambda: [1])
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    scenario_id: str = ""
    prompt: str = ""
    protocol_version: str = ""
    protocol_role: str = "diagnostic"
    repetition: int = Field(default=1, ge=1)
    source_fixture_id: str = ""
    source_document: str = ""

    def references(self) -> list[ReferenceModel]:
        r = self.reference
        return r if isinstance(r, list) else [r]

    def resolved_method(self) -> str:
        if self.condition:
            return CONDITION_METHODS[self.condition]
        assert self.method is not None
        return self.method

    @model_validator(mode="after")
    def validate_research_request(self):
        if not self.method and not self.condition:
            raise ValueError("one of method or condition is required")
        if self.condition and self.condition not in CONDITION_METHODS:
            raise ValueError(f"unsupported condition: {self.condition}")
        if self.method and self.condition:
            expected = CONDITION_METHODS[self.condition]
            if self.method != expected:
                raise ValueError(
                    f"method {self.method!r} does not implement condition "
                    f"{self.condition!r} (expected {expected!r})"
                )
        if len(self.reference_set_size) != 1:
            raise ValueError("single-run API requires exactly one reference_set_size")
        if self.reference_set_size[0] != len(self.references()):
            raise ValueError(
                "reference_set_size must equal the number of supplied references"
            )
        if self.source_document and self.resolved_method() != "rag":
            raise ValueError("source_document is allowed only for source_supplied_quote/rag")
        if self.source_document and len(self.references()) != 1:
            raise ValueError("source_document currently supports one reference per request")
        if self.prompt and self.resolved_method() == "rag" and not self.source_document:
            raise ValueError(
                "caller-supplied source_supplied_quote/rag prompts require "
                "source_document"
            )
        if self.prompt and self.language != "eng":
            raise ValueError(
                "caller-supplied prompts currently support English only; "
                "omit prompt to use the committed localized templates"
            )
        return self


class RunResponse(StrictModel):
    """Release-safe result package returned for one research request."""

    status: str
    request_id: str
    scenario_id: str | None = None
    condition_requested: str | None = None
    condition_executed: str | None = None
    method_executed: str
    protocol_version: str | None = None
    protocol_role: str
    repetition: int
    system_version: str
    run_id: str
    duration_seconds: float
    manifest: dict
    trials: list[dict]
    source_fixtures: list[dict]
    method_configs: dict
    scoring_config: dict
    report: dict


def api_schema_documents() -> dict[str, dict]:
    """Committed API schemas keyed by their public filenames."""
    return {
        "research-run-request.schema.json": RunRequest.model_json_schema(),
        "research-run-response.schema.json": RunResponse.model_json_schema(),
    }


def _require_token() -> str:
    token = (
        os.environ.get(TOKEN_ENV_VAR, "").strip()
        or os.environ.get(TOKEN_ENV_ALIAS, "").strip()
    )
    if not token:
        raise RuntimeError(
            f"Neither {TOKEN_ENV_VAR} nor {TOKEN_ENV_ALIAS} is set; the API "
            "refuses to start without a bearer token."
        )
    return token


async def verify_bearer(authorization: str = Header(default="")) -> None:
    """Constant-time bearer-token check against ``ENDPOINT_API_TOKEN``."""
    expected = _require_token()
    scheme, _, presented = authorization.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(
        presented.strip(), expected
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


app = FastAPI(
    title="scripture-fidelity single-run API",
    version="1",
    description="Execute one study permutation and return the full package.",
)


@app.get("/healthz")
async def healthz() -> dict:
    """Unauthenticated readiness probe."""
    return {"status": "ok"}


@app.get("/version")
async def version() -> dict:
    """Unauthenticated immutable build and schema identity."""
    from scripture_fidelity.provenance import build_identity

    return {
        "status": "ok",
        **build_identity(),
        "supported_conditions": CONDITION_METHODS,
        "supported_methods": sorted(set(CONDITION_METHODS.values())),
        "supported_source_providers": list(VALID_APIS),
        "supported_model_providers": list(VALID_PROVIDERS),
        "request_schema": "/openapi.json",
    }


# Study dimensions that map from the request payload to load_config env vars.
# Provider/Bible-API credentials are read from the process environment (.env)
# and are deliberately not part of the request.
_STUDY_ENV_VARS = (
    "REFERENCES", "METHODS", "TRANSLATIONS", "LANGUAGES",
    "LANGUAGE_PAIRING_MODE", "LANGUAGE_PAIRS", "PROTOCOL_ROLE",
    "MODELS", "TEMPERATURES", "REFERENCE_SET_SIZES", "PROMPT_FAMILIES",
)


def _build_config(req: RunRequest):
    """Map the request onto env vars and load_config, inheriting every
    ConfigError validation rule. The study env vars are set transiently and
    restored afterwards so concurrent requests never see each other's grid."""
    import json

    from scripture_fidelity.config import load_config

    payload = {
        "REFERENCES": json.dumps([r.model_dump() for r in req.references()]),
        "METHODS": json.dumps([req.resolved_method()]),
        "TRANSLATIONS": json.dumps([req.translation.model_dump()]),
        "LANGUAGES": json.dumps([req.language]),
        "LANGUAGE_PAIRING_MODE": req.language_pairing_mode,
        "LANGUAGE_PAIRS": json.dumps([req.language_pair]),
        "PROTOCOL_ROLE": req.protocol_role,
        "MODELS": json.dumps([req.model.model_dump()]),
        "TEMPERATURES": json.dumps([req.temperature]),
        "REFERENCE_SET_SIZES": json.dumps(req.reference_set_size),
        "PROMPT_FAMILIES": json.dumps(["method_specific"]),
    }
    saved = {k: os.environ.get(k) for k in _STUDY_ENV_VARS}
    try:
        os.environ.update(payload)
        # env_file=None: do not reload .env (it would override our payload);
        # provider credentials are already present in the process environment.
        config = load_config(env_file=None)
        config.request_id = req.request_id
        config.scenario_id = req.scenario_id
        config.protocol_version = req.protocol_version
        config.repetition = req.repetition
        config.prompt_override = req.prompt
        config.source_fixture_id = req.source_fixture_id
        config.source_document_override = req.source_document
        from scripture_fidelity.runner import validate_source_fixture

        validate_source_fixture(config)
        return config
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@app.post(
    "/v1/runs",
    dependencies=[Depends(verify_bearer)],
    response_model=RunResponse,
)
async def create_run(req: RunRequest) -> RunResponse:
    """Execute one permutation and return the full result package."""
    import time

    from scripture_fidelity.runner import run_study_in_memory

    try:
        config = _build_config(req)
    except ConfigError as e:
        # 422 Unprocessable Content: the request is well-formed JSON but the
        # study configuration it describes is invalid.
        raise HTTPException(status_code=422, detail=str(e)) from e

    started = time.monotonic()
    try:
        # run_study_in_memory is fully synchronous: it calls asyncio.run for
        # prefetch and Inspect's blocking eval, which refuses to run from a
        # thread that carries an ambient event loop. Offload it to a dedicated,
        # per-request worker thread (not asyncio.to_thread, whose shared default
        # executor keeps loop state) so the server loop is never blocked and
        # Inspect gets a clean thread.
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = await loop.run_in_executor(
                pool, run_study_in_memory, config
            )
    except Exception as e:  # noqa: BLE001 - convert to release-safe structured error
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "error_class": type(e).__name__,
                "message": "Run failed before completion; consult restricted server logs",
                "request_id": req.request_id,
                "scenario_id": req.scenario_id or None,
            },
        ) from e

    counts = result["manifest"]["counts"]
    status_str = (
        "completed_with_errors" if counts["errors"] else "completed"
    )
    return RunResponse(**{
        "status": status_str,
        "request_id": req.request_id,
        "scenario_id": req.scenario_id or None,
        "condition_requested": req.condition,
        "condition_executed": req.condition,
        "method_executed": req.resolved_method(),
        "protocol_version": req.protocol_version or None,
        "protocol_role": req.protocol_role,
        "repetition": req.repetition,
        "system_version": result["manifest"]["build_identity"]["system_version"],
        "run_id": result["run_id"],
        "duration_seconds": round(time.monotonic() - started, 3),
        "manifest": result["manifest"],
        "trials": result["trials"],
        "source_fixtures": result["source_fixtures"],
        "method_configs": result["method_configs"],
        "scoring_config": result["scoring_config"],
        "report": result["report"],
    })


def serve() -> None:
    """Console-script entry point: run the API with uvicorn.

    Host/port/workers are read from the environment (HOST, PORT, WEB_CONCURRENCY)
    so the same command works locally and on a container host like Cloud Run,
    which injects PORT.
    """
    import uvicorn

    load_dotenv()
    _require_token()  # fail fast if the bearer token is unset
    uvicorn.run(
        "scripture_fidelity.api:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8080")),
        workers=int(os.environ.get("WEB_CONCURRENCY", "1")),
    )


if __name__ == "__main__":
    serve()
