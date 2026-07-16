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

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from scripture_fidelity.config import ConfigError

TOKEN_ENV_VAR = "ENDPOINT_API_TOKEN"


class ReferenceModel(BaseModel):
    ref: str
    type: str | None = None
    description: str = ""


class TranslationModel(BaseModel):
    id: str
    language: str
    api: str
    api_bible_id: str
    name: str = ""
    rights: str = "unknown"
    verification: str = ""


class ModelModel(BaseModel):
    provider: str
    model: str


class RunRequest(BaseModel):
    """A single study permutation. ``reference`` accepts one object or a list
    (a list makes multi-reference set sizes such as [1, 3] meaningful)."""

    reference: Union[ReferenceModel, list[ReferenceModel]]
    method: str
    translation: TranslationModel
    language: str = "eng"
    language_pairing_mode: str = "matched"
    language_pair: list[str]
    model: ModelModel
    temperature: float = 0.25
    reference_set_size: list[int] = Field(default_factory=lambda: [1, 3])

    def references(self) -> list[ReferenceModel]:
        r = self.reference
        return r if isinstance(r, list) else [r]


def _require_token() -> str:
    token = os.environ.get(TOKEN_ENV_VAR, "").strip()
    if not token:
        raise RuntimeError(
            f"{TOKEN_ENV_VAR} is not set; the API refuses to start without a "
            "bearer token. Set it in the environment or .env."
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


# Study dimensions that map from the request payload to load_config env vars.
# Provider/Bible-API credentials are read from the process environment (.env)
# and are deliberately not part of the request.
_STUDY_ENV_VARS = (
    "REFERENCES", "METHODS", "TRANSLATIONS", "LANGUAGES",
    "LANGUAGE_PAIRING_MODE", "LANGUAGE_PAIRS", "PROTOCOL_ROLE",
    "MODELS", "TEMPERATURES", "REFERENCE_SET_SIZES",
)


def _build_config(req: RunRequest):
    """Map the request onto env vars and load_config, inheriting every
    ConfigError validation rule. The study env vars are set transiently and
    restored afterwards so concurrent requests never see each other's grid."""
    import json

    from scripture_fidelity.config import load_config

    payload = {
        "REFERENCES": json.dumps([r.model_dump() for r in req.references()]),
        "METHODS": json.dumps([req.method]),
        "TRANSLATIONS": json.dumps([req.translation.model_dump()]),
        "LANGUAGES": json.dumps([req.language]),
        "LANGUAGE_PAIRING_MODE": req.language_pairing_mode,
        "LANGUAGE_PAIRS": json.dumps([req.language_pair]),
        # diagnostic is the role that permits a multi-valued set-size grid
        # (e.g. [1, 3]) while keeping crossed pairing disabled by default.
        "PROTOCOL_ROLE": "diagnostic",
        "MODELS": json.dumps([req.model.model_dump()]),
        "TEMPERATURES": json.dumps([req.temperature]),
        "REFERENCE_SET_SIZES": json.dumps(req.reference_set_size),
    }
    saved = {k: os.environ.get(k) for k in _STUDY_ENV_VARS}
    try:
        os.environ.update(payload)
        # env_file=None: do not reload .env (it would override our payload);
        # provider credentials are already present in the process environment.
        return load_config(env_file=None)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@app.post("/v1/runs", dependencies=[Depends(verify_bearer)])
async def create_run(req: RunRequest) -> dict:
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
    except Exception as e:  # noqa: BLE001 - surface prefetch/provider failures
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Run failed before completion: {e}",
        ) from e

    counts = result["manifest"]["counts"]
    status_str = (
        "completed_with_errors" if counts["errors"] else "completed"
    )
    return {
        "status": status_str,
        "run_id": result["run_id"],
        "duration_seconds": round(time.monotonic() - started, 3),
        "manifest": result["manifest"],
        "trials": result["trials"],
        "source_fixtures": result["source_fixtures"],
        "method_configs": result["method_configs"],
        "scoring_config": result["scoring_config"],
        "report": result["report"],
    }


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
