# Container image for the scripture-fidelity single-run API.
#
# Runs `scripture-fidelity-serve` (uvicorn) which reads PORT from the
# environment — matching Cloud Run, which injects PORT and expects the
# container to listen on it. Secrets (ENDPOINT_API_TOKEN and provider keys)
# are supplied by the host at runtime, never baked into the image.
FROM python:3.12-slim

# Faster, cleaner Python in containers.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install the package with the [api] extra. Copy the metadata first so the
# dependency layer is cached across source-only changes.
COPY pyproject.toml uv.lock README.md ./
COPY scripture_fidelity ./scripture_fidelity
RUN pip install uv && uv sync --frozen --no-dev --extra api

# Cloud Run ignores EXPOSE but it documents the listening port for humans.
EXPOSE 8080

# One process; scale out with Cloud Run instances rather than in-container
# workers so a single long run never blocks a sibling request on the same
# instance. Override with WEB_CONCURRENCY if you understand the tradeoff.
CMD ["scripture-fidelity-serve"]
