# Deploying the single-run API

The `scripture-fidelity-serve` command runs an authenticated HTTP API that
executes **one** study permutation per request and returns the full result
package (the same data the CLI writes to `export/`, plus the recomputed
report). Nothing is persisted server-side — the caller owns the response.

- Endpoints: `POST /v1/runs` (bearer auth), `GET /healthz`, and
  `GET /version` (no auth).
- Auth: a bearer token from `ENDPOINT_API_TOKEN`; `ENDPOINT_API_KEY` is
  accepted as a compatibility alias.
- Recommended host: **Google Cloud Run** (scales to zero, managed HTTPS,
  first-class secrets). Render or a plain VM work with the same image.

## 1. Run locally

```bash
pip install -e ".[api]"
export ENDPOINT_API_TOKEN="$(openssl rand -hex 32)"
# plus the provider/Bible keys you normally put in .env
scripture-fidelity-serve            # listens on PORT (default 8080)
```

Smoke test (in another shell):

```bash
curl -s localhost:8080/healthz
curl -s -X POST localhost:8080/v1/runs \
  -H "Authorization: Bearer $ENDPOINT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @request.json | jq .status
```

See the bottom of this file for an example `request.json`.

## 2. Build the container

The repo ships a `Dockerfile`. Cloud Run can build it for you (step 3), or
build locally to test:

```bash
docker build -t scripture-fidelity-api .
docker run --rm -p 8080:8080 \
  -e ENDPOINT_API_TOKEN=dev-token \
  -e OPENAI_API_KEY=... -e ANTHROPIC_API_KEY=... \
  scripture-fidelity-api
```

## 3. Deploy to Google Cloud Run

### One-time setup

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com
```

### Store secrets in Secret Manager

Keep tokens/keys out of the image and out of `gcloud` history. Create one
secret per key (repeat for every provider/Bible key you use):

```bash
printf '%s' "$(openssl rand -hex 32)" \
  | gcloud secrets create ENDPOINT_API_TOKEN --data-file=-
printf '%s' "sk-..." \
  | gcloud secrets create OPENAI_API_KEY --data-file=-
```

### Deploy from source

Cloud Run builds the Dockerfile and deploys in one command:

```bash
gcloud run deploy scripture-fidelity-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --timeout 900 \
  --cpu 1 --memory 1Gi \
  --no-cpu-throttling \
  --max-instances 5 \
  --set-env-vars SOURCE_COMMIT="$(git rev-parse HEAD)",SOURCE_REPOSITORY_URL="$(git remote get-url origin)" \
  --set-secrets ENDPOINT_API_TOKEN=ENDPOINT_API_TOKEN:latest,\
OPENAI_API_KEY=OPENAI_API_KEY:latest,\
ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest
```

Notes on the flags:

- `--allow-unauthenticated` exposes the URL publicly; your **own** bearer
  token (`ENDPOINT_API_TOKEN`) is the access control. To additionally gate it
  behind Google IAM, drop this flag and call with an identity token.
- `--timeout 900` (15 min) covers a slow single run (tool/web-search methods
  with retries). Cloud Run's max is 3600s; raise if you need it.
- `--no-cpu-throttling` keeps the CPU running mid-request so the eval isn't
  frozen between network waits. `--cpu 1 --memory 1Gi` is a fine start.
- `--max-instances` caps cost/concurrency. Each run is CPU-light but holds a
  connection for its duration.

The command prints a `Service URL`. Test it:

```bash
SERVICE_URL=$(gcloud run services describe scripture-fidelity-api \
  --region us-central1 --format 'value(status.url)')
curl -s "$SERVICE_URL/healthz"
curl -s "$SERVICE_URL/version" | jq
```

## 4. Automated deploys (GitHub Actions)

`.github/workflows/deploy.yml` builds and deploys on every push to `main`
(skipping docs/test-only changes). It authenticates with **Workload Identity
Federation** — keyless, so no service-account JSON key is ever stored in
GitHub. Do this one-time setup, then pushes deploy themselves.

### Create a deployer service account

```bash
PROJECT_ID=YOUR_PROJECT_ID
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format 'value(projectNumber)')

gcloud iam service-accounts create gh-deployer \
  --project "$PROJECT_ID" --display-name "GitHub Actions deployer"
DEPLOY_SA="gh-deployer@$PROJECT_ID.iam.gserviceaccount.com"

# Roles needed to build from source and deploy to Cloud Run.
for ROLE in roles/run.admin roles/cloudbuild.builds.editor \
            roles/artifactregistry.admin roles/storage.admin \
            roles/iam.serviceAccountUser roles/secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member "serviceAccount:$DEPLOY_SA" --role "$ROLE"
done
```

### Create the Workload Identity pool + provider

```bash
gcloud iam workload-identity-pools create github \
  --project "$PROJECT_ID" --location global --display-name "GitHub"

gcloud iam workload-identity-pools providers create-oidc github \
  --project "$PROJECT_ID" --location global \
  --workload-identity-pool github --display-name "GitHub OIDC" \
  --issuer-uri "https://token.actions.githubusercontent.com" \
  --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition "assertion.repository=='apologist-project/llm-scripture-fidelity'"

# Let only this repo impersonate the deployer service account.
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA" \
  --project "$PROJECT_ID" --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/apologist-project/llm-scripture-fidelity"

# The full provider resource name the workflow needs:
echo "projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/github"
```

### Configure the repo

In **Settings → Secrets and variables → Actions → Variables**, add these
repository **variables** (not secrets — none are sensitive):

| Variable | Value |
|---|---|
| `GCP_PROJECT_ID` | your project id |
| `GCP_REGION` | e.g. `us-central1` |
| `GCP_SERVICE` | e.g. `scripture-fidelity-api` |
| `GCP_WIF_PROVIDER` | the `projects/.../providers/github` string printed above |
| `GCP_DEPLOY_SA` | `gh-deployer@PROJECT.iam.gserviceaccount.com` |
| `GCP_RUN_SECRETS` | `ENDPOINT_API_TOKEN=ENDPOINT_API_TOKEN:latest,OPENAI_API_KEY=OPENAI_API_KEY:latest,...` |

The Secret Manager secrets referenced by `GCP_RUN_SECRETS` must already exist
(step 2 above). After that, push to `main` — or run the workflow manually from
the **Actions** tab (`workflow_dispatch`) — and it deploys, printing the
service URL in the job log.

## 5. Alternatives

- **Render**: New → Web Service → connect the repo (it detects the
  Dockerfile) → add the same env vars in the dashboard → deploy. Simplest
  click-through; keeps a small instance warm rather than scaling to zero.
- **VM (DigitalOcean/Lightsail/Hetzner)**: run the image behind Caddy or
  nginx for TLS. Most control, no request-duration limits, flat cost.

All three use the identical image, so you are not locked in.

## 6. Request / response shape

The complete contract is in [RESEARCH_API.md](RESEARCH_API.md), with committed
JSON Schemas in `schemas/`. Minimal request:

```json
{
  "reference": {"ref": "John 3:16", "type": "well_known_single"},
  "condition": "native_parametric_quote",
  "translation": {"id": "BSB", "name": "Berean Standard Bible",
                  "language": "eng", "api": "ao_lab",
                  "api_bible_id": "BSB", "rights": "open",
                  "edition": "current API edition",
                  "license_basis": "provider terms",
                  "public_release": true},
  "language": "eng",
  "language_pairing_mode": "matched",
  "language_pair": ["eng", "BSB"],
  "model": {"provider": "openai", "model": "gpt-4-turbo",
            "supports_temperature": true},
  "temperature": 0.25,
  "reference_set_size": [1],
  "request_id": "fide-pilot-001",
  "scenario_id": "explicit-single-john-3-16-bsb",
  "protocol_version": "fid-056-pilot-v1",
  "protocol_role": "diagnostic",
  "repetition": 1
}
```

`reference` may be an array for multi-reference requests. Each API request has
exactly one `reference_set_size`, and that value must equal the number of
references. Use `temperature: null` with `supports_temperature: false` when a
provider must choose its own default. The `200` response contains the echoed
research identity plus the full package: `manifest`, per-reference `trials`,
`source_fixtures`, `method_configs`, `scoring_config`, and `report`.

Status codes: `401` (bad/missing token), `422` (invalid study config),
`502` (ground-truth prefetch/provider failure before the run). Error responses
contain a bounded error class and correlation IDs, not raw traces or secrets.
