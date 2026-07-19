# Research API Contract

This document defines the integration boundary between an independent research
caller and a hosted Scripture-fidelity implementation. It lets research teams
execute preregistered scenarios while implementation operators retain their
provider credentials, source agreements, and operational infrastructure.

The contract supports reproducible measurement. It is not a certification,
leaderboard, product endorsement, or permission to release restricted text.

## Version and discovery

`GET /version` requires no authentication and returns the deployed commit,
dependency-lock hash, schema version, prompt-template version, and supported
condition-to-method mappings. Capture this response once before and once after
each run batch. A changed identity means the batch spans multiple system
versions and must be analyzed separately or rerun.

Runtime OpenAPI is available at `GET /openapi.json`. Stable request and response
schemas are committed under `schemas/` and regenerated with:

```bash
uv run python scripts/export_api_schemas.py
```

## Authentication and retention

Send `Authorization: Bearer <token>` to `POST /v1/runs`. The server reads the
token from `ENDPOINT_API_TOKEN` or the compatibility alias `ENDPOINT_API_KEY`.
Credentials are deployment configuration and never appear in requests,
responses, logs intended for release, or committed fixtures.

API execution uses a temporary passage cache and writes no persistent result
directory. The caller owns and must securely retain the returned package. This
does not imply that upstream model or Bible API providers have no retention;
their applicable terms remain in force.

## Conditions

| Protocol condition | Executed method | Research interpretation |
|---|---|---|
| `native_parametric_quote` | `unassisted` | Model produces text without an assigned source tool. |
| `source_supplied_quote` | `rag` | Authoritative source text is supplied in context. |
| `tool_call_quote` | `tool_call` | Model may invoke a Bible retrieval tool. |
| `web_search_quote` | `web_search` | Model may invoke the configured web search tool. |
| `deterministic_render_given_reference` | `buffer_transform` | Requested reference is given; a deterministic layer renders source text. |
| `reference_token_then_replace` | `buffer_transform_selection` | Model selects a reference token; a deterministic layer retrieves and renders it. |

Prefer `condition` in preregistered requests. `method` remains available for
implementation diagnostics. If both are sent, they must represent the mapping
above. The response echoes `condition_requested`, `condition_executed`, and
`method_executed`; analysis must use the executed fields rather than infer the
condition from output quality.

Condition-specific tool, source-use, and placeholder instructions are applied
in the system layer. A caller-supplied user prompt therefore remains identical
across conditions while the harness changes only the assigned architecture.

## Caller-controlled research fields

- `request_id`: stable correlation identifier chosen by the caller. It is not
  an idempotency key; retry attempts should use distinct IDs or be tracked in a
  caller-side attempt table.
- `scenario_id`: stable scenario identifier shared with the preregistration.
- `protocol_version`: immutable protocol/run-plan version.
- `protocol_role`: `diagnostic`, `confirmatory`, `robustness`, or
  `exploratory`. Confirmatory requests require complete source provenance.
- `repetition`: one-based repeat index.
- `prompt`: exact caller-supplied user prompt. When omitted, the repository's
  versioned prompt template generates the prompt.
- `source_fixture_id`: expected authoritative fixture identity. The server
  rejects a mismatch before model execution.
- `source_document`: caller-supplied source text for a one-reference
  `source_supplied_quote` request. When combined with `prompt`, the caller's
  exact user request is preserved and the document is added through the pinned
  harness context wrapper.

The export records the caller-prompt hash, effective model-input hash, and
whether the prompt was caller-supplied. API runs
delete their temporary Inspect logs, so callers must retain their submitted
prompt registry; generated prompts are recoverable from the pinned template
version and inputs. Raw and final outputs are returned to the caller with
hashes. Restricted-source fixtures are hash/metadata-only and never include
source text in the exported package.

## Provider-aware generation controls

`temperature` accepts a number or `null`. Set the model field
`supports_temperature` to `false` and send `temperature: null` for endpoints
that reject the parameter or expose no caller-controlled temperature. A null
value means provider default, not temperature zero. Record model aliases only
when the provider documents their resolution; the response separately records
requested and resolved model identifiers where Inspect exposes both.

## Response and analysis unit

The API returns one request package. `trials` contains one row per requested
reference, so multi-reference prompts produce multiple rows sharing one
`request_id` and distinct `trial_id` values. `request_metrics` preserves the
request-level aggregate. Analyses must specify whether their unit is a request
or a reference and account for clustering when one request contains multiple
references.

Each response includes:

- build, dependency, prompt-template, timing, and execution provenance;
- requested and resolved model identity when available;
- source fixture IDs, retrieval timestamps, hashes, rights, verification mode,
  edition, license basis, and public-release metadata;
- raw/final output hashes, selected-reference fields, lookup fixture identity,
  failure tags, structured errors, usage, retry, and timing fields;
- deterministic text metrics plus method adherence and end-to-end exactness.

## Error handling

`401` indicates authentication failure. `422` indicates request or study-config
validation failure. `502` indicates a prefetch, provider, or execution failure.
Release-safe errors include a bounded class, generic message, `request_id`, and
`scenario_id`; server logs remain restricted. A failed request is data about
operability but is not automatically a quotation error. The run plan must state
retry and missingness rules before confirmatory execution.

## Required preflight

1. Pin and record the deployment commit and `/version` response.
2. Validate all requests against the committed request schema.
3. Confirm source edition, rights, verification, and release metadata.
4. Run a small integration pilot across each condition and provider family.
5. Verify caller IDs, prompt hashes, fixture hashes, selected references,
   per-reference rows, and structured failure behavior.
6. Freeze the scenario set and analysis plan before the confirmatory run.

## Publication boundary

The endpoint implementation, private credentials, partner traces, held-out
prompts, and restricted source text remain internal to their respective
organizations. Research teams may publish protocol materials, public-domain
scenarios, scoring definitions, aggregate results, and reviewed release-safe
artifacts under the participating organizations' written agreement. Product
claims and implementation details require separate approval and are not implied
by this contract.
