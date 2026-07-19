# llm-scripture-fidelity

Research study for methods of quoting Scripture with high fidelity.

The study measures how faithfully LLMs can reproduce Bible passages word for word, comparing five quotation methods across a configurable grid of scripture references, Bible translations, prompt languages, models, and sampling temperatures. It is built on [Inspect](https://inspect.aisi.org.uk/) for evaluation orchestration and scores every trial deterministically against ground-truth text fetched from Bible APIs.

## Quotation methods

| Method | How the model gets the text |
|---|---|
| `unassisted` | From its own training data — no assistance. |
| `rag` | The authoritative passage text is injected into the prompt; the model must reproduce it. |
| `tool_call` | The model is given a `get_passage` tool that fetches the exact text from a Bible API. |
| `web_search` | The model is given a `search_web` tool (Parallel.ai Search API) and must find the text on the open web. |
| `buffer_transform` | The model emits a `{{QUOTE:<reference>}}` placeholder that is programmatically replaced with the exact text in a post-generation transform. |

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/apologist-project/llm-scripture-fidelity.git
cd llm-scripture-fidelity
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

All study configuration lives in a `.env` file. Copy the example and fill in your keys:

```bash
cp .env.example .env
```

### Study variants (JSON arrays)

| Variable | Description |
|---|---|
| `REFERENCES` | Scripture references to test. Each entry is a string (`"John 3:16"`) or an object with a grouping label: `{"ref": "Psalm 117", "type": "chapter"}`. Supports single verses, ranges (`Romans 8:38-39`), cross-chapter ranges (`Luke 9:57-10:2`), and whole chapters. When `type` is omitted it is inferred (`single`/`range`/`chapter`). |
| `METHODS` | Any subset of `unassisted`, `rag`, `tool_call`, `buffer_transform`, `buffer_transform_selection`, `web_search`. |
| `TRANSLATIONS` | Bible translations. Each entry needs `id` (study-level label), `language` (ISO 639-3 of the text), `api` (which provider to use), and `api_bible_id` (the provider-specific identifier). Research runs should also declare `rights`, `verification`, `edition`, `license_basis`, and `public_release`; complete source provenance is mandatory for `confirmatory` runs. |
| `LANGUAGES` | Available prompt languages. In `matched` mode only declared `LANGUAGE_PAIRS` run; a full cross-product requires an explicitly exploratory `crossed` configuration. |
| `MODELS` | Models as `{"provider": ..., "model": ...}`. Set `"supports_temperature": false` for endpoints that reject the parameter. Providers map to Inspect prefixes: `openai`, `anthropic`, `google`, `together`, `xai` (mapped to Inspect's `grok` provider), and `mockllm` (for testing without API calls). |
| `TEMPERATURES` | Sampling temperatures, e.g. `[0.0, 0.7]`. Use `[null]` to omit temperature and use the provider default. |
| `REFERENCE_SET_SIZES` | Optional (default `[1]`). Reference set sizes, e.g. `[1, 3]`. For each size > 1 the references list is chunked (in order) into sets of that size, and each set becomes a single prompt asking for all of its passages at once — probing whether models handle every requested reference (e.g. calling `get_passage` once per reference). Size 1 reproduces standard single-reference samples. |

The run grid combines reference sets, methods, declared language-translation pairs, models, and temperatures. A full languages × translations cross-product runs only in explicitly exploratory `crossed` mode.

### Bible API providers

The `api` field of each translation selects the provider:

| `api` value | Service | Key required |
|---|---|---|
| `ao_lab` | [AO Lab Free Use Bible API](https://bible.helloao.org/) | None |
| `api_bible` | [API.Bible](https://scripture.api.bible/) | `API_BIBLE_API_KEY` |
| `youversion` | [YouVersion Platform](https://platform.youversion.com/) | `YOUVERSION_API_KEY` |

Discover provider-specific translation IDs with:

```bash
scripture-fidelity list-bibles --api ao_lab --language eng
scripture-fidelity list-bibles --api api_bible --language zho
```

### API keys

Set only the keys you need in `.env` (never commit it — it is gitignored):

- Model providers: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `TOGETHER_API_KEY`, `XAI_API_KEY`
- Bible APIs: `API_BIBLE_API_KEY`, `YOUVERSION_API_KEY`
- Web search: `PARALLEL_API_KEY` (required only when `web_search` is in `METHODS`)

## Running the study

Always start with a dry run to see the grid size and estimated call count before spending tokens:

```bash
scripture-fidelity run --dry-run
```

Then run the full study (or a subset):

```bash
# Full grid, 1 iteration each, terminal + HTML report
scripture-fidelity run

# 3 iterations per permutation, for measuring variance
scripture-fidelity run -n 3

# Narrow the grid without editing .env
scripture-fidelity run \
  --methods rag,tool_call \
  --models openai/gpt-4o-mini \
  --translations BSB \
  --references "John 3:16,Psalm 117" \
  --temperatures 0.0,0.7
```

`python -m scripture_fidelity` works identically to the `scripture-fidelity` entry point.

### `run` options

| Flag | Default | Description |
|---|---|---|
| `--env-file PATH` | `./.env` | Alternate config file. |
| `-n, --iterations N` | `1` | Iterations per permutation (Inspect epochs), for measuring variance. |
| `--results-dir DIR` | `results` | Base directory; each run gets a timestamped subdirectory. |
| `--concurrency N` | `10` | Max concurrent model connections. |
| `--max-tasks N` | `4` | Max Inspect tasks running in parallel. |
| `--display full\|conversation\|rich\|plain\|log\|none` | `rich` | Inspect progress display. |
| `--cache-dir DIR` | `.cache/passages` | Passage cache location. |
| `--dry-run` | | Print the grid and exit without any model calls. |
| `--methods`, `--models`, `--translations`, `--languages`, `--references` | | Comma-separated subsets of the configured lists. |
| `--temperatures` | | Comma-separated values that *replace* the configured list. |
| `--set-sizes` | | Comma-separated reference set sizes that *replace* `REFERENCE_SET_SIZES` (e.g. `1,3`). |

Each run prints the report to the terminal and writes to `results/<timestamp>/`:

- `config.json` — the exact resolved configuration for reproducibility
- `logs/` — Inspect eval logs (one per task × model), inspectable with `inspect view`
- `results.html` — HTML report
- `csv/` — one CSV per report table (mirrors the HTML tables), for spreadsheets

### Regenerating reports

Reports can be rebuilt from a past run's logs without re-running anything:

```bash
scripture-fidelity report results/20260710-212847
```

## Research API (local or hosted single run)

Besides the CLI, the study can be served locally or remotely as an authenticated HTTP endpoint that
executes **one** permutation per request and returns the full result package.
Unlike the CLI, an API run is ephemeral: nothing is written to `results/`, and
the caller owns the returned JSON.

```bash
pip install -e ".[api]"
export ENDPOINT_API_TOKEN="$(openssl rand -hex 32)"   # plus your provider/Bible keys
scripture-fidelity-serve                              # listens on PORT (default 8080)
```

- `GET /healthz` — unauthenticated liveness check.
- `GET /version` — unauthenticated immutable build, schema, commit, dependency-lock, and prompt-template identity.
- `POST /v1/runs` — bearer-authenticated (`Authorization: Bearer $ENDPOINT_API_TOKEN`). `ENDPOINT_API_KEY` is accepted as a backwards-compatible local alias.
  The body is one permutation and may identify either a low-level `method` or
  a protocol `condition`. Research callers can supply stable request/scenario
  IDs, an exact user prompt, protocol version, repetition number, and a
  verified source fixture. The response echoes those fields and includes the
  full result package with per-reference trial records and provenance.

The versioned collaboration contract, examples, and release boundaries are in
[docs/RESEARCH_API.md](docs/RESEARCH_API.md); committed JSON Schemas are in
[`schemas/`](schemas/). The runtime models live in
`scripture_fidelity/api.py`. For
optional containerization and hosting (Google Cloud Run, with a GitHub Actions
auto-deploy on push to `main`, plus Render/VM alternatives), see
[docs/DEPLOY.md](docs/DEPLOY.md).

## Reports and metrics

Both the terminal (Rich) and HTML reports contain:

- **Detail matrix** — one row per permutation (mean over iterations)
- **Averages by variant** — per method, model, translation, language, temperature, reference, and reference type
- **Placeholder correctness** — `buffer_transform` trials only
- **Method × model similarity pivot**

The HTML report is a self-contained file with color-coded cells and click-to-sort columns.

All metrics are deterministic string comparisons (no LLM judge). The quoted passage is extracted from the model's `<quote>...</quote>` tags (falling back to the whole completion), then compared to ground truth:

| Metric | Meaning |
|---|---|
| `exact` | Verbatim match (whitespace-collapsed). |
| `normalized` | Match after Unicode normalization, quote/dash unification, casefolding, and punctuation removal — catches "right text, wrong typography". |
| `similarity` | Normalized Levenshtein similarity (0–1) on the normalized strings. |
| `cer` | Character error rate (0 is perfect). For Chinese, comparison is done with whitespace removed. |
| `verse_coverage` | Fraction of ground-truth verses appearing verbatim (normalized) in the answer. |
| `answered` | Whether the model produced any quote at all. |
| `placeholder_ok` | For `buffer_transform`: whether the model emitted exactly one well-formed placeholder per requested reference. |
| `tool_used` | For `tool_call`/`web_search`: whether the model actually invoked its assigned tool (`get_passage`/`search_web`). For multi-reference `tool_call` samples this is the *coverage*: the fraction of requested references actually looked up via `get_passage` (one call for three references scores 0.33). |

Observed text fidelity is never overwritten when a method instruction is disobeyed. `method_adherence` records tool/placeholder compliance, while `end_to_end_exact` is the conjunction of final-output exactness and applicable selection, lookup, replacement, and adherence components. This preserves the difference between quoting accurately from memory and successfully executing a tool-mediated condition.

### Multi-reference samples

With `REFERENCE_SET_SIZES` sizes > 1, a single prompt asks for several passages and requires one attributed quote block per passage: `<quote ref="John 3:16">...</quote>`. Scoring extracts each reference's quote by its `ref` attribute (falling back to positional order for unattributed blocks) and compares it to that reference's own ground truth. Exports retain each reference's answer and metrics plus separate request-level means and conjunctions, so a dropped passage is attributable rather than represented only by an aggregate. For `buffer_transform`, one `{{QUOTE:<ref>}}` placeholder per reference is required. `web_search` adherence stays at "tool invoked at least once", since one search can legitimately cover several passages.

## Caching

Ground-truth passages are cached on disk (`.cache/passages` by default), so repeated runs do not re-hit the Bible APIs. The same fetched text serves as the RAG context, the `get_passage` tool output, the buffer-transform replacement source, and the scoring ground truth — guaranteeing a consistent baseline across methods. Delete the cache directory to force re-fetching.

## Testing

```bash
pytest
```

The suite covers reference parsing, scoring/normalization, the buffer-transform, and configuration validation. For an end-to-end pipeline check without spending tokens, use the mock provider:

```bash
scripture-fidelity run --models mockllm/model --translations BSB \
  --references "John 3:16" --results-dir results-smoke
```

(requires `{"provider": "mockllm", "model": "model"}` in your `MODELS` list).

## Project structure

```
scripture_fidelity/
  cli.py            # run / report / list-bibles subcommands
  config.py         # .env parsing and validation
  references.py     # "John 3:16" -> canonical USFM reference
  prompts.py        # per-language prompt templates for each method
  solvers.py        # Inspect solvers/tools for the five methods
  task.py           # Inspect task factory (one task per variant)
  runner.py         # grid expansion, ground-truth prefetch, eval execution
  scoring.py        # metrics + Inspect scorer
  bible/            # ao_lab / api_bible / youversion clients + disk cache
  report/           # terminal (Rich) and HTML (Jinja2) reports
tests/              # unit tests
```
