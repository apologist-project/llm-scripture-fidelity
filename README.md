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
| `output_buffer` | The model emits a `{{QUOTE:<reference>}}` placeholder that is programmatically replaced with the exact text in a post-generation transform. |

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
| `METHODS` | Any subset of `unassisted`, `rag`, `tool_call`, `output_buffer`, `web_search`. |
| `TRANSLATIONS` | Bible translations. Each entry needs `id` (study-level label), `language` (ISO 639-3 of the text), `api` (which provider to use), and `api_bible_id` (the provider-specific identifier). Optional `name` for display. |
| `LANGUAGES` | Prompt languages, crossed with every translation. Prompt templates exist for `eng`, `zho`, `spa`, `fra`, `deu`, `hin`, `ara`, `por`, `urd`, `rus`, and `ben`. |
| `MODELS` | Models as `{"provider": ..., "model": ...}`. Providers map to Inspect prefixes: `openai`, `anthropic`, `google`, `together`, `grok`, and `mockllm` (for testing without API calls). |
| `TEMPERATURES` | Sampling temperatures, e.g. `[0.0, 0.7]`. |

The run grid is the full cross product: references × methods × translations × languages × models × temperatures.

### Bible API providers

The `api` field of each translation selects the provider:

| `api` value | Service | Key required |
|---|---|---|
| `ao_lab` | [AO Lab Free Use Bible API](https://bible.helloao.org/) | None |
| `api_bible` | [API.Bible](https://scripture.api.bible/) | `API_BIBLE_API_KEY` |
| `youversion` | [YouVersion Platform](https://platform.youversion.com/) | `YOUVERSION_APP_KEY` |

Discover provider-specific translation IDs with:

```bash
scripture-fidelity list-bibles --api ao_lab --language eng
scripture-fidelity list-bibles --api api_bible --language zho
```

### API keys

Set only the keys you need in `.env` (never commit it — it is gitignored):

- Model providers: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `TOGETHER_API_KEY`, `XAI_API_KEY`
- Bible APIs: `API_BIBLE_API_KEY`, `YOUVERSION_APP_KEY`
- Web search: `PARALLEL_API_KEY` (required only when `web_search` is in `METHODS`)

## Running the study

Always start with a dry run to see the grid size and estimated call count before spending tokens:

```bash
scripture-fidelity run --dry-run
```

Then run the full study (or a subset):

```bash
# Full grid, 1 iteration each, terminal report
scripture-fidelity run

# 3 iterations per permutation, terminal + HTML report
scripture-fidelity run -n 3 --output both

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
| `--output cli\|html\|both\|none` | `cli` | Report target after the run. |
| `--html-file PATH` | `<run dir>/report.html` | HTML report location. |
| `--results-dir DIR` | `results` | Base directory; each run gets a timestamped subdirectory. |
| `--concurrency N` | `10` | Max concurrent model connections. |
| `--max-tasks N` | `4` | Max Inspect tasks running in parallel. |
| `--display full\|conversation\|rich\|plain\|log\|none` | `rich` | Inspect progress display. |
| `--cache-dir DIR` | `.cache/passages` | Passage cache location. |
| `--dry-run` | | Print the grid and exit without any model calls. |
| `--methods`, `--models`, `--translations`, `--languages`, `--references` | | Comma-separated subsets of the configured lists. |
| `--temperatures` | | Comma-separated values that *replace* the configured list. |

Each run writes to `results/<timestamp>/`:

- `config.json` — the exact resolved configuration for reproducibility
- `logs/` — Inspect eval logs (one per task × model), inspectable with `inspect view`

### Regenerating reports

Reports can be rebuilt from a past run's logs without re-running anything:

```bash
scripture-fidelity report results/20260710-212847
scripture-fidelity report results/20260710-212847 --output html --html-file report.html
```

## Reports and metrics

Both the terminal (Rich) and HTML reports contain:

- **Detail matrix** — one row per permutation (mean over iterations)
- **Averages by variant** — per method, model, translation, language, temperature, reference, and reference type
- **Placeholder correctness** — `output_buffer` trials only
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
| `placeholder_ok` | For `output_buffer`: whether the model emitted exactly one well-formed placeholder with the correct reference. |

## Caching

Ground-truth passages are cached on disk (`.cache/passages` by default), so repeated runs do not re-hit the Bible APIs. The same fetched text serves as the RAG context, the `get_passage` tool output, the output-buffer replacement source, and the scoring ground truth — guaranteeing a consistent baseline across methods. Delete the cache directory to force re-fetching.

## Testing

```bash
pytest
```

The suite covers reference parsing, scoring/normalization, the output-buffer transform, and configuration validation. For an end-to-end pipeline check without spending tokens, use the mock provider:

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
