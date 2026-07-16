"""Self-contained HTML report generation (Jinja2)."""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from scripture_fidelity.report.data import (
    DIMENSIONS,
    REPORT_METRICS,
    TrialRow,
    aggregate,
    pivot,
)

_DETAIL_DIMS = (
    "model",
    "method",
    "translation",
    "language",
    "temperature",
    "set_size",
    "reference",
)

_METRIC_LABELS = {
    "exact": "Exact",
    "normalized": "Normalized",
    "similarity": "Similarity",
    "cer": "CER",
    "verse_coverage": "Coverage",
    "placeholder_ok": "Placeholder OK",
    "tool_used": "Tool used",
    "final_output_exact": "Final output exact",
    "method_adherence": "Method adherence",
    "end_to_end_exact": "End-to-end exact",
}


def _cell(metric: str, value: float) -> dict:
    good = value <= 0.01 if metric == "cer" else value >= 0.99
    mid = value <= 0.10 if metric == "cer" else value >= 0.90
    cls = "good" if good else "mid" if mid else "bad"
    return {"text": f"{value:.3f}", "cls": f"num {cls}", "sort": value}


def _plain(text) -> dict:
    return {"text": str(text), "cls": "", "sort": str(text)}


def _metric_cells(means: dict[str, float], count: int) -> list[dict]:
    cells = [_cell(m, means.get(m, 0.0)) for m in REPORT_METRICS]
    cells.append({"text": str(count), "cls": "num dim", "sort": count})
    return cells


def build_sections(rows: list[TrialRow]) -> list[dict]:
    metric_headers = [_METRIC_LABELS[m] for m in REPORT_METRICS] + ["n"]
    sections = []

    detail_rows = [
        [_plain(k) for k in key] + _metric_cells(means, count)
        for key, means, count in aggregate(rows, _DETAIL_DIMS)
    ]
    sections.append(
        {
            "title": "Detail matrix (mean over iterations)",
            "headers": [d.capitalize() for d in _DETAIL_DIMS] + metric_headers,
            "rows": detail_rows,
        }
    )

    for dim in DIMENSIONS:
        values = {getattr(r, dim) for r in rows}
        if len(values) < 2 and dim not in ("method", "model"):
            continue
        sections.append(
            {
                "title": f"Averages by {dim}",
                "headers": [dim.capitalize()] + metric_headers,
                "rows": [
                    [_plain(key[0])] + _metric_cells(means, count)
                    for key, means, count in aggregate(rows, (dim,))
                ],
            }
        )

    buffer_rows = [r for r in rows if r.method == "buffer_transform"]
    if buffer_rows:
        sections.append(
            {
                "title": "Buffer transform: placeholder correctness by model",
                "headers": ["Model", _METRIC_LABELS["placeholder_ok"], "n"],
                "rows": [
                    [
                        _plain(key[0]),
                        _cell("placeholder_ok", means.get("placeholder_ok", 0.0)),
                        {"text": str(count), "cls": "num dim", "sort": count},
                    ]
                    for key, means, count in aggregate(buffer_rows, ("model",))
                ],
            }
        )

    tool_rows = [r for r in rows if r.method in ("tool_call", "web_search")]
    if tool_rows:
        sections.append(
            {
                "title": "Tool adherence: assigned tool invoked, by model",
                "headers": ["Model", "Method", _METRIC_LABELS["tool_used"], "n"],
                "rows": [
                    [
                        _plain(key[0]),
                        _plain(key[1]),
                        _cell("tool_used", means.get("tool_used", 0.0)),
                        {"text": str(count), "cls": "num dim", "sort": count},
                    ]
                    for key, means, count in aggregate(tool_rows, ("model", "method"))
                ],
            }
        )

    row_vals, col_vals, cells = pivot(rows, "method", "model", "similarity")
    sections.append(
        {
            "title": "Similarity — method × model",
            "headers": ["Method"] + [str(c) for c in col_vals],
            "rows": [
                [_plain(rv)]
                + [
                    _cell("similarity", cells[(rv, cv)])
                    if (rv, cv) in cells
                    else _plain("-")
                    for cv in col_vals
                ]
                for rv in row_vals
            ],
        }
    )
    return sections


def write_html_report(
    rows: list[TrialRow], output_path: Path, title: str = "Scripture Quotation Fidelity"
) -> Path:
    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("template.html")
    html = template.render(
        title=title,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        trial_count=len(rows),
        sections=build_sections(rows),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _slug(title: str) -> str:
    """Filesystem-safe filename stem derived from a section title."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "table"


def write_csv_reports(rows: list[TrialRow], output_dir: Path) -> list[Path]:
    """Write one CSV per report table into ``output_dir``.

    The CSVs mirror the HTML report exactly: each table in :func:`build_sections`
    becomes ``<slug>.csv`` with the same header row and the plain text of each
    cell (no color classes). Returns the paths written, in section order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sections = build_sections(rows)
    written: list[Path] = []
    used: dict[str, int] = {}
    for section in sections:
        stem = _slug(section["title"])
        # Disambiguate the rare case of two sections slugging to the same name.
        used[stem] = used.get(stem, 0) + 1
        if used[stem] > 1:
            stem = f"{stem}-{used[stem]}"
        path = output_dir / f"{stem}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(section["headers"])
            for row in section["rows"]:
                writer.writerow([cell["text"] for cell in row])
        written.append(path)

    # Guard against drift between the HTML and CSV outputs: both are generated
    # from build_sections(), so a CSV per table means one file per HTML table.
    if len(written) != len(sections):
        raise RuntimeError(
            f"CSV/table mismatch: wrote {len(written)} CSV files for "
            f"{len(sections)} report tables (filename collision?)."
        )
    return written
