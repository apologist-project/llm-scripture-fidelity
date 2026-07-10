"""Rich terminal report: detail matrix, per-variant averages, pivots."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

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
    "reference",
)

_METRIC_LABELS = {
    "exact": "Exact",
    "normalized": "Norm",
    "similarity": "Sim",
    "cer": "CER",
    "verse_coverage": "Cov",
    "answered": "Ans",
    "placeholder_ok": "PlcOK",
}


def _style(metric: str, value: float) -> str:
    good = value <= 0.01 if metric == "cer" else value >= 0.99
    mid = value <= 0.10 if metric == "cer" else value >= 0.90
    color = "green" if good else "yellow" if mid else "red"
    return f"[{color}]{value:.3f}[/{color}]"


def _metrics_columns(table: Table) -> None:
    for metric in REPORT_METRICS:
        table.add_column(_METRIC_LABELS[metric], justify="right")
    table.add_column("n", justify="right", style="dim")


def _metrics_cells(means: dict[str, float], count: int) -> list[str]:
    cells = [_style(m, means.get(m, 0.0)) for m in REPORT_METRICS]
    cells.append(str(count))
    return cells


def print_report(rows: list[TrialRow], console: Console | None = None) -> None:
    console = console or Console()
    if not rows:
        console.print("[red]No scored trials found.[/red]")
        return

    # Detail matrix: one row per permutation (mean over epochs)
    detail = Table(title="Quotation fidelity — detail matrix (mean over iterations)")
    for dim in _DETAIL_DIMS:
        detail.add_column(dim.capitalize())
    _metrics_columns(detail)
    for key, means, count in aggregate(rows, _DETAIL_DIMS):
        detail.add_row(*[str(k) for k in key], *_metrics_cells(means, count))
    console.print(detail)

    # Averages for each study variant
    for dim in DIMENSIONS:
        values = {getattr(r, dim) for r in rows}
        if len(values) < 2 and dim not in ("method", "model"):
            continue
        table = Table(title=f"Averages by {dim}")
        table.add_column(dim.capitalize())
        _metrics_columns(table)
        for key, means, count in aggregate(rows, (dim,)):
            table.add_row(str(key[0]), *_metrics_cells(means, count))
        console.print(table)

    # placeholder_ok only applies to output_buffer trials
    buffer_rows = [r for r in rows if r.method == "output_buffer"]
    if buffer_rows:
        table = Table(title="Output buffer: placeholder correctness by model")
        table.add_column("Model")
        table.add_column("PlcOK", justify="right")
        table.add_column("n", justify="right", style="dim")
        for key, means, count in aggregate(buffer_rows, ("model",)):
            table.add_row(
                str(key[0]),
                _style("placeholder_ok", means.get("placeholder_ok", 0.0)),
                str(count),
            )
        console.print(table)

    # Method x model pivot on similarity
    row_vals, col_vals, cells = pivot(rows, "method", "model", "similarity")
    table = Table(title="Similarity — method × model")
    table.add_column("Method")
    for cv in col_vals:
        table.add_column(str(cv), justify="right")
    for rv in row_vals:
        table.add_row(
            str(rv),
            *[
                _style("similarity", cells[(rv, cv)]) if (rv, cv) in cells else "-"
                for cv in col_vals
            ],
        )
    console.print(table)
