"""Load Inspect eval logs into flat trial rows and aggregate them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from scripture_fidelity.task import SCORER_NAME

# Dimensions a report can group by (attribute names on TrialRow)
DIMENSIONS = [
    "model",
    "method",
    "translation",
    "language",
    "temperature",
    "set_size",
    "reference",
    "ref_type",
]

# Metrics shown in reports, in display order
REPORT_METRICS = ["exact", "normalized", "similarity", "cer", "verse_coverage"]


@dataclass
class TrialRow:
    model: str
    method: str
    translation: str
    language: str  # prompt language
    temperature: float
    reference: str
    ref_type: str
    epoch: int
    set_size: int = 1
    metrics: dict[str, float] = field(default_factory=dict)
    answer: str = ""

    def key(self, dims: tuple[str, ...]) -> tuple:
        return tuple(getattr(self, d) for d in dims)


def load_rows(log_dir: str | Path) -> list[TrialRow]:
    """Flatten all eval logs in a directory into per-trial rows."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    rows: list[TrialRow] = []
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info)
        model = str(log.eval.model)
        for sample in log.samples or []:
            md = sample.metadata or {}
            scores = sample.scores or {}
            score = scores.get(SCORER_NAME)
            if score is None or not isinstance(score.value, dict):
                score = next(
                    (s for s in scores.values() if isinstance(s.value, dict)), None
                )
            if score is None:
                continue
            rows.append(
                TrialRow(
                    model=model,
                    method=md.get("method", "?"),
                    translation=md.get("translation", "?"),
                    language=md.get("prompt_language", "?"),
                    temperature=float(md.get("temperature", 0.0)),
                    reference=md.get("reference", str(sample.id)),
                    ref_type=md.get("ref_type", "?"),
                    epoch=sample.epoch,
                    set_size=int(md.get("set_size", 1)),
                    metrics={k: float(v) for k, v in score.value.items()},
                    answer=score.answer or "",
                )
            )
    return rows


def aggregate(
    rows: list[TrialRow], dims: tuple[str, ...]
) -> list[tuple[tuple, dict[str, float], int]]:
    """Mean of every metric grouped by ``dims``.

    Returns a sorted list of (key, mean_metrics, trial_count).
    """
    groups: dict[tuple, list[TrialRow]] = {}
    for row in rows:
        groups.setdefault(row.key(dims), []).append(row)

    result = []
    for key in sorted(groups, key=lambda k: tuple(str(p) for p in k)):
        members = groups[key]
        keys = set()
        for row in members:
            keys.update(row.metrics)
        means = {
            k: sum(r.metrics.get(k, 0.0) for r in members) / len(members)
            for k in sorted(keys)
        }
        result.append((key, means, len(members)))
    return result


def pivot(
    rows: list[TrialRow], row_dim: str, col_dim: str, metric: str
) -> tuple[list, list, dict[tuple, float]]:
    """Mean of ``metric`` cross-tabulated by two dimensions."""
    row_vals = sorted({getattr(r, row_dim) for r in rows}, key=str)
    col_vals = sorted({getattr(r, col_dim) for r in rows}, key=str)
    cells: dict[tuple, float] = {}
    for rv in row_vals:
        for cv in col_vals:
            members = [
                r.metrics.get(metric, 0.0)
                for r in rows
                if getattr(r, row_dim) == rv and getattr(r, col_dim) == cv
            ]
            if members:
                cells[(rv, cv)] = sum(members) / len(members)
    return row_vals, col_vals, cells
