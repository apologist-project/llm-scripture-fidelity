"""Command line interface for the scripture-fidelity study."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def _quiet_native_runtime_logs() -> None:
    """Suppress noisy native gRPC/Abseil INFO lines (e.g. fork_posix.cc).

    These are emitted by C++ gRPC when Google (and similar) clients are used
    under asyncio/thread pools. Must be set before those libraries initialize.
    """
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GRPC_TRACE", "")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")


def main(argv: list[str] | None = None) -> int:
    _quiet_native_runtime_logs()
    parser = argparse.ArgumentParser(
        prog="scripture-fidelity",
        description="Study of methods for LLMs to quote Scripture with high fidelity",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the study grid")
    run.add_argument("--env-file", default=None, help="Path to .env (default: ./.env)")
    run.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=1,
        help="Iterations per permutation (Inspect epochs, default 1)",
    )
    run.add_argument(
        "--results-dir", default="results", help="Base directory for run output"
    )
    run.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent model connections (default 10)",
    )
    run.add_argument(
        "--max-tasks",
        type=int,
        default=4,
        help="Max Inspect tasks running in parallel (default 4)",
    )
    run.add_argument(
        "--display",
        default="rich",
        choices=["full", "conversation", "rich", "plain", "log", "none"],
        help="Inspect progress display (default rich)",
    )
    run.add_argument(
        "--cache-dir",
        default=None,
        help="Passage cache directory (default .cache/passages)",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the permutation grid, run dependency checks, and exit "
            "without executing the study"
        ),
    )
    run.add_argument(
        "--confirm-large-run",
        action="store_true",
        help="Authorize a run whose planned call volume exceeds the threshold",
    )
    run.add_argument(
        "--retries",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Retry each model API call up to N times on rate limits and other "
            "transient provider/HTTP errors (default 5; use 0 to disable)"
        ),
    )
    for flag, help_text in [
        ("--methods", "Comma-separated subset of METHODS"),
        ("--models", "Comma-separated subset of MODELS (provider/model)"),
        ("--translations", "Comma-separated subset of TRANSLATIONS ids"),
        ("--languages", "Comma-separated subset of LANGUAGES"),
        ("--references", "Comma-separated subset of REFERENCES"),
        ("--temperatures", "Comma-separated temperatures (replaces TEMPERATURES)"),
        (
            "--set-sizes",
            "Comma-separated reference set sizes (replaces REFERENCE_SET_SIZES)",
        ),
    ]:
        run.add_argument(flag, default=None, help=help_text)

    report = sub.add_parser("report", help="Regenerate reports from an existing run")
    report.add_argument(
        "run_dir", help="Run directory (e.g. results/20260710-120000) or log dir"
    )

    bibles = sub.add_parser("list-bibles", help="List Bibles available from an API")
    bibles.add_argument(
        "--api", required=True, choices=["ao_lab", "api_bible", "esv", "youversion"]
    )
    bibles.add_argument(
        "--language", default=None, help="Filter by ISO 639-3 language (e.g. eng)"
    )
    bibles.add_argument("--env-file", default=None, help="Path to .env for API keys")

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "report":
            return _cmd_report(args)
        return _cmd_list_bibles(args)
    except KeyboardInterrupt:
        console.print("[red]Interrupted[/red]")
        return 130


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _apply_overrides(config, args):
    """Narrow (or, for temperatures, replace) config lists from CLI flags."""
    from scripture_fidelity.config import ConfigError

    def subset(name: str, items: list, selected: list[str] | None, key):
        if selected is None:
            return items
        by_key = {key(item): item for item in items}
        chosen = []
        for sel in selected:
            if sel not in by_key:
                raise ConfigError(
                    f"--{name}: {sel!r} not in configured {name} "
                    f"(available: {sorted(by_key)})"
                )
            chosen.append(by_key[sel])
        return chosen

    config.methods = subset("methods", config.methods, _csv(args.methods), str)
    config.models = subset(
        "models", config.models, _csv(args.models), lambda m: m.inspect_model
    )
    config.translations = subset(
        "translations", config.translations, _csv(args.translations), lambda t: t.id
    )
    config.languages = subset("languages", config.languages, _csv(args.languages), str)
    config.references = subset(
        "references", config.references, _csv(args.references), lambda r: r.ref
    )
    if args.temperatures is not None:
        config.temperatures = [
            None if t.casefold() in {"default", "none", "null"} else float(t)
            for t in _csv(args.temperatures)
        ]
    if args.set_sizes is not None:
        sizes = [int(s) for s in _csv(args.set_sizes)]
        if any(s < 1 for s in sizes):
            raise ConfigError("--set-sizes: sizes must be positive integers")
        config.set_sizes = sizes
    return config


def _print_grid(config, iterations: int, retries: int) -> None:
    from rich.table import Table

    from scripture_fidelity.runner import call_accounting

    table = Table(title="Study grid")
    table.add_column("Variant")
    table.add_column("Count", justify="right")
    table.add_column("Values")
    pairs = config.variant_pairs()
    for name, values in [
        ("references", [r.ref for r in config.references]),
        ("set sizes", [str(s) for s in config.set_sizes]),
        ("methods", config.methods),
        ("translations", [t.id for t in config.translations]),
        ("languages", config.languages),
        (
            f"language pairs ({config.language_pairing_mode})",
            [f"{lang}\u2192{t.id}" for lang, t in pairs],
        ),
        ("models", [m.inspect_model for m in config.models]),
        (
            "temperatures",
            [
                "provider default" if t is None else f"{t:g}"
                for t in config.temperatures
            ],
        ),
        ("prompt families", config.prompt_families),
    ]:
        table.add_row(name, str(len(values)), ", ".join(str(v) for v in values))
    console.print(table)

    permutations = config.permutation_count()
    trials = permutations * iterations
    omit_models = config.temperature_omit_models()
    # Task files are still one per temperature in TEMPERATURES; models that
    # omit temperature run in a separate null-temperature batch.
    capable_task_temps = len(config.temperatures)
    omit_task_temps = 1 if omit_models else 0
    capable_tasks = (
        len(config.methods)
        * len(pairs)
        * capable_task_temps
        * len(config.set_sizes)
        * len(config.prompt_families)
    )
    omit_tasks = (
        (
            len(config.methods)
            * len(pairs)
            * omit_task_temps
            * len(config.set_sizes)
            * len(config.prompt_families)
        )
        if omit_models
        else 0
    )
    console.print(
        f"Protocol role: [bold]{config.protocol_role}[/bold] | "
        f"Inspect tasks: [bold]{capable_tasks + omit_tasks}[/bold] "
        f"({len(config.temperature_capable_models())} models × "
        f"{capable_task_temps} temps"
        + (
            f"; {len(omit_models)} models omit temperature"
            if omit_models
            else ""
        )
        + ") | "
        f"permutations: [bold]{permutations}[/bold] | "
        f"trials (x{iterations} iterations): [bold]{trials}[/bold]"
    )

    accounting = call_accounting(config, iterations, max_retries=retries)
    console.print(
        f"Planned requests: [bold]{accounting['planned_requests']}[/bold] "
        f"({accounting['samples_per_epoch']} samples x "
        f"{accounting['epochs']} epochs) | "
        f"observations per reference: "
        f"[bold]{accounting['observations_per_reference']}[/bold] | "
        f"model-turn ceiling: "
        f"[bold]{accounting['planned_model_turn_ceiling']}[/bold] | "
        f"provider-attempt ceiling: "
        f"[bold]{accounting['max_provider_api_attempts']}[/bold] "
        f"(up to {accounting['max_http_retries_per_attempt']} "
        f"transport retries per turn)"
    )
    if omit_models:
        console.print(
            "Temperature omitted for: "
            + ", ".join(m.inspect_model for m in omit_models)
        )


def _emit_reports(log_dir: Path) -> int:
    from scripture_fidelity.report.cli_report import print_report
    from scripture_fidelity.report.data import load_rows
    from scripture_fidelity.report.html_report import (
        write_csv_reports,
        write_html_report,
    )

    rows = load_rows(log_dir)
    if not rows:
        console.print(f"[red]No scored trials found in {log_dir}[/red]")
        return 1
    print_report(rows, console)
    path = log_dir.parent / "results.html"
    write_html_report(rows, path)
    console.print(f"HTML report written to [bold]{path}[/bold]")
    csv_dir = log_dir.parent / "csv"
    csv_paths = write_csv_reports(rows, csv_dir)
    console.print(f"{len(csv_paths)} CSV tables written to [bold]{csv_dir}[/bold]")
    return 0


def _cmd_run(args) -> int:
    from scripture_fidelity.config import ConfigError, load_config
    from scripture_fidelity.runner import new_run_id, run_study

    try:
        config = _apply_overrides(load_config(args.env_file), args)
    except ConfigError as e:
        console.print(f"[red]Config error:[/red] {e}")
        return 2

    if args.retries < 0:
        console.print("[red]Config error:[/red] --retries must be >= 0")
        return 2

    _print_grid(config, args.iterations, retries=args.retries)

    from scripture_fidelity.preflight import (
        print_dependency_report,
        run_dependency_checks,
    )

    console.print(
        "Running dependency checks (live probes: first reference, "
        "highest temperature for models that support it)…"
    )
    check_failures = print_dependency_report(run_dependency_checks(config), console)

    if args.dry_run:
        if check_failures:
            console.print(
                "[yellow]Dry run finished with dependency failures; "
                "study was not executed.[/yellow]"
            )
            return 2
        console.print(
            "[yellow]Dry run: live dependency probes finished; "
            "study was not executed.[/yellow]"
        )
        return 0

    if check_failures:
        console.print(
            f"[red]Aborting run:[/red] {check_failures} dependency check(s) "
            f"failed. Fix the issues above and retry (or use --dry-run to "
            f"re-validate)."
        )
        return 2

    from scripture_fidelity.runner import CALL_VOLUME_THRESHOLD, call_accounting

    accounting = call_accounting(
        config, args.iterations, max_retries=args.retries
    )
    if (
        accounting["max_generation_attempts"] > CALL_VOLUME_THRESHOLD
        and not args.confirm_large_run
    ):
        console.print(
            f"[red]Run blocked:[/red] planned call volume "
            f"({accounting['max_provider_api_attempts']} provider attempts "
            f"including transport retries) exceeds the threshold of "
            f"{CALL_VOLUME_THRESHOLD}. Review the grid with --dry-run and "
            f"re-run with --confirm-large-run to authorize."
        )
        return 2

    run_dir = Path(args.results_dir) / new_run_id()
    console.print(f"Run directory: [bold]{run_dir}[/bold]")
    console.print(
        "Transport retries on rate-limit/transient errors: "
        f"[bold]{args.retries}[/bold]"
    )
    log_dir = run_study(
        config,
        run_dir,
        epochs=args.iterations,
        max_connections=args.concurrency,
        max_tasks=args.max_tasks,
        display=args.display,
        cache_dir=args.cache_dir,
        max_retries=args.retries,
    )
    return _emit_reports(log_dir)


def _cmd_report(args) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        console.print(f"[red]No such directory: {run_dir}[/red]")
        return 2
    # Prefer the normalized export package when present so reports are
    # recomputed from the auditable rows rather than Inspect internals.
    export_dir = run_dir / "export"
    if (export_dir / "trials.jsonl").is_file():
        from scripture_fidelity.report.cli_report import print_report
        from scripture_fidelity.report.data import rows_from_export
        from scripture_fidelity.report.html_report import (
            write_csv_reports,
            write_html_report,
        )

        rows = rows_from_export(export_dir)
        if not rows:
            console.print(f"[red]No trial rows found in {export_dir}[/red]")
            return 1
        print_report(rows, console)
        path = run_dir / "results.html"
        write_html_report(rows, path)
        console.print(f"HTML report written to [bold]{path}[/bold]")
        csv_dir = run_dir / "csv"
        csv_paths = write_csv_reports(rows, csv_dir)
        console.print(f"{len(csv_paths)} CSV tables written to [bold]{csv_dir}[/bold]")
        return 0
    log_dir = run_dir / "logs" if (run_dir / "logs").is_dir() else run_dir
    return _emit_reports(log_dir)


def _cmd_list_bibles(args) -> int:
    from dotenv import load_dotenv

    from scripture_fidelity.bible.service import get_provider

    if args.env_file:
        load_dotenv(args.env_file, override=True)
    else:
        load_dotenv()

    provider = get_provider(args.api)
    bibles = asyncio.run(provider.list_bibles(args.language))
    if not bibles:
        console.print("[yellow]No Bibles found.[/yellow]")
        return 0

    from rich.table import Table

    table = Table(title=f"Bibles available from {args.api}")
    for column in bibles[0]:
        table.add_column(column)
    for bible in bibles:
        table.add_row(*[str(v) if v is not None else "" for v in bible.values()])
    console.print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
