"""Command line interface for the scripture-fidelity study."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scripture-fidelity",
        description="Study of methods for LLMs to quote Scripture with high fidelity",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the study grid")
    run.add_argument("--env-file", default=None, help="Path to .env (default: ./.env)")
    run.add_argument(
        "-n", "--iterations", type=int, default=1,
        help="Iterations per permutation (Inspect epochs, default 1)",
    )
    run.add_argument(
        "--output", choices=["cli", "html", "both", "none"], default="cli",
        help="Report target after the run (default cli)",
    )
    run.add_argument("--html-file", default=None, help="HTML report output path")
    run.add_argument(
        "--results-dir", default="results", help="Base directory for run output"
    )
    run.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent model connections (default 10)",
    )
    run.add_argument(
        "--max-tasks", type=int, default=4,
        help="Max Inspect tasks running in parallel (default 4)",
    )
    run.add_argument(
        "--display", default="rich",
        choices=["full", "conversation", "rich", "plain", "log", "none"],
        help="Inspect progress display (default rich)",
    )
    run.add_argument(
        "--cache-dir", default=None, help="Passage cache directory (default .cache/passages)"
    )
    run.add_argument(
        "--dry-run", action="store_true",
        help="Print the permutation grid and exit without calling any model",
    )
    for flag, help_text in [
        ("--methods", "Comma-separated subset of METHODS"),
        ("--models", "Comma-separated subset of MODELS (provider/model)"),
        ("--translations", "Comma-separated subset of TRANSLATIONS ids"),
        ("--languages", "Comma-separated subset of LANGUAGES"),
        ("--references", "Comma-separated subset of REFERENCES"),
        ("--temperatures", "Comma-separated temperatures (replaces TEMPERATURES)"),
    ]:
        run.add_argument(flag, default=None, help=help_text)

    report = sub.add_parser("report", help="Regenerate reports from an existing run")
    report.add_argument(
        "run_dir", help="Run directory (e.g. results/20260710-120000) or log dir"
    )
    report.add_argument(
        "--output", choices=["cli", "html", "both"], default="cli",
        help="Report target (default cli)",
    )
    report.add_argument("--html-file", default=None, help="HTML report output path")

    bibles = sub.add_parser("list-bibles", help="List Bibles available from an API")
    bibles.add_argument(
        "--api", required=True, choices=["helloao", "api_bible", "youversion"]
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
        config.temperatures = [float(t) for t in _csv(args.temperatures)]
    return config


def _print_grid(config, iterations: int) -> None:
    from rich.table import Table

    table = Table(title="Study grid")
    table.add_column("Variant")
    table.add_column("Count", justify="right")
    table.add_column("Values")
    for name, values in [
        ("references", [r.ref for r in config.references]),
        ("methods", config.methods),
        ("translations", [t.id for t in config.translations]),
        ("languages", config.languages),
        ("models", [m.inspect_model for m in config.models]),
        ("temperatures", [f"{t:g}" for t in config.temperatures]),
    ]:
        table.add_row(name, str(len(values)), ", ".join(str(v) for v in values))
    console.print(table)

    permutations = config.permutation_count()
    trials = permutations * iterations
    tasks = (
        len(config.methods)
        * len(config.translations)
        * len(config.languages)
        * len(config.temperatures)
    )
    console.print(
        f"Inspect tasks: [bold]{tasks}[/bold] "
        f"(x {len(config.models)} models) | "
        f"permutations: [bold]{permutations}[/bold] | "
        f"trials (x{iterations} iterations): [bold]{trials}[/bold]"
    )


def _emit_reports(log_dir: Path, output: str, html_file: str | None) -> int:
    from scripture_fidelity.report.data import load_rows

    rows = load_rows(log_dir)
    if not rows:
        console.print(f"[red]No scored trials found in {log_dir}[/red]")
        return 1
    if output in ("cli", "both"):
        from scripture_fidelity.report.cli_report import print_report

        print_report(rows, console)
    if output in ("html", "both"):
        from scripture_fidelity.report.html_report import write_html_report

        path = Path(html_file) if html_file else log_dir.parent / "report.html"
        write_html_report(rows, path)
        console.print(f"HTML report written to [bold]{path}[/bold]")
    return 0


def _cmd_run(args) -> int:
    from scripture_fidelity.config import ConfigError, load_config
    from scripture_fidelity.runner import new_run_id, run_study

    try:
        config = _apply_overrides(load_config(args.env_file), args)
    except ConfigError as e:
        console.print(f"[red]Config error:[/red] {e}")
        return 2

    _print_grid(config, args.iterations)
    if args.dry_run:
        console.print("[yellow]Dry run: no model calls made.[/yellow]")
        return 0

    import os

    if "web_search" in config.methods and not os.environ.get("PARALLEL_API_KEY"):
        console.print(
            "[red]Config error:[/red] PARALLEL_API_KEY is not set "
            "(required for the web_search method)"
        )
        return 2

    run_dir = Path(args.results_dir) / new_run_id()
    console.print(f"Run directory: [bold]{run_dir}[/bold]")
    log_dir = run_study(
        config,
        run_dir,
        epochs=args.iterations,
        max_connections=args.concurrency,
        max_tasks=args.max_tasks,
        display=args.display,
        cache_dir=args.cache_dir,
    )
    if args.output == "none":
        console.print(f"Logs written to [bold]{log_dir}[/bold]")
        return 0
    return _emit_reports(log_dir, args.output, args.html_file)


def _cmd_report(args) -> int:
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        console.print(f"[red]No such directory: {run_dir}[/red]")
        return 2
    log_dir = run_dir / "logs" if (run_dir / "logs").is_dir() else run_dir
    return _emit_reports(log_dir, args.output, args.html_file)


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
